#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
레이더 속도 추정 유틸리티
- 투영/차선 판정/대표점 선택
- 속도 필터링 및 트랙 상태 갱신
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np

# 투영된 레이더 포인트를 객체 bbox에 소유권 할당
def assign_points_to_bboxes(
    proj_uvs: np.ndarray,
    proj_valid: np.ndarray,
    objects: Iterable[Dict[str, object]],
) -> np.ndarray:

    if proj_uvs is None or proj_valid is None:
        return np.zeros(0, dtype=int)

    num_pts = len(proj_uvs)
    point_owner = np.full(num_pts, -1, dtype=int)
    if num_pts == 0:
        return point_owner

    objs = list(objects)
    if not objs:
        return point_owner

    valid_idxs = np.where(proj_valid)[0]
    if valid_idxs.size == 0:
        return point_owner

    for i in valid_idxs:
        u = float(proj_uvs[i, 0])
        v = float(proj_uvs[i, 1])

        candidates = []
        for obj in objs:
            g_id = int(obj["id"])
            x1, y1, x2, y2 = obj["bbox"]
            if not (x1 <= u <= x2 and y1 <= v <= y2):
                continue

            ref_u = 0.5 * (x1 + x2)
            ref_v = float(y2)
            diag = float(np.hypot(max(1, x2 - x1), max(1, y2 - y1)))
            norm_dist = float(np.hypot(u - ref_u, v - ref_v) / max(diag, 1.0))
            candidates.append((norm_dist, -float(y2), g_id))

        if not candidates:
            continue

        candidates.sort(key=lambda t: (t[0], t[1]))
        point_owner[i] = candidates[0][2]

    return point_owner

# 3D 포인트(N, 3)를 2D 픽셀(N, 2)로 투영
def project_points(K, R, t, points_3d):
    if points_3d.shape[0] == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=bool)

    P = K @ np.hstack([R, t.reshape(3, 1)])
    pts_h = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    proj = (P @ pts_h.T).T

    z = proj[:, 2]
    valid = z > 1e-4
    u = proj[:, 0] / np.maximum(z, 1e-9)
    v = proj[:, 1] / np.maximum(z, 1e-9)

    uv = np.vstack([u, v]).T
    return uv, valid


# 정렬된 1D 값을 간격(max_gap) 기준으로 군집화
def cluster_1d_by_gap(values: np.ndarray, max_gap: float):
    if values is None or len(values) == 0:
        return []

    values = np.sort(values)
    clusters = []
    start = values[0]
    last = values[0]
    cluster = [values[0]]

    for v in values[1:]:
        if (v - last) <= max_gap:
            cluster.append(v)
        else:
            clusters.append(cluster)
            cluster = [v]
        last = v

    clusters.append(cluster)
    return clusters


# 가장 전방 클러스터를 선택해 도플러 중앙값 계산
def select_best_forward_cluster(forward_vals: np.ndarray, dopplers: np.ndarray, max_gap_m: float = 2.0):
    clusters = cluster_1d_by_gap(forward_vals, max_gap=max_gap_m)
    if not clusters:
        return None

    means = [np.mean(c) for c in clusters]
    best_idx = int(np.argmax(means))
    best_cluster = clusters[best_idx]

    mask = np.isin(forward_vals, best_cluster)
    best_dop = dopplers[mask]

    if best_dop.size == 0:
        return None
    return float(np.median(best_dop))


# 속도 스칼라 신호를 평활화하는 1차 칼만 필터
class ScalarKalman:
    def __init__(self, x0=0.0, P0=25.0, Q=2.0, R=9.0):
        self.x = float(x0)
        self.P = float(P0)
        self.Q = float(Q)
        self.R = float(R)

    def predict(self):
        self.P = self.P + self.Q
        return self.x

    def update(self, z: float):
        z = float(z)
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1.0 - K) * self.P
        return self.x


# PointCloud2 메시지에서 xyz와 도플러 컬럼 파싱
def parse_radar_pointcloud(radar_msg) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        from sensor_msgs import point_cloud2 as pc2
    except Exception:
        return None, None

    try:
        field_names = [f.name for f in radar_msg.fields]
        if not {"x", "y", "z"}.issubset(field_names):
            return None, None

        doppler_candidates = ("doppler", "doppler_mps", "velocity", "vel", "radial_velocity")
        power_candidates = ("power", "intensity", "rcs")

        doppler_field = next((f for f in doppler_candidates if f in field_names), None)
        power_field = next((f for f in power_candidates if f in field_names), None)

        target_fields = ["x", "y", "z"]
        if power_field:
            target_fields.append(power_field)
        if doppler_field:
            target_fields.append(doppler_field)

        g = pc2.read_points(radar_msg, field_names=tuple(target_fields), skip_nans=True)
        radar_points_all = np.array(list(g))

        if radar_points_all.shape[0] > 0:
            radar_points = radar_points_all[:, :3]
            col_idx = 3
            radar_doppler = None

            if power_field:
                col_idx += 1
            if doppler_field:
                radar_doppler = radar_points_all[:, col_idx]
        else:
            radar_points, radar_doppler = None, None
    except Exception:
        return None, None

    return radar_points, radar_doppler


# 단일 2D 포인트가 포함되는 차선 이름 반환
def find_target_lane(point_xy: Tuple[int, int], active_lane_polys: Dict[str, np.ndarray]) -> Optional[str]:
    for name, poly in active_lane_polys.items():
        if cv2.pointPolygonTest(poly, point_xy, False) >= 0:
            return name
    return None


# bbox 후보 지점들을 이용해 차선 포함 여부 판정
def find_target_lane_from_bbox(
    bbox: Tuple[int, int, int, int],
    active_lane_polys: Dict[str, np.ndarray],
) -> Optional[str]:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    m = 6

    # 하단 중심, 중심, 하단 좌/우 순 차선 포함 여부 검사
    candidates = [
        (cx, y2),
        (cx, cy),
        (x1 + m, max(y1, y2 - m)),
        (x2 - m, max(y1, y2 - m)),
    ]

    for pt in candidates:
        lane = find_target_lane(pt, active_lane_polys)
        if lane is not None:
            return lane
    return None


# 객체별 차선 경로/로컬 번호/표시 라벨 상태 갱신
def update_lane_tracking(
    g_id: int,
    target_lane: Optional[str],
    vehicle_lane_paths: Dict[int, str],
    lane_counters: Dict[str, int],
    global_to_local_ids: Dict[int, int],
) -> Tuple[str, str, int, str]:
    curr_lane_str = str(target_lane) if target_lane else 'None'

    if g_id not in vehicle_lane_paths:
        vehicle_lane_paths[g_id] = curr_lane_str
    else:
        path_parts = vehicle_lane_paths[g_id].split("->")
        last_lane = path_parts[-1]

        if curr_lane_str != last_lane and curr_lane_str != 'None':
            vehicle_lane_paths[g_id] += f"->{curr_lane_str}"

    lane_path = vehicle_lane_paths[g_id]

    if target_lane and g_id not in global_to_local_ids:
        lane_counters[target_lane] += 1
        global_to_local_ids[g_id] = lane_counters[target_lane]

    local_no = global_to_local_ids.get(g_id, g_id)

    if "->" in lane_path:
        label = f"No: {local_no} ({lane_path})"
    else:
        label = f"No: {local_no} ({target_lane})" if target_lane else f"ID: {g_id}"

    return curr_lane_str, lane_path, local_no, label

# 속도 매칭용 bbox 기준점(top/center/bottom) 계산
def get_bbox_reference_point(
    bbox: Tuple[int, int, int, int],
    mode: str = "center",
) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    mode = (mode or "center").lower()
    if mode in {"bottom", "bottom_center", "bottom-center"}:
        return int((x1 + x2) / 2), int(y2)
    if mode in {"top", "top_center", "top-center"}:
        return int((x1 + x2) / 2), int(y1)
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

# bbox 내부 레이더 포인트에서 대표점/속도/품질점수 계산
def select_representative_point(
    proj_uvs: np.ndarray,
    proj_valid: np.ndarray,
    dop_raw: np.ndarray,
    bbox: Tuple[int, int, int, int],
    pts_raw: Optional[np.ndarray],
    vel_gate_mps: float,
    rep_pt_mem: Dict[int, Iterable[float]],
    pt_alpha: float,
    now_ts: float,
    g_id: int,
    min_speed_kmh: float,
    ref_mode: str = "center",
    trim_ratio: float = 0.2,
) -> Tuple[Optional[float], Optional[Tuple[int, int]], Optional[float], float, float]:
    x1, y1, x2, y2 = bbox
    meas_kmh = None
    rep_pt = None
    rep_vel_raw = None
    score = 0.0
    radar_dist = -1.0

    # bbox 내부 포인트 선별
    in_box = (
        proj_valid
        & (proj_uvs[:, 0] >= x1)
        & (proj_uvs[:, 0] <= x2)
        & (proj_uvs[:, 1] >= y1)
        & (proj_uvs[:, 1] <= y2)
    )
    idxs = np.where(in_box)[0]

    if idxs.size == 0:
        return meas_kmh, rep_pt, rep_vel_raw, score, radar_dist

    # 도플러 이상치 제거를 위한 게이팅
    med_dop = np.median(dop_raw[idxs])
    vel_mask = np.abs(dop_raw[idxs] - med_dop) < vel_gate_mps
    valid_idxs = idxs[vel_mask] if np.any(vel_mask) else idxs

    target_idxs = valid_idxs

    # 목표 기준점은 bbox 기준점 + 직전 프레임 메모리 보정
    t_u, t_v = get_bbox_reference_point(bbox, ref_mode)
    if g_id in rep_pt_mem:
        prev_u, prev_v, last_ts = rep_pt_mem[g_id]
        if now_ts - last_ts < 1.0:
            t_u, t_v = prev_u, prev_v

    if len(target_idxs) == 0:
        return meas_kmh, rep_pt, rep_vel_raw, score, radar_dist

    # 목표 기준점과 가까운 상위 k개 포인트로 대표점 생성
    current_pts = proj_uvs[target_idxs]
    dists = np.hypot(current_pts[:, 0] - t_u, current_pts[:, 1] - t_v)

    k = min(len(dists), 3)
    sorted_indices = np.argsort(dists)
    closest_indices = sorted_indices[:k]

    best_pts = current_pts[closest_indices]
    meas_u = np.mean(best_pts[:, 0])
    meas_v = np.mean(best_pts[:, 1])

    if pts_raw is not None:
        best_3d_indices = target_idxs[closest_indices]
        best_3d_pts = pts_raw[best_3d_indices]
        best_dists_3d = np.linalg.norm(best_3d_pts, axis=1)
        radar_dist = np.mean(best_dists_3d)

    # 대표 속도는 트리밍 가능한 중앙값 사용
    vel_samples = dop_raw[target_idxs[closest_indices]]
    if trim_ratio > 0.0 and vel_samples.size >= 5:
        k = int(np.floor(vel_samples.size * trim_ratio))
        vel_sorted = np.sort(vel_samples)
        vel_samples = vel_sorted[k:vel_sorted.size - k] if (vel_sorted.size - 2 * k) > 0 else vel_sorted
    rep_vel_raw = np.median(vel_samples) * 3.6

    # 대표점 시계열은 EMA로 완화
    if g_id in rep_pt_mem:
        prev_u, prev_v, last_ts = rep_pt_mem[g_id]
        if now_ts - last_ts < 1.0:
            smooth_u = pt_alpha * meas_u + (1.0 - pt_alpha) * prev_u
            smooth_v = pt_alpha * meas_v + (1.0 - pt_alpha) * prev_v
        else:
            smooth_u, smooth_v = meas_u, meas_v
    else:
        smooth_u, smooth_v = meas_u, meas_v

    rep_pt_mem[g_id] = [smooth_u, smooth_v, now_ts]
    rep_pt = (int(smooth_u), int(smooth_v))

    if abs(rep_vel_raw) >= min_speed_kmh:
        meas_kmh = abs(rep_vel_raw)

    # bbox 기준점 대비 픽셀 오차를 0~100 점수로 환산
    target_pt = get_bbox_reference_point(bbox, ref_mode)
    dist_px = np.hypot(target_pt[0] - rep_pt[0], target_pt[1] - rep_pt[1])
    diag_len = np.hypot(x2 - x1, y2 - y1)
    err_ratio = dist_px / max(1, diag_len)
    score = max(0, min(100, 100 - (err_ratio * 200)))

    return meas_kmh, rep_pt, rep_vel_raw, score, radar_dist


# 측정 속도와 트랙 메모리를 결합해 최종 속도/점수 추정
def update_speed_estimate(
    g_id: int,
    meas_kmh: Optional[float],
    score: float,
    now_ts: float,
    vel_memory: Dict[int, Dict[str, float]],
    kf_vel: Dict[int, ScalarKalman],
    kf_score: Dict[int, ScalarKalman],
    hold_sec: float = 0.6,
    decay_sec: float = 0.8,
    decay_floor_kmh: float = 1.0,
    rate_limit_kmh_per_s: float = 12.0,
) -> Tuple[Optional[float], float]:
    if meas_kmh is not None:
        if abs(meas_kmh) > 120.0:
            meas_kmh = None

    mem = vel_memory.get(g_id, {})
    last_vel = mem.get("vel_kmh", 0.0)
    last_ts = mem.get("ts", now_ts)
    fail_cnt = mem.get("fail_count", 0)

    # 급격한 속도 점프는 연속 실패 카운트 기반으로 차단
    JUMP_THRESHOLD = 15.0
    if meas_kmh is not None and last_vel > 5.0:
        diff = abs(meas_kmh - last_vel)
        if diff > JUMP_THRESHOLD:
            fail_cnt += 1
            if fail_cnt > 5:
                fail_cnt = 0
            else:
                meas_kmh = None
        else:
            fail_cnt = 0
    else:
        fail_cnt = 0

    # 트랙별 칼만 필터 초기화
    if g_id not in kf_vel:
        kf_vel[g_id] = ScalarKalman(x0=meas_kmh if meas_kmh is not None else 0.0, Q=0.8, R=5.0)
        kf_score[g_id] = ScalarKalman(x0=score if score > 0 else 80.0, Q=0.001, R=500.0)

    kf_v = kf_vel[g_id]
    kf_s = kf_score[g_id]

    v_pred = kf_v.predict()
    s_pred = kf_s.predict()

    # 프레임 간 속도 변화율 제한
    def _rate_limit(prev_vel: float, new_vel: float, dt: float) -> float:
        if dt <= 0.0 or rate_limit_kmh_per_s <= 0.0:
            return new_vel
        max_delta = rate_limit_kmh_per_s * dt
        delta = float(np.clip(new_vel - prev_vel, -max_delta, max_delta))
        return prev_vel + delta
    
    # 측정값이 있으면 업데이트, 없으면 홀드/감쇠 정책 적용
    if meas_kmh is not None:
        if abs(meas_kmh) < 1:
            meas_kmh = 0.0

        vel_out = kf_v.update(meas_kmh)
        vel_out = _rate_limit(last_vel, vel_out, max(0.0, now_ts - last_ts))
        score_out = kf_s.update(score)

        vel_memory[g_id] = {
            "vel_kmh": vel_out,
            "ts": now_ts,
            "fail_count": fail_cnt,
            "score": score_out
        }
    else:
        time_gap = now_ts - last_ts

        if time_gap <= hold_sec:
            vel_out = v_pred
        elif time_gap <= hold_sec + decay_sec:
            ratio = (hold_sec + decay_sec - time_gap) / max(decay_sec, 1e-6)
            vel_out = v_pred * max(0.0, ratio)
        else:
            vel_out = 0.0

        if vel_out < decay_floor_kmh:
            vel_out = 0.0

        if vel_out < decay_floor_kmh:
            vel_out = 0.0
        vel_out = _rate_limit(last_vel, vel_out, max(0.0, now_ts - last_ts))
        
        score_out = s_pred * 0.998

        mem["fail_count"] = fail_cnt
        vel_memory[g_id] = mem

    # 신뢰도 하한 미달 시 결과 무효화
    if score_out < 10:
        score_out = 0
        vel_out = None

    return vel_out, score_out


# CSV 저장용 단일 레코드 딕셔너리 생성
def build_record_row(
    now_ts: float,
    g_id: int,
    local_no: int,
    lane_path: str,
    curr_lane_str: str,
    radar_dist: float,
    meas_kmh: Optional[float],
    vel_out: Optional[float],
    bbox: Tuple[int, int, int, int],
    target_pt: Tuple[int, int],
    cluster_pt: Optional[Tuple[int, int]],
) -> Dict[str, float]:
    x1, y1, x2, y2 = bbox
    bbox_w = float(max(1, x2 - x1))
    bbox_h = float(max(1, y2 - y1))

    ref_u, ref_v = float(target_pt[0]), float(target_pt[1])
    matched = int(cluster_pt is not None)
    proj_u = float(cluster_pt[0]) if cluster_pt is not None else np.nan
    proj_v = float(cluster_pt[1]) if cluster_pt is not None else np.nan

    du = (proj_u - ref_u) if matched else np.nan
    dv = (proj_v - ref_v) if matched else np.nan
    pixel_err = float(np.hypot(du, dv)) if matched else np.nan
    ru = (du / bbox_w) if matched else np.nan
    rv = (dv / bbox_h) if matched else np.nan

    return {
        "Time": now_ts,
        "Global_ID": g_id,
        "Local_ID": local_no,
        "Lane_Path": lane_path,
        "Final_Lane": curr_lane_str,
        "Unique_Key": f"{local_no}({lane_path})",
        "Radar_Dist": radar_dist,
        "Radar_Vel": meas_kmh if meas_kmh is not None else np.nan,
        "Track_Vel": vel_out if vel_out is not None else np.nan,
        "Score": score,
        "BBox_W": bbox_w,
        "BBox_H": bbox_h,
        "Ref_U": ref_u,
        "Ref_V": ref_v,
        "Proj_U": proj_u,
        "Proj_V": proj_v,
        "Delta_U": du,
        "Delta_V": dv,
        "Pixel_Error": pixel_err,
        "RU": ru,
        "RV": rv,
        "Is_Matched": matched,
    }