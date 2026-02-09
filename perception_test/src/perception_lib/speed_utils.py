#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np


def project_points(K, R, t, points_3d):
    """
    3D Points (N,3) -> 2D Pixels (N,2) projection
    """
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


class ScalarKalman:
    '''속도(스칼라)만 필터링하는 간단한 칼만 필터.'''
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


def find_target_lane(point_xy: Tuple[int, int], active_lane_polys: Dict[str, np.ndarray]) -> Optional[str]:
    for name, poly in active_lane_polys.items():
        if cv2.pointPolygonTest(poly, point_xy, False) >= 0:
            return name
    return None


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

    med_dop = np.median(dop_raw[idxs])
    vel_mask = np.abs(dop_raw[idxs] - med_dop) < vel_gate_mps
    valid_idxs = idxs[vel_mask] if np.any(vel_mask) else idxs

    target_idxs = valid_idxs

    t_u, t_v = get_bbox_reference_point(bbox, ref_mode)
    if g_id in rep_pt_mem:
        prev_u, prev_v, last_ts = rep_pt_mem[g_id]
        if now_ts - last_ts < 1.0:
            t_u, t_v = prev_u, prev_v

    if len(target_idxs) == 0:
        return meas_kmh, rep_pt, rep_vel_raw, score, radar_dist

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

    vel_samples = dop_raw[target_idxs[closest_indices]]
    if trim_ratio > 0.0 and vel_samples.size >= 5:
        k = int(np.floor(vel_samples.size * trim_ratio))
        vel_sorted = np.sort(vel_samples)
        vel_samples = vel_sorted[k:vel_sorted.size - k] if (vel_sorted.size - 2 * k) > 0 else vel_sorted
    rep_vel_raw = np.median(vel_samples) * 3.6

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

    target_pt = get_bbox_reference_point(bbox, ref_mode)
    dist_px = np.hypot(target_pt[0] - rep_pt[0], target_pt[1] - rep_pt[1])
    diag_len = np.hypot(x2 - x1, y2 - y1)
    err_ratio = dist_px / max(1, diag_len)
    score = max(0, min(100, 100 - (err_ratio * 200)))

    return meas_kmh, rep_pt, rep_vel_raw, score, radar_dist


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
    decay_floor_kmh: float = 5.0,
    rate_limit_kmh_per_s: float = 12.0,
) -> Tuple[Optional[float], float]:
    if meas_kmh is not None:
        if abs(meas_kmh) > 120.0:
            meas_kmh = None

    mem = vel_memory.get(g_id, {})
    last_vel = mem.get("vel_kmh", 0.0)
    last_ts = mem.get("ts", now_ts)
    fail_cnt = mem.get("fail_count", 0)

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

    if g_id not in kf_vel:
        kf_vel[g_id] = ScalarKalman(x0=meas_kmh if meas_kmh is not None else 0.0, Q=0.8, R=5.0)
        kf_score[g_id] = ScalarKalman(x0=score if score > 0 else 80.0, Q=0.001, R=500.0)

    kf_v = kf_vel[g_id]
    kf_s = kf_score[g_id]

    v_pred = kf_v.predict()
    s_pred = kf_s.predict()

    def _rate_limit(prev_vel: float, new_vel: float, dt: float) -> float:
        if dt <= 0.0 or rate_limit_kmh_per_s <= 0.0:
            return new_vel
        max_delta = rate_limit_kmh_per_s * dt
        delta = float(np.clip(new_vel - prev_vel, -max_delta, max_delta))
        return prev_vel + delta
    
    if meas_kmh is not None:
        if abs(meas_kmh) < 4.0:
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

    if score_out < 10:
        score_out = 0
        vel_out = None

    return vel_out, score_out


def build_record_row(
    now_ts: float,
    g_id: int,
    local_no: int,
    lane_path: str,
    curr_lane_str: str,
    radar_dist: float,
    meas_kmh: Optional[float],
    vel_out: Optional[float],
    score: float,
) -> Dict[str, float]:
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
        "Score": score
    }
