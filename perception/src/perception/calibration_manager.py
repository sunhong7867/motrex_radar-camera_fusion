#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
calibration_manager.py  (CARLA 캘리브레이션 파이프라인 로직 기반)

역할
- detection(bbox) + radar_history(최근 N프레임 레이더 점들)로부터
  (3D radar centroid) <-> (2D image point) 대응점을 누적 수집
- outlier 제거(z-score) + 가까운 클러스터 선택 + centroid 추출
- solvePnP 2-stage: SQPnP(전역) + ITERATIVE(정밀)로 extrinsic 추정 :contentReference[oaicite:3]{index=3}
- extrinsic.json 저장 + calibration log 저장

현장 팁
- REQUIRED_SAMPLES 늘릴수록 안정하지만, 수집 시간이 늘어남
- MIN_CALIB_DIST는 너무 가까우면(특히 10~15m 이하) 오차가 커져서 20~30m 권장
"""

import csv
import datetime
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

from . import perception_utils


# ==============================================================================
# [설정] 캘리브레이션 필터 파라미터
# ==============================================================================
MIN_CALIB_DIST = 23.0     # 너무 가까운 물체는 제외 권장
REQUIRED_SAMPLES = 10     # 최소 샘플 수
DUPLICATE_THR_PX = 50     # 2D 중복 방지(픽셀 거리)
# ==============================================================================

_BUFFER_3D: List[np.ndarray] = []
_BUFFER_2D: List[List[float]] = []


def reset_calibration_buffer():
    global _BUFFER_3D, _BUFFER_2D
    _BUFFER_3D = []
    _BUFFER_2D = []
    print("[CALIB] Buffer cleared.")


@dataclass
class CalibrationResult:
    R: np.ndarray
    t: np.ndarray
    reproj_error_px: float
    inliers: np.ndarray
    method: str = "Unknown"


def _reprojection_error(
    obj_pts: np.ndarray, img_pts: np.ndarray,
    K: np.ndarray, D: Optional[np.ndarray],
    rvec: np.ndarray, tvec: np.ndarray
) -> float:
    if cv2 is None:
        return float("nan")
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)  # :contentReference[oaicite:4]{index=4}
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(proj - img_pts.reshape(-1, 2), axis=1)
    return float(np.mean(err))


def _filter_outliers_zscore(points: np.ndarray, threshold: float = 2.0) -> np.ndarray:
    if len(points) < 5:
        return points
    mean = np.mean(points, axis=0)
    std = np.std(points, axis=0)
    std[std == 0] = 1e-6
    z_scores = np.abs((points - mean) / std)
    valid_mask = (z_scores < threshold).all(axis=1)
    if np.sum(valid_mask) < 3:
        return points
    return points[valid_mask]


def save_calibration_log(result: CalibrationResult, total_pts: int, pts_3d: np.ndarray, pts_2d: np.ndarray):
    base_dir = "calibration_logs"
    os.makedirs(base_dir, exist_ok=True)

    history_path = os.path.join(base_dir, "calibration_history.csv")
    file_exists = os.path.isfile(history_path)

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tx, ty, tz = result.t.flatten()
    count_inliers = 0 if result.inliers is None else len(result.inliers)

    with open(history_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Timestamp", "Method", "Total_Samples", "Inliers_Count",
                "Reproj_Error_px", "Tx_m", "Ty_m", "Tz_m"
            ])
        writer.writerow([
            now_str, result.method, total_pts, count_inliers,
            f"{result.reproj_error_px:.4f}",
            f"{tx:.4f}", f"{ty:.4f}", f"{tz:.4f}"
        ])

    print(f"[CALIB] Log saved: {history_path} (Method: {result.method})")


def save_extrinsic_json(path: str, R: np.ndarray, t: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> None:
    extr = perception_utils.ExtrinsicRT(R=R, t=t.reshape(3, 1))
    perception_utils.save_extrinsic_json(path, extr, meta=meta)


def load_extrinsic_json(path: str) -> Tuple[np.ndarray, np.ndarray]:
    extr = perception_utils.load_extrinsic_json(path)
    return extr.R, extr.t.reshape(3, 1)


def run_calibration_pipeline(
    detections: List[Dict],                 # [{"bbox":(x1,y1,x2,y2), ...}, ...]
    radar_history: List[np.ndarray],        # [ (Mi,4/5) ... ] 최근 N프레임
    K: np.ndarray,
    width: int,
    height: int,
    output_path: str = "extrinsic.json"
) -> Tuple[bool, str]:
    """
    return: (success, message)
    """

    global _BUFFER_3D, _BUFFER_2D

    if cv2 is None:
        return False, "OpenCV(cv2) missing."

    if not detections:
        return False, "No detections found."
    if radar_history is None or len(radar_history) == 0:
        return False, "No radar history."

    # [1] 축 정렬 (CARLA->OpenCV 좌표계 맞추기)
    R_align = np.array([
        [0.0, 1.0, 0.0],    # Rad Y -> Cam X
        [0.0, 0.0, -1.0],   # Rad Z -> Cam -Y
        [1.0, 0.0, 0.0]     # Rad X -> Cam Z
    ], dtype=np.float64)

    # [2] 초기값: output_path가 있으면 그걸로 시작, 없으면 0
    if os.path.exists(output_path):
        prev_R, prev_t = load_extrinsic_json(output_path)
        tvec_init = prev_t.astype(np.float64)
    else:
        tvec_init = np.zeros((3, 1), dtype=np.float64)

    rvec_init = np.zeros((3, 1), dtype=np.float64)

    # 투영용 임시 extrinsic(축 정렬 + t만)
    extr_init = perception_utils.ExtrinsicRT(R=R_align, t=tvec_init)
    cam_model = perception_utils.CameraModel(K=K, D=None)

    new_pts_count = 0
    margin = 30

    # [3] bbox별로 2D 타겟점 + radar_history에서 들어오는 후보 점들 중 bbox 근처 점 모으기
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]

        # 타겟점을 bbox 하단 65% 지점(범퍼 근사)으로 설정 (CARLA 로직)
        box_h = y2 - y1
        box_cy = y1 + (box_h * 0.65)
        box_cx = (x1 + x2) / 2.0

        # 2D 중복 방지
        is_dup = False
        for stored_uv in _BUFFER_2D:
            dist_sq = (stored_uv[0] - box_cx) ** 2 + (stored_uv[1] - box_cy) ** 2
            if dist_sq < DUPLICATE_THR_PX ** 2:
                is_dup = True
                break
        if is_dup:
            continue

        points_for_this_car: List[np.ndarray] = []

        # radar_history 스캔
        for frame in radar_history:
            if frame is None or len(frame) == 0:
                continue
            arr = np.asarray(frame, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] < 4:
                continue

            for p in arr:
                pt3d = np.array([float(p[0]), float(p[1]), float(p[2])], dtype=np.float64)

                uv_int, _ = perception_utils.project_radar_to_image(
                    pt3d.reshape(1, 3), cam_model, extr_init,
                    width, height, use_distortion=False
                )
                if uv_int.shape[0] == 0:
                    continue
                u, v = uv_int[0]
                if (x1 - margin <= u <= x2 + margin) and (y1 - margin <= v <= y2 + margin):
                    # 너무 가까운 건 제외 (CARLA 코드에서 pt3d[0]>10 같은 gate가 있었음)
                    if pt3d[0] > 10.0:
                        points_for_this_car.append(pt3d)

        # [4] 충분한 점이 모이면 outlier 제거 + 가까운 클러스터 선정 + centroid
        if len(points_for_this_car) >= 5:
            pts_arr = np.array(points_for_this_car, dtype=np.float64)
            filtered = _filter_outliers_zscore(pts_arr, threshold=2.0)

            if len(filtered) >= 3:
                filtered = filtered[filtered[:, 0].argsort()]  # x(전방거리) 기준
                cutoff = max(3, int(len(filtered) * 0.3))
                closest_cluster = filtered[:cutoff]
                centroid = np.median(closest_cluster, axis=0)

                if centroid[0] < MIN_CALIB_DIST:
                    continue

                _BUFFER_3D.append(centroid)
                _BUFFER_2D.append([box_cx, box_cy])
                new_pts_count += 1

    total_pts = len(_BUFFER_3D)
    if new_pts_count > 0:
        print(f"[CALIB] Added {new_pts_count} pts. Total: {total_pts}")

    if total_pts < REQUIRED_SAMPLES:
        return False, f"Collecting... {total_pts}/{REQUIRED_SAMPLES}"

    # [5] PnP 2-stage (SQPnP -> ITERATIVE refine)
    try:
        pts_3d = np.array(_BUFFER_3D, dtype=np.float64)
        pts_2d = np.array(_BUFFER_2D, dtype=np.float64)

        pts_3d_aligned = (R_align @ pts_3d.T).T

        obj = pts_3d_aligned.reshape(-1, 1, 3)
        img = pts_2d.reshape(-1, 1, 2)

        best_rvec, best_tvec, best_err, best_method = None, None, float("inf"), "None"

        # stage-1: SQPnP (가능하면 사용)
        if hasattr(cv2, "SOLVEPNP_SQPNP"):
            ret_sq, r_sq, t_sq = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_SQPNP)  # :contentReference[oaicite:5]{index=5}
            if ret_sq:
                err_sq = _reprojection_error(obj, img, K, None, r_sq, t_sq)
                best_rvec, best_tvec, best_err, best_method = r_sq, t_sq, err_sq, "SQPnP"

        # stage-2: iterative refine
        init_r = best_rvec if best_rvec is not None else rvec_init
        init_t = best_tvec if best_tvec is not None else tvec_init

        ok, r_fin, t_fin = cv2.solvePnP(
            obj, img, K, None,
            rvec=init_r, tvec=init_t,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE
        )  # :contentReference[oaicite:6]{index=6}

        if ok:
            err_fin = _reprojection_error(obj, img, K, None, r_fin, t_fin)
            if err_fin < best_err:
                best_rvec, best_tvec, best_err = r_fin, t_fin, err_fin
                best_method = best_method + "+Refine"

        if best_rvec is None:
            return False, "All PnP solvers failed."

        R_delta, _ = cv2.Rodrigues(best_rvec)
        R_final = R_delta @ R_align
        t_final = best_tvec.reshape(3, 1)

        inliers = np.arange(len(obj)).astype(int)
        result = CalibrationResult(R=R_final, t=t_final, reproj_error_px=best_err, inliers=inliers, method=best_method)

        save_extrinsic_json(output_path, R_final, t_final, meta={"method": best_method, "reproj_px": float(best_err)})
        save_calibration_log(result, total_pts, pts_3d, pts_2d)

        reset_calibration_buffer()

        tx, ty, tz = t_final.flatten()
        return True, f"Success! T=[{tx:.2f}, {ty:.2f}, {tz:.2f}] (Err: {best_err:.2f}px)"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"Optimization Failed: {e}"
