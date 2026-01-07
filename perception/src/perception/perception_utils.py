#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
perception_utils.py  (CARLA 로직 기반 공용 유틸)

포함 기능
- CameraModel(K,D) / ExtrinsicRT(R,t) 데이터 구조
- Radar 3D -> Camera -> Pixel 투영 (pinhole / cv2.projectPoints)
- 투영 결과와 keep index 반환 (association 노드에서 필수)
- bbox 내부 레이더 점 선택 + 속도(도플러) 집계 (median / weighted median)
- CARLA에서 쓰던 radar history(deque) 유사 버퍼
- $(find pkg) 경로 치환 유틸 (launch subst_value 없이도 안전)

참고:
- PointCloud2 파싱은 노드에서 sensor_msgs.point_cloud2.read_points로 처리하는게 일반적. :contentReference[oaicite:0]{index=0}
- 왜곡 포함 투영은 cv2.projectPoints가 표준. :contentReference[oaicite:1]{index=1}
"""

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

try:
    import yaml
except Exception:
    yaml = None

try:
    import rospkg
except Exception:
    rospkg = None


_FIND_RE = re.compile(r"\$\(\s*find\s+([A-Za-z0-9_]+)\s*\)")


def resolve_ros_path(path: str) -> str:
    """'$(find pkg)' 치환. rospkg 없으면 원문 반환."""
    if not isinstance(path, str) or "$(" not in path:
        return path
    if rospkg is None:
        return path

    rp = rospkg.RosPack()

    def _sub(m):
        pkg = m.group(1)
        try:
            return rp.get_path(pkg)
        except Exception:
            return m.group(0)

    return _FIND_RE.sub(_sub, path)


# -------------------------------
# 1) Camera model / Extrinsic
# -------------------------------
@dataclass
class CameraModel:
    K: np.ndarray                  # (3,3)
    D: Optional[np.ndarray] = None # (k1,k2,p1,p2[,k3...]) or None


@dataclass
class ExtrinsicRT:
    R: np.ndarray  # (3,3)
    t: np.ndarray  # (3,1)

    def as_3x4(self) -> np.ndarray:
        return np.hstack([self.R, self.t.reshape(3, 1)])


# -------------------------------
# 2) Loaders (yaml/json)
# -------------------------------
def load_extrinsic_json(path: str) -> ExtrinsicRT:
    path = resolve_ros_path(path)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    R = np.array(payload["R"], dtype=np.float64).reshape(3, 3)
    t = np.array(payload["t"], dtype=np.float64).reshape(3, 1)
    return ExtrinsicRT(R=R, t=t)


def save_extrinsic_json(path: str, extr: ExtrinsicRT, meta: Optional[Dict[str, Any]] = None) -> None:
    path = resolve_ros_path(path)
    payload = {"R": extr.R.tolist(), "t": extr.t.reshape(3).tolist()}
    if meta:
        payload["meta"] = meta
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_camera_intrinsic_yaml(path: str) -> Tuple[CameraModel, int, int]:
    """
    camera_intrinsic.yaml (여러 포맷 대응)
    return: (CameraModel, width, height)
    """
    path = resolve_ros_path(path)
    if yaml is None:
        raise RuntimeError("pyyaml not installed.")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    w = int(cfg.get("image_width", cfg.get("width", 0)))
    h = int(cfg.get("image_height", cfg.get("height", 0)))

    K = None
    D = None

    if isinstance(cfg.get("camera_matrix"), dict):
        data = cfg["camera_matrix"].get("data")
        if data is not None:
            K = np.array(data, dtype=np.float64).reshape(3, 3)

    if K is None and "K" in cfg:
        k = cfg["K"]
        K = np.array(k, dtype=np.float64).reshape(3, 3)

    if isinstance(cfg.get("distortion_coefficients"), dict):
        d = cfg["distortion_coefficients"].get("data")
        if d is not None:
            D = np.array(d, dtype=np.float64).reshape(-1)

    if D is None and "D" in cfg:
        D = np.array(cfg["D"], dtype=np.float64).reshape(-1)

    if K is None:
        raise ValueError(f"K not found in intrinsic yaml: {path}")

    return CameraModel(K=K, D=D), w, h


def load_extrinsic_yaml(path: str) -> ExtrinsicRT:
    """
    camera_radar_extrinsic.yaml (여러 포맷 대응)
    지원:
      - R: [[...],[...],[...]], t: [tx,ty,tz]
      - rpy_deg: [r,p,y], xyz: [x,y,z]
    """
    path = resolve_ros_path(path)
    if yaml is None:
        raise RuntimeError("pyyaml not installed.")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if "R" in cfg and "t" in cfg:
        R = np.array(cfg["R"], dtype=np.float64).reshape(3, 3)
        t = np.array(cfg["t"], dtype=np.float64).reshape(3, 1)
        return ExtrinsicRT(R=R, t=t)

    if "rpy_deg" in cfg and "xyz" in cfg:
        r, p, y = [float(v) * math.pi / 180.0 for v in cfg["rpy_deg"]]
        x, yy, z = [float(v) for v in cfg["xyz"]]
        R = rpy_to_R(r, p, y)
        t = np.array([[x], [yy], [z]], dtype=np.float64)
        return ExtrinsicRT(R=R, t=t)

    raise ValueError(f"Invalid extrinsic yaml: {path}")


# -------------------------------
# 3) CARLA helper: K from fov
# -------------------------------
def get_camera_intrinsic_from_fov(fov_deg: float, w: int, h: int) -> np.ndarray:
    """CARLA처럼 fov만 있을 때 K를 만드는 유틸(실카메라는 보통 캘리브레이션 K 사용)."""
    f = w / (2.0 * math.tan((float(fov_deg) * math.pi / 180.0) / 2.0))
    return np.array([[f, 0, w / 2.0],
                     [0, f, h / 2.0],
                     [0, 0, 1.0]], dtype=np.float64)


def rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX(yaw->pitch->roll)"""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [0,    0, 1]], dtype=np.float64)
    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]], dtype=np.float64)
    Rx = np.array([[1,  0,   0],
                   [0, cr, -sr],
                   [0, sr,  cr]], dtype=np.float64)
    return Rz @ Ry @ Rx


# -------------------------------
# 4) Projection (CARLA core)
# -------------------------------
def project_radar_to_image(
    radar_xyz: np.ndarray,          # (N,3) radar frame
    cam: CameraModel,
    extr: ExtrinsicRT,
    w: int,
    h: int,
    use_distortion: bool = True,
    z_min_m: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    return:
      uv_int: (M,2) int pixel coords
      z_cam : (M,)  depth in camera frame
    """
    radar_xyz = np.asarray(radar_xyz, dtype=np.float64)
    if radar_xyz.ndim != 2 or radar_xyz.shape[1] != 3:
        raise ValueError("radar_xyz must be (N,3)")

    Xr = radar_xyz.T  # (3,N)
    Xc = (extr.R @ Xr) + extr.t.reshape(3, 1)

    z = Xc[2, :]
    valid = z > float(z_min_m)
    if not np.any(valid):
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=np.float64)

    Xc = Xc[:, valid]
    z = z[valid]

    if use_distortion and (cam.D is not None) and (cv2 is not None):
        # cv2.projectPoints: intrinsic+distortion로 투영 :contentReference[oaicite:2]{index=2}
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)
        pts = Xc.T.reshape(-1, 1, 3)
        img_pts, _ = cv2.projectPoints(pts, rvec, tvec, cam.K, cam.D)
        uv = img_pts.reshape(-1, 2)
    else:
        uvw = cam.K @ Xc
        uv = (uvw[:2, :] / uvw[2:3, :]).T

    if w > 0 and h > 0:
        u = uv[:, 0]
        v = uv[:, 1]
        in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        uv = uv[in_img]
        z = z[in_img]

    uv_int = np.round(uv).astype(np.int32)
    return uv_int, z


def project_radar_to_image_with_indices(
    radar_xyz: np.ndarray,          # (N,3) radar frame
    cam: CameraModel,
    extr: ExtrinsicRT,
    w: int,
    h: int,
    use_distortion: bool = True,
    z_min_m: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    association_node에서 쓰기 좋은 형태:
    return:
      uv_float: (M,2) float64
      keep_idx: (M,)  원래 radar_xyz 인덱스
    """
    radar_xyz = np.asarray(radar_xyz, dtype=np.float64)
    if radar_xyz.ndim != 2 or radar_xyz.shape[1] != 3:
        raise ValueError("radar_xyz must be (N,3)")

    Xr = radar_xyz.T
    Xc = (extr.R @ Xr) + extr.t.reshape(3, 1)

    z = Xc[2, :]
    valid = z > float(z_min_m)
    if not np.any(valid):
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.int32)

    idx_z = np.where(valid)[0]
    Xc = Xc[:, valid]

    if use_distortion and (cam.D is not None) and (cv2 is not None):
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)
        pts = Xc.T.reshape(-1, 1, 3)
        img_pts, _ = cv2.projectPoints(pts, rvec, tvec, cam.K, cam.D)
        uv = img_pts.reshape(-1, 2).astype(np.float64)
    else:
        uvw = cam.K @ Xc
        uv = (uvw[:2, :] / uvw[2:3, :]).T.astype(np.float64)

    if w > 0 and h > 0:
        u = uv[:, 0]
        v = uv[:, 1]
        in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    else:
        in_img = np.ones((uv.shape[0],), dtype=bool)

    if not np.any(in_img):
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.int32)

    return uv[in_img], idx_z[in_img].astype(np.int32)


# -------------------------------
# 5) Speed aggregation (CARLA core + 확장)
# -------------------------------
def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    m = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[m], weights[m]
    if values.size == 0:
        return float("nan")

    idx = np.argsort(values)
    v = values[idx]
    w = weights[idx]
    c = np.cumsum(w)
    cutoff = 0.5 * c[-1]
    return float(v[np.searchsorted(c, cutoff)])


def estimate_speed_in_bbox(
    radar_points: np.ndarray,   # (N,4) [x,y,z,dop] 또는 (N,5) [x,y,z,dop,power]
    bxyxy: Tuple[float, float, float, float],
    cam: CameraModel,
    extr: ExtrinsicRT,
    w: int,
    h: int,
    min_points: int = 1,
    use_distortion: bool = False,
    min_abs_doppler: float = 0.0,
    min_power: float = 0.0,
    topk_by_power: int = 0,
    use_power_weighted: bool = True,
    use_abs_speed: bool = True,
) -> float:
    """
    CARLA에서 하던 "bbox 내부 레이더 도플러 median"을 실사용용으로 확장:
    - power gate / top-k / weighted median 지원
    return: speed_kph (nan 가능)
    """
    if radar_points is None or len(radar_points) == 0:
        return float("nan")

    x1, y1, x2, y2 = map(float, bxyxy)
    if x2 <= x1 or y2 <= y1:
        return float("nan")

    rp = np.asarray(radar_points, dtype=np.float64)
    if rp.ndim != 2 or rp.shape[1] < 4:
        return float("nan")

    xyz = rp[:, :3]
    dop = rp[:, 3]

    uv, keep = project_radar_to_image_with_indices(
        radar_xyz=xyz, cam=cam, extr=extr, w=w, h=h,
        use_distortion=use_distortion, z_min_m=1e-6
    )
    if uv.shape[0] == 0:
        return float("nan")

    u = uv[:, 0]
    v = uv[:, 1]
    inside = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)

    sel_idx = keep[inside]
    if sel_idx.size < int(min_points):
        return float("nan")

    sel_dop = dop[sel_idx]
    # doppler gate
    m = np.isfinite(sel_dop) & (np.abs(sel_dop) >= max(0.0, float(min_abs_doppler)))
    sel_dop = sel_dop[m]
    if sel_dop.size < int(min_points):
        return float("nan")

    # power가 있으면 사용
    speed_mps = float("nan")
    if rp.shape[1] >= 5:
        pwr = rp[:, 4][sel_idx][m]
        mp = np.isfinite(pwr) & (pwr >= float(min_power))
        sel_dop2, pwr2 = sel_dop[mp], pwr[mp]
        if sel_dop2.size < int(min_points):
            return float("nan")

        if topk_by_power > 0 and sel_dop2.size > topk_by_power:
            idx = np.argsort(pwr2)[-int(topk_by_power):]
            sel_dop2, pwr2 = sel_dop2[idx], pwr2[idx]

        if use_power_weighted:
            speed_mps = weighted_median(sel_dop2, pwr2)
        else:
            speed_mps = float(np.median(sel_dop2))
    else:
        speed_mps = float(np.median(sel_dop))

    if not np.isfinite(speed_mps):
        return float("nan")

    if use_abs_speed:
        speed_mps = abs(speed_mps)

    return float(speed_mps * 3.6)


# -------------------------------
# 6) Radar history buffer (CARLA 느낌 유지)
# -------------------------------
class RadarHistoryBuffer:
    """
    CARLA 코드에서 radar_history(최근 N 프레임)를 돌리던 패턴을 그대로 가져오기 위한 버퍼.
    - add(frame_points): (M,4/5) numpy
    - get_all(): List[np.ndarray]
    """
    def __init__(self, maxlen: int = 15):
        from collections import deque
        self._dq: Deque[np.ndarray] = deque(maxlen=int(maxlen))

    def clear(self):
        self._dq.clear()

    def add(self, frame_points: np.ndarray):
        if frame_points is None:
            return
        arr = np.asarray(frame_points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 4:
            return
        self._dq.append(arr)

    def get_all(self) -> List[np.ndarray]:
        return list(self._dq)

    def concat(self) -> np.ndarray:
        if len(self._dq) == 0:
            return np.zeros((0, 4), dtype=np.float64)
        return np.concatenate(list(self._dq), axis=0)
