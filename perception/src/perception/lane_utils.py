#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lane_utils.py  (CARLA lane_polys 로직 기반)

- lane_polys.json 로드/저장
- "선택된 차선 내부인지" 필터 (GUI/파라미터 lane_state dict)
- cv2.pointPolygonTest 기반 (없으면 ray casting fallback)

CARLA 코드의 전역 LANE_POLYS 패턴을 유지하되,
ROS에서는 파일 경로를 외부에서 넘기거나 기본 경로를 계산해 사용.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

from .perception_utils import resolve_ros_path


LANE_NAMES = ["IN1", "IN2", "OUT1", "OUT2"]

# 전역 폴리곤(원하면 class로 감싸도 되는데, CARLA 코드 스타일 유지)
LANE_POLYS: Dict[str, Optional[np.ndarray]] = {name: None for name in LANE_NAMES}


def default_lane_polys_path() -> str:
    """
    기본값: perception/src/nodes/lane_polys.json
    (네 프로젝트 구조에 맞춤)
    """
    here = Path(__file__).resolve().parent   # .../src/perception
    nodes_dir = (here.parent / "nodes")     # .../src/nodes
    return str(nodes_dir / "lane_polys.json")


def load_lane_polys(path: Optional[str] = None) -> Dict[str, Optional[np.ndarray]]:
    global LANE_POLYS
    if path is None:
        path = default_lane_polys_path()
    path = resolve_ros_path(path)

    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                kk = str(k)
                if kk in LANE_POLYS and v is not None:
                    LANE_POLYS[kk] = np.array(v, dtype=np.int32).reshape(-1, 2)
            print(f"[lane] loaded: {path}")
        else:
            print(f"[lane] not found: {path} (skip)")
    except Exception as e:
        print(f"[lane] load error: {e}")

    return LANE_POLYS


def save_lane_polys(path: Optional[str] = None) -> None:
    if path is None:
        path = default_lane_polys_path()
    path = resolve_ros_path(path)

    try:
        out = {}
        for k, v in LANE_POLYS.items():
            out[k] = v.tolist() if isinstance(v, np.ndarray) else None
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"[lane] saved: {path}")
    except Exception as e:
        print(f"[lane] save error: {e}")


def set_lane_polys(polys_dict: Dict[str, Optional[np.ndarray]], path: Optional[str] = None) -> None:
    global LANE_POLYS
    # 외부(GUI)에서 업데이트
    for k in LANE_NAMES:
        LANE_POLYS[k] = None
    for k, v in polys_dict.items():
        kk = str(k)
        if kk in LANE_POLYS and v is not None:
            LANE_POLYS[kk] = np.array(v, dtype=np.int32).reshape(-1, 2)
    save_lane_polys(path)


def _point_in_poly(u: float, v: float, poly: np.ndarray) -> bool:
    if poly is None or len(poly) < 3:
        return False

    if cv2 is not None:
        contour = poly.reshape(-1, 1, 2).astype(np.int32)
        return cv2.pointPolygonTest(contour, (float(u), float(v)), False) >= 0

    # fallback (ray casting)
    x, y = float(u), float(v)
    inside = False
    n = poly.shape[0]
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)):
            xinters = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if x < xinters:
                inside = not inside
    return inside


def is_in_selected_lanes(u: int, v: int, lane_state: Dict[str, bool]) -> bool:
    """
    CARLA 코드 lane_state 형태 유지:
      lane_state = {
        'lane_on': True,
        'in1': True, 'in2': False, 'out1': False, 'out2': False
      }
    """
    if not lane_state.get("lane_on", True):
        return True  # 필터 OFF면 통과

    selected = [name for name in LANE_NAMES if lane_state.get(name.lower(), False)]
    if len(selected) == 0:
        return True  # 선택된 lane이 없으면 필터링 안함

    for name in selected:
        poly = LANE_POLYS.get(name)
        if poly is None:
            continue
        if _point_in_poly(u, v, poly):
            return True

    return False


def hit_lane_name(u: int, v: int, targets: Optional[List[str]] = None) -> Optional[str]:
    """
    (u,v)가 어떤 lane polygon 안에 들어가는지 lane name 반환
    """
    names = targets if targets else LANE_NAMES
    for name in names:
        poly = LANE_POLYS.get(name)
        if poly is None:
            continue
        if _point_in_poly(u, v, poly):
            return name
    return None
