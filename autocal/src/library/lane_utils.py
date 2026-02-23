#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
차선 ROI 폴리곤 파일 유틸리티
- lane_polys JSON 로드
- 차선별 폴리곤 JSON 저장
"""

import json
import os
import numpy as np

# 지원 차선 이름
LANE_NAMES = ["IN1", "IN2", "IN3", "OUT1", "OUT2", "OUT3"]

# JSON 파일에서 차선별 폴리곤 좌표 로드
def load_lane_polys(json_path):
    polys = {name: np.empty((0, 2), dtype=np.int32) for name in LANE_NAMES}

    if not os.path.exists(json_path):
        return polys

    try:
        # 파일 로드 및 차선별 ndarray 변환
        with open(json_path, 'r') as f:
            data = json.load(f)

        for k, v in data.items():
            if v and len(v) > 0:
                polys[k] = np.array(v, dtype=np.int32)
    except Exception as e:
        print(f"[LaneUtils] Error loading {json_path}: {e}")

    return polys


# 차선 폴리곤 딕셔너리를 JSON 파일로 저장
def save_lane_polys(json_path, polys_dict):
    data = {}
    # ndarray 형태를 JSON 저장 가능한 리스트로 변환
    for k, v in polys_dict.items():
        if isinstance(v, np.ndarray) and v.size > 0:
            data[k] = v.tolist()
        else:
            data[k] = []

    try:
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"[LaneUtils] Saved to {json_path}")
        return True
    except Exception as e:
        print(f"[LaneUtils] Error saving {json_path}: {e}")
        return False