#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import numpy as np

# 기본 차선 이름 정의
LANE_NAMES = ["IN1", "IN2", "IN3", "OUT1", "OUT2", "OUT3"]

def load_lane_polys(json_path):
    """
    JSON 파일에서 차선 폴리곤을 로드합니다.
    Returns: dict { 'IN1': np.array([[x,y],...]), ... }
    """
    polys = {name: np.empty((0, 2), dtype=np.int32) for name in LANE_NAMES}
    
    if not os.path.exists(json_path):
        return polys

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        for k, v in data.items():
            if v and len(v) > 0:
                polys[k] = np.array(v, dtype=np.int32)
    except Exception as e:
        print(f"[LaneUtils] Error loading {json_path}: {e}")
        
    return polys

def save_lane_polys(json_path, polys_dict):
    """
    차선 폴리곤 딕셔너리를 JSON 파일로 저장합니다.
    """
    data = {}
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