#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO 검출 유틸리티
- 모델 로드 및 추론 실행
- 검출 결과를 공용 dict 포맷으로 변환
"""

import cv2
import numpy as np
import torch

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    print("[YOLO] ultralytics not installed.")

# YOLO 모델 초기화와 추론을 담당하는 래퍼
class YOLODetector:

    # 가중치 파일과 디바이스 설정으로 모델 초기화
    def __init__(self, weights_path, device='cuda:0', conf=0.25):
        if YOLO is None:
            raise ImportError("ultralytics package is missing")
            
        self.device = device
        self.conf = conf
        print(f"[YOLO] Loading weights: {weights_path}")
        self.model = YOLO(weights_path)
    
    # 단일 프레임 추론 후 bbox/class/score 목록 반환
    def detect(self, img_bgr):
        # 단일 채널 입력은 BGR 변환 후 추론
        if len(img_bgr.shape) == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

        # 모델 추론 실행
        results = self.model.predict(
            source=img_bgr,
            conf=self.conf,
            device=self.device,
            verbose=False,
            agnostic_nms=True
        )

        # 결과 포맷을 GUI/노드 공용 dict 형식으로 변환
        detections = []
        if len(results) > 0:
            res = results[0]
            if res.boxes:
                for box in res.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cls = int(box.cls[0].item())
                    score = float(box.conf[0].item())
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'cls': cls,
                        'score': score
                    })
        return detections