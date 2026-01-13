#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import torch

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    print("[YOLO] ultralytics not installed.")

class YOLODetector:
    def __init__(self, weights_path, device='cuda:0', conf=0.25):
        if YOLO is None:
            raise ImportError("ultralytics package is missing")
            
        self.device = device
        self.conf = conf
        print(f"[YOLO] Loading weights: {weights_path}")
        self.model = YOLO(weights_path)
    
    def detect(self, img_bgr):
        """
        이미지를 입력받아 검출 결과 리스트를 반환합니다.
        Return: [{'bbox': [x1, y1, x2, y2], 'cls': int, 'score': float}, ...]
        """
        # 흑백 이미지 대응
        if len(img_bgr.shape) == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

        results = self.model.predict(
            source=img_bgr,
            conf=self.conf,
            device=self.device,
            verbose=False,
            agnostic_nms=True
        )

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