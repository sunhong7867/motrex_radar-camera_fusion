#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
yolo_detector.py  (CARLA YOLO 래퍼 기반)

- Ultralytics YOLO 로드
- infer(img_bgr) -> dets(list[dict]) 반환 (CARLA 코드와 동일)
- weights 기본 경로는 perception/models/best.pt 를 권장

주의:
- Jetson Orin에서는 device="cuda:0" 권장
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from .perception_utils import resolve_ros_path


class YOLODetector:
    def __init__(
        self,
        weights: str = "$(find perception)/models/best.pt",
        device: str = "cuda:0",
        conf: float = 0.20,
        iou: float = 0.40,
        agnostic_nms: bool = True,
        verbose: bool = False,
    ):
        try:
            from ultralytics import YOLO
        except Exception:
            raise RuntimeError("ultralytics 미설치. (pip install ultralytics)")

        weights = resolve_ros_path(weights)
        wp = Path(weights)
        if not wp.exists():
            raise FileNotFoundError(f"YOLO weight not found: {wp}")

        self.model = YOLO(str(wp))
        self.device = device
        self.conf = float(conf)
        self.iou = float(iou)
        self.agnostic_nms = bool(agnostic_nms)
        self.verbose = bool(verbose)

        # class names 출력은 디버깅에 유용
        try:
            print("[YOLO] class names:", self.model.names)
        except Exception:
            pass

    def infer(
        self,
        img_bgr,
        conf: Optional[float] = None,
        imgsz: int = 640,
        max_det: int = 50,
        classes: Optional[List[int]] = None,
        half: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        return: [{"bbox":(x1,y1,x2,y2), "cls":int, "conf":float}, ...]
        """
        target_conf = float(conf) if conf is not None else self.conf

        results = self.model.predict(
            source=img_bgr,
            conf=target_conf,
            iou=self.iou,
            device=self.device,
            verbose=self.verbose,
            stream=False,
            agnostic_nms=self.agnostic_nms,
            imgsz=int(imgsz),
            max_det=int(max_det),
            classes=classes,
            half=bool(half),
        )

        if results is None or len(results) == 0:
            return []

        r = results[0]
        if getattr(r, "boxes", None) is None:
            return []

        dets: List[Dict[str, Any]] = []
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            dets.append({
                "bbox": (x1, y1, x2, y2),
                "cls": int(box.cls[0]),
                "conf": float(box.conf[0]),
            })
        return dets
