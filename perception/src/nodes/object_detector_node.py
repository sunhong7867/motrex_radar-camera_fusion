#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
object_detector_node.py

역할:
- 카메라 Image 토픽을 구독하여 YOLO로 객체(차량 등)를 검출
- 검출 결과를 perception/DetectionArray(msg)로 publish
- (옵션) 디버그용 bbox 오버레이 이미지 publish

입력:
- ~camera/image_topic (sensor_msgs/Image) [default: /mv_camera/image_raw]
- (선택) ~camera/camera_info_topic (sensor_msgs/CameraInfo)  # 여기선 필수 아님

출력:
- ~out_topic (perception/DetectionArray) [default: /perception/detections]
- ~debug/image_out_topic (sensor_msgs/Image) [default: /perception/detections_vis] (옵션)

파라미터(여러 키를 지원):
- weights:  ~detector/weights, ~detector/weights_path, ~weights, ~weights_path
- imgsz:    ~detector/img_size, ~detector/imgsz, ~img_size, ~imgsz
- conf:     ~detector/conf_thres, ~detector/conf, ~conf_thres, ~conf
- iou:      ~detector/iou_thres, ~detector/iou, ~iou_thres, ~iou
- classes:  ~detector/classes, ~classes  (예: [2,3,5,7])
- max_det:  ~detector/max_det, ~max_det
- device:   ~detector/device, ~device  (예: "cuda:0" / "cpu")
- half:     ~detector/half, ~half
- rate:     ~run_rate_hz, ~max_fps (처리율 제한, default 15Hz)
- skip:     ~skip_frames (N이면 N프레임마다 1번 처리)
- debug:    ~debug/publish_image, ~publish_image (bbox 그린 이미지 publish)
"""

import os
import time
from typing import Any, List, Optional

import numpy as np
import rospy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from perception.msg import DetectionArray, Detection, BoundingBox

try:
    import cv2
except Exception:
    cv2 = None


def get_param_first(names: List[str], default: Any) -> Any:
    for n in names:
        if rospy.has_param(n):
            return rospy.get_param(n)
    return default


class ObjectDetectorNode:
    def __init__(self):
        # --------------------
        # Topics
        # --------------------
        self.image_topic = get_param_first(
            ["~camera/image_topic", "~image_topic", "~camera_topic"],
            "/mv_camera/image_raw"
        )
        self.out_topic = get_param_first(
            ["~out_topic", "~detector/out_topic", "~detections_topic"],
            "/perception/detections"
        )

        # debug image publish
        self.publish_debug_image = bool(get_param_first(
            ["~debug/publish_image", "~publish_image"],
            True
        ))
        self.debug_image_topic = get_param_first(
            ["~debug/image_out_topic", "~image_out_topic"],
            "/perception/detections_vis"
        )

        # --------------------
        # Detector params
        # --------------------
        self.weights = str(get_param_first(
            ["~detector/weights", "~detector/weights_path", "~weights", "~weights_path"],
            "$(find perception)/models/best.pt"  # launch에서 subst_value 쓰는 걸 권장
        ))

        self.imgsz = int(get_param_first(
            ["~detector/img_size", "~detector/imgsz", "~img_size", "~imgsz"],
            640
        ))
        self.conf = float(get_param_first(
            ["~detector/conf_thres", "~detector/conf", "~conf_thres", "~conf"],
            0.25
        ))
        self.iou = float(get_param_first(
            ["~detector/iou_thres", "~detector/iou", "~iou_thres", "~iou"],
            0.45
        ))
        self.max_det = int(get_param_first(
            ["~detector/max_det", "~max_det"],
            50
        ))
        self.device = str(get_param_first(
            ["~detector/device", "~device"],
            "cuda:0"
        ))
        self.half = bool(get_param_first(
            ["~detector/half", "~half"],
            False
        ))
        self.agnostic_nms = bool(get_param_first(
            ["~detector/agnostic_nms", "~agnostic_nms"],
            False
        ))
        self.verbose = bool(get_param_first(
            ["~detector/verbose", "~verbose"],
            False
        ))

        classes = get_param_first(["~detector/classes", "~classes"], [])
        self.classes: Optional[List[int]] = None
        try:
            if isinstance(classes, list) and len(classes) > 0:
                self.classes = [int(x) for x in classes]
        except Exception:
            self.classes = None

        # --------------------
        # Rate limiting
        # --------------------
        self.run_rate_hz = float(get_param_first(["~run_rate_hz", "~max_fps"], 15.0))
        self.min_interval = 1.0 / max(1e-3, self.run_rate_hz)
        self.skip_frames = int(get_param_first(["~skip_frames"], 0))
        self._frame_count = 0
        self._last_run_time = 0.0

        # --------------------
        # ROS I/O
        # --------------------
        self.bridge = CvBridge()
        self.pub_det = rospy.Publisher(self.out_topic, DetectionArray, queue_size=5)
        self.pub_img = rospy.Publisher(self.debug_image_topic, Image, queue_size=2) if self.publish_debug_image else None

        # --------------------
        # Load model
        # --------------------
        self.model = self._load_model()

        rospy.Subscriber(self.image_topic, Image, self._on_image, queue_size=1, buff_size=2**24)

        rospy.loginfo(f"[detector] image_topic={self.image_topic}")
        rospy.loginfo(f"[detector] out_topic={self.out_topic}")
        rospy.loginfo(f"[detector] weights={self.weights}, imgsz={self.imgsz}, conf={self.conf}, iou={self.iou}, classes={self.classes}")

    def _load_model(self):
        # ultralytics YOLO 사용(실사용에서 제일 간단/안정)
        try:
            from ultralytics import YOLO
        except Exception as e:
            rospy.logerr("[detector] ultralytics not found. Install: pip install ultralytics")
            raise

        # weights 경로가 $(find ...) 형태면 launch subst_value가 해결해주는 게 정석
        # 그래도 파일이 없으면 바로 로그로 알려주기
        if os.path.isabs(self.weights) and (not os.path.exists(self.weights)):
            rospy.logwarn(f"[detector] weights file not found: {self.weights}")

        model = YOLO(self.weights)
        return model

    def _should_run(self) -> bool:
        # skip_frames: N이면 N프레임마다 1번 처리 (예: 2면 2프레임마다 1회)
        if self.skip_frames > 0:
            self._frame_count += 1
            if (self._frame_count % (self.skip_frames + 1)) != 1:
                return False

        now = time.time()
        if (now - self._last_run_time) < self.min_interval:
            return False

        self._last_run_time = now
        return True

    def _on_image(self, msg: Image):
        if not self._should_run():
            return

        # ROS Image -> cv2 BGR
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"[detector] cv_bridge convert failed: {e}")
            return

        # YOLO inference
        det_array = DetectionArray()
        det_array.header = msg.header

        try:
            # ultralytics predict
            # classes=None이면 전체 클래스, 리스트면 필터
            results = self.model.predict(
                source=cv_img,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                classes=self.classes,
                max_det=self.max_det,
                device=self.device,
                half=self.half,
                agnostic_nms=self.agnostic_nms,
                verbose=self.verbose
            )
            if results is None or len(results) == 0:
                self.pub_det.publish(det_array)
                return

            r0 = results[0]
            boxes = getattr(r0, "boxes", None)
            if boxes is None or len(boxes) == 0:
                self.pub_det.publish(det_array)
                return

            # boxes.xyxy, boxes.conf, boxes.cls
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
            conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
            cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.array(boxes.cls)

            # publish detections
            for i in range(xyxy.shape[0]):
                x1, y1, x2, y2 = xyxy[i].tolist()
                d = Detection()
                d.id = int(i)              # tracker 전이면 프레임 내 인덱스로 id 부여
                d.cls = int(cls[i])
                d.score = float(conf[i])

                bb = BoundingBox()
                bb.xmin = int(max(0, round(x1)))
                bb.ymin = int(max(0, round(y1)))
                bb.xmax = int(max(0, round(x2)))
                bb.ymax = int(max(0, round(y2)))
                d.bbox = bb

                det_array.detections.append(d)

            self.pub_det.publish(det_array)

        except Exception as e:
            rospy.logwarn_throttle(1.0, f"[detector] inference failed: {e}")
            # 그래도 빈 DetectionArray는 publish (downstream 안정)
            self.pub_det.publish(det_array)
            return

        # debug image (optional)
        if self.publish_debug_image and self.pub_img is not None and (cv2 is not None):
            dbg = cv_img.copy()
            for d in det_array.detections:
                x1, y1, x2, y2 = d.bbox.xmin, d.bbox.ymin, d.bbox.xmax, d.bbox.ymax
                cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    dbg, f"id:{d.id} c:{d.cls} {d.score:.2f}",
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
            try:
                out_msg = self.bridge.cv2_to_imgmsg(dbg, encoding="bgr8")
                out_msg.header = msg.header
                self.pub_img.publish(out_msg)
            except Exception:
                pass


def main():
    rospy.init_node("object_detector_node", anonymous=False)
    ObjectDetectorNode()
    rospy.spin()


if __name__ == "__main__":
    main()
