#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
object_detector_node.py (Final)

역할:
- 카메라 이미지(/camera/image_raw)를 받아 YOLO 추론 수행
- 결과(BBox, Class, Conf)를 DetectionArray 메시지로 변환하여 발행
- [중요] 흑백(Mono8) 이미지가 들어오면 RGB로 변환하여 YOLO에 입력
- detector.yaml 설정 파일과 완벽 연동

데이터 흐름:
[Camera] (/camera/image_raw)
       ⬇
[Object Detector] (YOLO 추론)
       ⬇ (/perception_test/detections_raw)
[Lane Detector] (차선 필터링)
"""

import rospy
import cv2
import numpy as np
import os
import re
import rospkg
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from perception_test.msg import DetectionArray, Detection, BoundingBox

# YOLO 라이브러리 (ultralytics)
try:
    from ultralytics import YOLO
except ImportError:
    print("[Error] ultralytics not installed. Run: pip3 install ultralytics")
    YOLO = None

class ObjectDetectorNode:
    def __init__(self):
        rospy.init_node("object_detector_node")

        if YOLO is None:
            rospy.logerr("[ObjectDetector] ultralytics library is missing!")
            return

        # 1. 파라미터 로드 (detector.yaml 대응)
        # weights 경로 처리 ($(find perception_test) 치환)
        raw_weights = rospy.get_param("~detector/weights", "$(find perception_test)/models/yolo11m.pt")
        self.weights_path = self.resolve_ros_path(raw_weights)

        self.conf_thres = float(rospy.get_param("~detector/conf_thres", 0.6))
        self.iou_thres = float(rospy.get_param("~detector/iou_thres", 0.5))
        self.classes = rospy.get_param("~detector/classes", [2, 3, 5, 7]) # Car, Motorcycle, Bus, Truck
        self.device = rospy.get_param("~detector/device", "cuda:0")
        self.img_size = int(rospy.get_param("~detector/img_size", 640))
        
        # 디버그용 이미지 발행 여부
        self.publish_debug = rospy.get_param("~runtime/publish_debug", True)

        # 2. 토픽 설정
        # 입력: 카메라 노드에서 오는 영상
        self.sub_topic = rospy.get_param("~topics/image_in", "/camera/image_raw")
        # 출력: Lane Detector로 보낼 원본 검출 결과 (_raw 권장)
        self.pub_topic = rospy.get_param("~topics/detections_out", "/perception_test/detections_raw")
        self.debug_topic = rospy.get_param("~topics/debug_image", "/perception_test/detector_debug")

        # 3. 모델 로딩
        rospy.loginfo(f"[YOLO] Loading model from {self.weights_path} to {self.device}")
        try:
            self.model = YOLO(self.weights_path)
        except Exception as e:
            rospy.logerr(f"[YOLO] Failed to load model: {e}")
            # 모델 로드 실패 시 노드 종료 방지를 위해 더미 처리하거나 exit(1)
            exit(1)

        # 4. 초기화
        self.bridge = CvBridge()
        self.pub = rospy.Publisher(self.pub_topic, DetectionArray, queue_size=5)
        self.pub_debug = rospy.Publisher(self.debug_topic, Image, queue_size=1)
        self.sub = rospy.Subscriber(self.sub_topic, Image, self.callback, queue_size=1, buff_size=2**24)

        rospy.loginfo(f"[ObjectDetector] Ready. In: {self.sub_topic}, Out: {self.pub_topic}")

    def resolve_ros_path(self, path: str) -> str:
        """$(find pkg) 패턴을 절대 경로로 변환"""
        if "$(" in path:
            try:
                rp = rospkg.RosPack()
                def _sub(m): return rp.get_path(m.group(1))
                return re.sub(r"\$\(\s*find\s+([A-Za-z0-9_]+)\s*\)", _sub, path)
            except Exception as e:
                rospy.logwarn(f"[ObjectDetector] Path resolution failed: {e}")
                return path
        return path

    def callback(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            return

        brightness_enable = rospy.get_param("~runtime/brightness_enable", False)
        brightness_gain = float(rospy.get_param("~runtime/brightness_gain", 1.0))

        img_input = cv_img
        if brightness_enable and abs(brightness_gain - 1.0) > 1e-3:
            img_input = cv2.convertScaleAbs(img_input, alpha=brightness_gain, beta=0)

        if len(img_input.shape) == 2:
            img_input = cv2.cvtColor(img_input, cv2.COLOR_GRAY2BGR)

        # YOLO 추론
        results = self.model.predict(
            source=img_input,
            conf=self.conf_thres,
            iou=self.iou_thres,
            device=self.device,
            imgsz=self.img_size,
            classes=self.classes,
            verbose=False,
            agnostic_nms=True # 여기서도 강제 적용
        )

        det_array = DetectionArray()
        det_array.header = msg.header
        det_array.detections = []

        if len(results) > 0:
            result = results[0]
            if result.boxes:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    detection = Detection()
                    detection.id = 0 
                    
                    # [핵심 수정] 
                    # 원래 코드: detection.cls = int(box.cls[0].item())
                    # 수정 코드: 무조건 0번(단일 클래스)으로 통일
                    detection.cls = 0  
                    
                    detection.score = float(box.conf[0].item())
                    
                    bbox = BoundingBox()
                    bbox.xmin = x1; bbox.ymin = y1; bbox.xmax = x2; bbox.ymax = y2
                    detection.bbox = bbox

                    det_array.detections.append(detection)

        self.pub.publish(det_array)
        
    def publish_debug_image(self, img, det_array):
        """검출 결과를 그린 이미지를 발행 (Rviz 확인용)"""
        debug_img = img.copy()
        for det in det_array.detections:
            bb = det.bbox
            cv2.rectangle(debug_img, (bb.xmin, bb.ymin), (bb.xmax, bb.ymax), (0, 255, 0), 2)
            label = f"{det.cls} {det.score:.2f}"
            cv2.putText(debug_img, label, (bb.xmin, max(0, bb.ymin-5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        try:
            msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
            self.pub_debug.publish(msg)
        except Exception:
            pass

def main():
    try:
        ObjectDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()