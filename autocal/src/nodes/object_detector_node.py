#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
역할:
- 카메라 영상 토픽을 수신해 YOLO 추론을 수행하고, 검출 결과를 `DetectionArray`로 변환해 후단 파이프라인으로 전달한다.
- 입력 영상이 단일 채널일 때 BGR로 변환하고, 선택적 밝기 보정을 적용해 다양한 입력 조건에서 추론 일관성을 유지한다.
- ROS 파라미터로 가중치 경로, 임계값, 클래스, 디바이스, 출력 토픽을 제어하며, 디버그 이미지를 옵션으로 발행한다.

입력:
- `~topics/image_in` (`sensor_msgs/Image`): 카메라 입력 영상.
- `~detector/*` 파라미터: 모델 가중치/임계값/클래스/디바이스/입력크기.
- `~runtime/*` 파라미터: 밝기 보정 및 디버그 옵션.

출력:
- `~topics/detections_out` (`autocal/DetectionArray`): 원본 검출 결과.
- `~topics/debug_image` (`sensor_msgs/Image`, 선택): 박스 시각화 디버그 이미지.
"""

import re

import cv2
import rospkg
import rospy
from autocal.msg import BoundingBox, Detection, DetectionArray
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# ---------------------------------------------------------
# 객체 검출 파라미터
# ---------------------------------------------------------
DEFAULT_WEIGHTS = "$(find autocal)/models/yolo11m.pt"  # YOLO 가중치 경로
DEFAULT_CONF_THRES = 0.6                                # confidence 임계값
DEFAULT_IOU_THRES = 0.5                                 # NMS IoU 임계값
DEFAULT_CLASSES = [2, 3, 5, 7]                          # 추론 대상 클래스
DEFAULT_DEVICE = "cuda:0"                               # 추론 디바이스
DEFAULT_IMG_SIZE = 640                                  # 추론 입력 크기
DEFAULT_PUBLISH_DEBUG = True                            # 디버그 이미지 발행 여부
DEFAULT_BRIGHTNESS_ENABLE = False                       # 밝기 보정 사용 여부
DEFAULT_BRIGHTNESS_GAIN = 1.0                           # 밝기 보정 계수

DEFAULT_IMAGE_IN_TOPIC = "/camera/image_raw"            # 입력 영상 토픽
DEFAULT_DETECTIONS_OUT_TOPIC = "/autocal/detections_raw"  # 출력 검출 토픽
DEFAULT_DEBUG_IMAGE_TOPIC = "/autocal/detector_debug"      # 디버그 이미지 토픽


class ObjectDetectorNode:
    def __init__(self):
        """
        모델, 파라미터, 입출력 토픽, 브리지 객체를 초기화
        """
        rospy.init_node("object_detector_node")

        if YOLO is None:
            rospy.logerr("[ObjectDetector] ultralytics library is missing")
            return

        # detector 파라미터 로드
        raw_weights = rospy.get_param("~detector/weights", DEFAULT_WEIGHTS)
        self.weights_path = self.resolve_ros_path(raw_weights)
        self.conf_thres = float(rospy.get_param("~detector/conf_thres", DEFAULT_CONF_THRES))
        self.iou_thres = float(rospy.get_param("~detector/iou_thres", DEFAULT_IOU_THRES))
        self.classes = rospy.get_param("~detector/classes", DEFAULT_CLASSES)
        self.device = rospy.get_param("~detector/device", DEFAULT_DEVICE)
        self.img_size = int(rospy.get_param("~detector/img_size", DEFAULT_IMG_SIZE))

        # runtime 파라미터 로드
        self.publish_debug = bool(rospy.get_param("~runtime/publish_debug", DEFAULT_PUBLISH_DEBUG))
        self.brightness_enable = bool(rospy.get_param("~runtime/brightness_enable", DEFAULT_BRIGHTNESS_ENABLE))
        self.brightness_gain = float(rospy.get_param("~runtime/brightness_gain", DEFAULT_BRIGHTNESS_GAIN))

        # 토픽 파라미터 로드
        self.sub_topic = rospy.get_param("~topics/image_in", DEFAULT_IMAGE_IN_TOPIC)
        self.pub_topic = rospy.get_param("~topics/detections_out", DEFAULT_DETECTIONS_OUT_TOPIC)
        self.debug_topic = rospy.get_param("~topics/debug_image", DEFAULT_DEBUG_IMAGE_TOPIC)

        # YOLO 모델 로드
        rospy.loginfo(f"[YOLO] Loading model from {self.weights_path} to {self.device}")
        try:
            self.model = YOLO(self.weights_path)
        except Exception as exc:
            rospy.logerr(f"[YOLO] Failed to load model: {exc}")
            raise

        # ROS 통신 객체 초기화
        self.bridge = CvBridge()
        self.pub = rospy.Publisher(self.pub_topic, DetectionArray, queue_size=5)
        self.pub_debug = rospy.Publisher(self.debug_topic, Image, queue_size=1)
        self.sub = rospy.Subscriber(self.sub_topic, Image, self.callback, queue_size=1, buff_size=2**24)

        rospy.loginfo(f"[ObjectDetector] Ready. In={self.sub_topic}, Out={self.pub_topic}")

    def resolve_ros_path(self, path: str) -> str:
        """
        `$(find pkg)` 형식 경로를 실제 절대 경로로 변환
        """
        if "$(" not in path:
            return path

        try:
            rp = rospkg.RosPack()

            def _sub(match):
                """
                패키지명을 RosPack 경로로 치환
                """
                return rp.get_path(match.group(1))

            return re.sub(r"\$\(\s*find\s+([A-Za-z0-9_]+)\s*\)", _sub, path)
        except Exception as exc:
            rospy.logwarn(f"[ObjectDetector] Path resolution failed: {exc}")
            return path

    def callback(self, msg: Image):
        """
        입력 이미지를 전처리하고 YOLO 추론 결과를 DetectionArray로 발행
        """
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError:
            return

        # 밝기 보정 및 채널 정규화
        img_input = cv_img
        if self.brightness_enable and abs(self.brightness_gain - 1.0) > 1e-3:
            img_input = cv2.convertScaleAbs(img_input, alpha=self.brightness_gain, beta=0)
        if len(img_input.shape) == 2:
            img_input = cv2.cvtColor(img_input, cv2.COLOR_GRAY2BGR)

        # YOLO 추론 실행
        results = self.model.predict(
            source=img_input,
            conf=self.conf_thres,
            iou=self.iou_thres,
            device=self.device,
            imgsz=self.img_size,
            classes=self.classes,
            verbose=False,
            agnostic_nms=True,
        )

        # DetectionArray 메시지 구성
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
                    detection.cls = 0
                    detection.score = float(box.conf[0].item())

                    bbox = BoundingBox()
                    bbox.xmin = int(x1)
                    bbox.ymin = int(y1)
                    bbox.xmax = int(x2)
                    bbox.ymax = int(y2)
                    detection.bbox = bbox

                    det_array.detections.append(detection)

        # 검출 결과 발행
        self.pub.publish(det_array)

        # 디버그 이미지 발행
        if self.publish_debug:
            self.publish_debug_image(img_input, det_array)

    def publish_debug_image(self, img, det_array: DetectionArray):
        """
        검출 박스를 그린 디버그 이미지를 생성해 발행
        """
        debug_img = img.copy()
        for det in det_array.detections:
            bb = det.bbox
            cv2.rectangle(debug_img, (bb.xmin, bb.ymin), (bb.xmax, bb.ymax), (0, 255, 0), 2)
            label = f"{det.cls} {det.score:.2f}"
            cv2.putText(
                debug_img,
                label,
                (bb.xmin, max(0, bb.ymin - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
            self.pub_debug.publish(debug_msg)
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
