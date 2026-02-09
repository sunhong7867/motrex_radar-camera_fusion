#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lane_detector_node.py (Final)

역할:
- YOLO에서 나온 DetectionArray를 받아, 설정된 차선(ROI) 안에 있는 차량만 필터링
- 필터링된 결과만 다음 단계(Association Node)로 전달
- lane_polys.json 파일 변경 시 실시간 반영 (Hot-Reload)

데이터 흐름:
[Object Detector] (/perception_test/detections_raw)
       ⬇
[Lane Detector]   (필터링 수행)
       ⬇
[Association Node] (/perception_test/detections)
"""

import os
import json
import time
import numpy as np
import rospy
import rospkg

from perception_test.msg import DetectionArray, Detection, BoundingBox
from std_msgs.msg import Header

try:
    import cv2
except ImportError:
    cv2 = None

class LaneDetectorNode:
    def __init__(self):
        rospy.init_node("lane_detector_node")

        # 1. 토픽 설정
        # 입력: YOLO가 보낸 원본 검출 결과
        self.sub_topic = rospy.get_param("~input_topic", "/perception_test/detections_raw")
        # 출력: 차선 필터링이 끝난 결과 (Association 노드가 받을 토픽)
        self.pub_topic = rospy.get_param("~output_topic", "/perception_test/detections")

        # 2. ROI 설정 파일 경로
        # 기본값: 패키지 내 src/nodes/lane_polys.json
        try:
            rospack = rospkg.RosPack()
            pkg_path = rospack.get_path("perception_test")
            default_poly_path = os.path.join(pkg_path, "src", "nodes", "lane_polys.json")
        except Exception:
            default_poly_path = "lane_polys.json"
            
        self.poly_path = rospy.get_param("~roi_source/path", default_poly_path)
        
        # 3. 파라미터
        # 검사할 차선 목록 (예: ["IN1"] 이면 IN1 차선 차량만 통과)
        self.target_lanes = rospy.get_param("~policy/target_lanes", ["IN1", "IN2", "IN3", "OUT1", "OUT2", "OUT3"])
        self.anchor_point = rospy.get_param("~policy/anchor_point", "bbox_bottom_center") # or bbox_center
        self.reload_sec = 1.0

        # 내부 변수
        self.lane_polys = {} # { "IN1": np.array([[x,y],...]), ... }
        self._last_poly_load = 0.0
        self._poly_mtime = 0.0

        # Pub/Sub
        self.pub = rospy.Publisher(self.pub_topic, DetectionArray, queue_size=10)
        self.sub = rospy.Subscriber(self.sub_topic, DetectionArray, self.callback, queue_size=10)

        rospy.loginfo(f"[LaneDetector] In: {self.sub_topic}, Out: {self.pub_topic}")
        rospy.loginfo(f"[LaneDetector] ROI Path: {self.poly_path}")
        
        # 초기 로드 시도
        self._load_polys()

    def _load_polys(self):
        """lane_polys.json 파일이 변경되었으면 다시 로드"""
        now = time.time()
        # 너무 잦은 파일 접근 방지
        if now - self._last_poly_load < self.reload_sec:
            return

        self._last_poly_load = now
        
        if not os.path.exists(self.poly_path):
            # 파일이 아직 없으면 경고는 가끔만 출력
            # rospy.logwarn_throttle(10.0, f"[LaneDetector] Poly file not found: {self.poly_path}")
            return

        try:
            mtime = os.stat(self.poly_path).st_mtime
            if mtime <= self._poly_mtime:
                return # 변경 없음

            with open(self.poly_path, "r") as f:
                data = json.load(f)
            
            new_polys = {}
            for name, points in data.items():
                if points and len(points) > 2:
                    # cv2.pointPolygonTest를 위해 int32 numpy 배열로 변환
                    new_polys[name] = np.array(points, dtype=np.int32)
            
            self.lane_polys = new_polys
            self._poly_mtime = mtime
            rospy.loginfo(f"[LaneDetector] Reloaded {len(new_polys)} lanes from json.")
            
        except Exception as e:
            rospy.logwarn(f"[LaneDetector] Failed to load polys: {e}")

    def is_inside(self, bbox, poly):
        """bbox의 기준점이 폴리곤 안에 있는지 판별"""
        if poly is None or cv2 is None: 
            return False

        xmin, ymin = bbox.xmin, bbox.ymin
        xmax, ymax = bbox.xmax, bbox.ymax
        
        # 기준점 계산 (차량의 바닥 중앙 or 박스 중앙)
        if self.anchor_point == "bbox_center":
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
        else: # bbox_bottom_center (기본값) - 차량 바퀴 위치 기준이 정확함
            cx = (xmin + xmax) / 2.0
            cy = float(ymax)

        # cv2.pointPolygonTest 사용 (dist >= 0 이면 내부 또는 경계)
        result = cv2.pointPolygonTest(poly, (cx, cy), False)
        return result >= 0

    def callback(self, msg: DetectionArray):
        # 1. ROI 최신화 체크
        self._load_polys()

        # 차선 설정이 아예 없으면? -> 안전하게 모두 통과시킬지, 막을지 결정
        # 여기서는 "설정 전에는 모두 통과" 시켜서 디버깅을 돕습니다.
        if not self.lane_polys:
            self.pub.publish(msg)
            return

        filtered_dets = []
        
        # 2. 각 차량 검사
        for det in msg.detections:
            is_valid = False
            matched_lane = None
            
            # target_lanes에 있는 차선들만 검사
            for lane_name in self.target_lanes:
                poly = self.lane_polys.get(lane_name)
                if poly is not None:
                    if self.is_inside(det.bbox, poly):
                        is_valid = True
                        matched_lane = lane_name
                        break # 하나라도 속하면 통과
            
            if is_valid:
                det.lane_id = matched_lane if matched_lane is not None else ""
                filtered_dets.append(det)

        # 3. 결과 발행
        out_msg = DetectionArray()
        out_msg.header = msg.header # 타임스탬프 유지
        out_msg.detections = filtered_dets
        self.pub.publish(out_msg)

def main():
    try:
        LaneDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()