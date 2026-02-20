#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
역할:
- 원본 검출 결과(`DetectionArray`)를 수신하고, 객체 bbox의 기준점이 지정된 차선 ROI 폴리곤 내부에 있는지 판별해 유효 객체만 통과시킨다.
- `lane_polys.json` 파일 변경을 주기적으로 감시해 런타임 중 차선 다각형을 재로딩하며, 운영 중 ROI 수정 내용을 즉시 반영한다.
- 필터링된 결과를 다음 단계 연관 노드에서 사용 가능한 `DetectionArray`로 발행하고, 매칭된 차선 이름을 `lane_id`에 기록한다.

입력:
- `~input_topic` (`autocal/DetectionArray`): 객체 검출 결과 입력 토픽.
- `~roi_source/path` (`json`): 차선 다각형 좌표 파일 경로.
- `~policy/target_lanes` (`list[str]`): 통과 허용 차선 목록.
- `~policy/anchor_point` (`str`): `bbox_bottom_center` 또는 `bbox_center` 기준점 모드.

출력:
- `~output_topic` (`autocal/DetectionArray`): 차선 필터링이 반영된 검출 결과.
"""

import json
import os
import time

import numpy as np
import rospy
import rospkg
from autocal.msg import DetectionArray

try:
    import cv2
except ImportError:
    cv2 = None

# ---------------------------------------------------------
# 차선 필터링 파라미터
# ---------------------------------------------------------
DEFAULT_INPUT_TOPIC = "/autocal/detections_raw"                      # 원본 검출 입력 토픽
DEFAULT_OUTPUT_TOPIC = "/autocal/detections"                        # 차선 필터링 출력 토픽
DEFAULT_TARGET_LANES = ["IN1", "IN2", "IN3", "OUT1", "OUT2", "OUT3"]  # 통과 허용 차선 목록
DEFAULT_ANCHOR_POINT = "bbox_bottom_center"                         # bbox 기준점 모드
DEFAULT_RELOAD_SEC = 1.0                                             # ROI 파일 재로딩 최소 주기(초)


class LaneDetectorNode:
    def __init__(self):
        """
        노드 파라미터, ROI 경로, 입출력 토픽, 내부 상태를 초기화
        """
        rospy.init_node("lane_detector_node")

        # 입력/출력 토픽 파라미터 로드
        self.sub_topic = rospy.get_param("~input_topic", DEFAULT_INPUT_TOPIC)
        self.pub_topic = rospy.get_param("~output_topic", DEFAULT_OUTPUT_TOPIC)

        # ROI 파일 기본 경로 계산
        try:
            rospack = rospkg.RosPack()
            pkg_path = rospack.get_path("autocal")
            default_poly_path = os.path.join(pkg_path, "src", "nodes", "lane_polys.json")
        except Exception:
            default_poly_path = "lane_polys.json"

        self.poly_path = rospy.get_param("~roi_source/path", default_poly_path)

        # 정책 파라미터 로드
        self.target_lanes = rospy.get_param("~policy/target_lanes", DEFAULT_TARGET_LANES)
        self.anchor_point = rospy.get_param("~policy/anchor_point", DEFAULT_ANCHOR_POINT)
        self.reload_sec = float(rospy.get_param("~runtime/reload_sec", DEFAULT_RELOAD_SEC))

        # ROI 캐시 상태
        self.lane_polys = {}
        self._last_poly_load = 0.0
        self._poly_mtime = 0.0

        # 퍼블리셔/서브스크라이버 연결
        self.pub = rospy.Publisher(self.pub_topic, DetectionArray, queue_size=10)
        self.sub = rospy.Subscriber(self.sub_topic, DetectionArray, self.callback, queue_size=10)

        rospy.loginfo(f"[LaneDetector] Ready. In={self.sub_topic}, Out={self.pub_topic}")
        rospy.loginfo(f"[LaneDetector] ROI path: {self.poly_path}")

        # 시작 시 ROI 파일 1회 로드
        self._load_polys()

    def _load_polys(self):
        """
        ROI JSON 파일 변경 여부를 확인하고 변경 시 다각형 캐시를 갱신
        """
        now = time.time()

        # 파일 접근 주기 제한
        if now - self._last_poly_load < self.reload_sec:
            return
        self._last_poly_load = now

        # ROI 파일 미존재 시 재시도 대기
        if not os.path.exists(self.poly_path):
            return

        try:
            mtime = os.stat(self.poly_path).st_mtime
            if mtime <= self._poly_mtime:
                return

            with open(self.poly_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 유효한 다각형(점 3개 이상)만 캐시에 반영
            new_polys = {}
            for lane_name, points in data.items():
                if points and len(points) > 2:
                    new_polys[lane_name] = np.array(points, dtype=np.int32)

            self.lane_polys = new_polys
            self._poly_mtime = mtime
            rospy.loginfo(f"[LaneDetector] Reloaded {len(new_polys)} lane polygons")

        except Exception as exc:
            rospy.logwarn(f"[LaneDetector] Failed to load ROI json: {exc}")

    def is_inside(self, bbox, poly) -> bool:
        """
        bbox 기준점이 다각형 내부 또는 경계에 위치하는지 판별
        """
        if poly is None or cv2 is None:
            return False

        xmin, ymin = bbox.xmin, bbox.ymin
        xmax, ymax = bbox.xmax, bbox.ymax

        # 기준점 모드에 따라 앵커 좌표 계산
        if self.anchor_point == "bbox_center":
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
        else:
            cx = (xmin + xmax) / 2.0
            cy = float(ymax)

        return cv2.pointPolygonTest(poly, (cx, cy), False) >= 0

    def callback(self, msg: DetectionArray):
        """
        입력 검출 리스트를 ROI 기준으로 필터링해 출력 토픽으로 발행
        """
        # 최신 ROI 반영
        self._load_polys()

        # ROI 미설정 상태에서는 전체 통과
        if not self.lane_polys:
            self.pub.publish(msg)
            return

        filtered_dets = []

        # 객체별 차선 포함 여부 검사
        for det in msg.detections:
            is_valid = False
            matched_lane = None

            for lane_name in self.target_lanes:
                poly = self.lane_polys.get(lane_name)
                if poly is not None and self.is_inside(det.bbox, poly):
                    is_valid = True
                    matched_lane = lane_name
                    break

            if is_valid:
                det.lane_id = matched_lane if matched_lane is not None else ""
                filtered_dets.append(det)

        # 필터링 결과 발행
        out_msg = DetectionArray()
        out_msg.header = msg.header
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
