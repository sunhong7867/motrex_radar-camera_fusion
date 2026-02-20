#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
역할:
- 오프라인 bag 파일을 읽어 카메라 이미지/레이더 포인트클라우드를 실시간 재생처럼 재발행한다.
- intrinsic JSON을 읽어 `CameraInfo`를 구성하고 영상과 함께 발행해 후단 노드가 온라인과 동일하게 동작하도록 만든다.
- `offline_sources.json`의 source 프로파일을 적용해 bag 경로/토픽 매핑을 선택적으로 전환할 수 있다.

입력:
- `~sources_json` (`json`): 오프라인 소스 프로파일 파일 경로.
- `~source_name` (`str`): 선택할 소스 이름.
- `~bag_path`, `~intrinsic_path` (`str`): bag/intrinsic 파일 경로.
- `~bag_*_topic`, `~out_*_topic` (`str`): 입력 bag 토픽과 출력 토픽 매핑.

출력:
- `~out_image_topic` (`sensor_msgs/Image`): 재생 이미지 토픽.
- `~out_radar_topic` (`sensor_msgs/PointCloud2`): 재생 레이더 토픽.
- `~out_info_topic` (`sensor_msgs/CameraInfo`): 카메라 내부파라미터 토픽.
"""

import json
import os

import rosbag
import rospy
from rospkg import RosPack
from sensor_msgs.msg import CameraInfo, Image, PointCloud2

# ---------------------------------------------------------
# 오프라인 재생 파라미터
# ---------------------------------------------------------
DEFAULT_SOURCE_NAME = ""                              # offline_sources에서 사용할 소스 이름
DEFAULT_BAG_IMAGE_TOPIC = "/camera/image_raw"        # bag 내부 이미지 토픽
DEFAULT_BAG_RADAR_TOPIC = "/point_cloud"             # bag 내부 레이더 토픽
DEFAULT_OUT_IMAGE_TOPIC = "/camera/image_raw"        # 재발행 이미지 토픽
DEFAULT_OUT_RADAR_TOPIC = "/point_cloud"             # 재발행 레이더 토픽
DEFAULT_OUT_INFO_TOPIC = "/camera/camera_info"       # 재발행 CameraInfo 토픽


def _resolve_ros_path(path: str) -> str:
    """
    `$(find autocal)` 경로 패턴을 실제 절대 경로로 치환
    """
    if not isinstance(path, str):
        return path
    if "$(find autocal)" in path:
        rp = RosPack()
        return path.replace("$(find autocal)", rp.get_path("autocal"))
    return path


class OfflineProvider:
    def __init__(self):
        """
        오프라인 소스 설정, 토픽 매핑, 퍼블리셔, CameraInfo를 초기화
        """
        rospy.init_node("offline_provider", anonymous=False)

        r = RosPack()
        pkg_path = r.get_path("autocal")

        # 소스 JSON 및 기본 경로 파라미터 로드
        default_sources_json = os.path.join(pkg_path, "config", "offline_sources.json")
        self.sources_json = _resolve_ros_path(rospy.get_param("~sources_json", default_sources_json))
        self.source_name = rospy.get_param("~source_name", DEFAULT_SOURCE_NAME)

        self.bag_path = rospy.get_param("~bag_path", os.path.join(pkg_path, "models/my_test_data.bag"))
        self.intrinsic_path = rospy.get_param("~intrinsic_path", os.path.join(pkg_path, "config/intrinsic.json"))

        # bag 토픽과 출력 토픽 매핑 로드
        self.bag_image_topic = rospy.get_param("~bag_image_topic", DEFAULT_BAG_IMAGE_TOPIC)
        self.bag_radar_topic = rospy.get_param("~bag_radar_topic", DEFAULT_BAG_RADAR_TOPIC)
        self.out_image_topic = rospy.get_param("~out_image_topic", DEFAULT_OUT_IMAGE_TOPIC)
        self.out_radar_topic = rospy.get_param("~out_radar_topic", DEFAULT_OUT_RADAR_TOPIC)
        self.out_info_topic = rospy.get_param("~out_info_topic", DEFAULT_OUT_INFO_TOPIC)

        # source_name이 있으면 JSON 프로파일로 덮어쓰기
        self._apply_source_config_from_json()

        # 퍼블리셔 초기화
        self.pub_img = rospy.Publisher(self.out_image_topic, Image, queue_size=10)
        self.pub_radar = rospy.Publisher(self.out_radar_topic, PointCloud2, queue_size=10)
        self.pub_info = rospy.Publisher(self.out_info_topic, CameraInfo, queue_size=10)

        rospy.loginfo(f"[OfflineProvider] Loading CameraInfo: {self.intrinsic_path}")
        self.camera_info = self.load_camera_info(self.intrinsic_path)

    def _apply_source_config_from_json(self):
        """
        offline_sources.json에서 선택 소스 설정을 로드해 경로/토픽을 덮어쓰기
        """
        if not os.path.exists(self.sources_json):
            rospy.logwarn(f"[OfflineProvider] sources_json not found: {self.sources_json}")
            return

        try:
            with open(self.sources_json, "r", encoding="utf-8") as f:
                payload = json.load(f)

            sources = payload.get("sources", {}) if isinstance(payload, dict) else {}
            selected = self.source_name or payload.get("active_source", "")
            if not selected:
                return

            src = sources.get(selected)
            if not isinstance(src, dict):
                rospy.logwarn(f"[OfflineProvider] source '{selected}' not found")
                return

            self.bag_path = _resolve_ros_path(src.get("bag_path", self.bag_path))
            self.intrinsic_path = _resolve_ros_path(src.get("intrinsic_path", self.intrinsic_path))
            self.bag_image_topic = src.get("bag_image_topic", self.bag_image_topic)
            self.bag_radar_topic = src.get("bag_radar_topic", self.bag_radar_topic)
            self.out_image_topic = src.get("out_image_topic", self.out_image_topic)
            self.out_radar_topic = src.get("out_radar_topic", self.out_radar_topic)
            self.out_info_topic = src.get("out_info_topic", self.out_info_topic)

            rospy.loginfo(f"[OfflineProvider] Loaded source '{selected}'")

        except Exception as exc:
            rospy.logwarn(f"[OfflineProvider] Failed to load sources_json: {exc}")

    def load_camera_info(self, json_path: str) -> CameraInfo:
        """
        intrinsic JSON 내용을 CameraInfo 메시지로 변환
        """
        ci = CameraInfo()
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                calib_data = json.load(f)

            image_size = calib_data.get("image_size", {}) if isinstance(calib_data, dict) else {}
            ci.width = int(image_size.get("width", 1920))
            ci.height = int(image_size.get("height", 1200))
            ci.distortion_model = calib_data.get("distortion_model", "plumb_bob")

            d = calib_data.get("D", [0.0] * 5)
            k = calib_data.get("K", [[0.0] * 3 for _ in range(3)])
            r = calib_data.get("R", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            p = calib_data.get("P", [[0.0] * 4 for _ in range(3)])

            ci.D = [float(x) for x in d]
            ci.K = [float(x) for row in k for x in row]
            ci.R = [float(x) for row in r for x in row]
            ci.P = [float(x) for row in p for x in row]

        except Exception as exc:
            rospy.logwarn(f"[OfflineProvider] Failed to parse intrinsic json: {exc}")

        return ci

    def run(self):
        """
        bag 파일을 루프 재생하며 이미지/레이더/CameraInfo를 시각 동기화해 발행
        """
        if not os.path.exists(self.bag_path):
            rospy.logerr(f"[OfflineProvider] Bag file not found: {self.bag_path}")
            return

        rospy.loginfo(f"[OfflineProvider] Start playback: {self.bag_path}")

        while not rospy.is_shutdown():
            with rosbag.Bag(self.bag_path) as bag:
                start_time_real = None
                start_time_ros = None

                for topic, msg, t in bag.read_messages(topics=[self.bag_image_topic, self.bag_radar_topic]):
                    if rospy.is_shutdown():
                        break

                    if start_time_real is None:
                        start_time_real = rospy.Time.now().to_sec()
                        start_time_ros = t.to_sec()

                    elapsed_ros = t.to_sec() - start_time_ros
                    target_real = start_time_real + elapsed_ros
                    now_real = rospy.Time.now().to_sec()
                    sleep_dt = target_real - now_real
                    if sleep_dt > 0:
                        rospy.sleep(sleep_dt)

                    if topic == self.bag_image_topic:
                        msg.header.stamp = rospy.Time.now()
                        self.camera_info.header = msg.header
                        self.pub_img.publish(msg)
                        self.pub_info.publish(self.camera_info)
                    elif topic == self.bag_radar_topic:
                        msg.header.stamp = rospy.Time.now()
                        self.pub_radar.publish(msg)

            rospy.loginfo("[OfflineProvider] Replaying bag from beginning")


if __name__ == "__main__":
    try:
        node = OfflineProvider()
        node.run()
    except rospy.ROSInterruptException:
        pass
