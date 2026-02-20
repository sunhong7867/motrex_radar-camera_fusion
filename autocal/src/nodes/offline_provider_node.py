#!/usr/bin/env python3
import os
import json
import yaml
import rospy
import rosbag
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from rospkg import RosPack

def _resolve_ros_path(path: str) -> str:
    if not isinstance(path, str):
        return path
    if "$(find autocal)" in path:
        rp = RosPack()
        return path.replace("$(find autocal)", rp.get_path("autocal"))
    return path

class OfflineProvider:
    def __init__(self):
        rospy.init_node('offline_provider', anonymous=False)
        
        # 1. 경로 및 파라미터 설정
        r = RosPack()
        pkg_path = r.get_path('autocal')

        # 오프라인 소스 JSON 설정 (bag 이름/경로/토픽을 한 파일에서 관리)
        default_sources_json = os.path.join(pkg_path, 'config', 'offline_sources.json')
        self.sources_json = _resolve_ros_path(rospy.get_param('~sources_json', default_sources_json))
        self.source_name = rospy.get_param('~source_name', '')
        
        # 기본값 설정 (launch 파일에서 변경 가능)
        self.bag_path = rospy.get_param('~bag_path', os.path.join(pkg_path, 'models/my_test_data.bag'))
        self.intrinsic_path = rospy.get_param('~intrinsic_path', os.path.join(pkg_path, 'config/camera_intrinsic.yaml'))
        
        # 토픽 이름 설정 (Bag 파일 내부의 토픽 명 -> 발행할 토픽 명)
        # 만약 Bag 파일 안의 토픽 이름과 시스템에서 쓰는 이름이 다르면 여기서 매칭해줍니다.
        self.bag_image_topic = rospy.get_param('~bag_image_topic', '/camera/image_raw')
        self.bag_radar_topic = rospy.get_param('~bag_radar_topic', '/point_cloud')
        
        self.out_image_topic = rospy.get_param('~out_image_topic', '/camera/image_raw')
        self.out_radar_topic = rospy.get_param('~out_radar_topic', '/point_cloud')
        self.out_info_topic = rospy.get_param('~out_info_topic', '/camera/camera_info')
        self._apply_source_config_from_json()

        # 2. 퍼블리셔 설정
        self.pub_img = rospy.Publisher(self.out_image_topic, Image, queue_size=10)
        self.pub_radar = rospy.Publisher(self.out_radar_topic, PointCloud2, queue_size=10)
        self.pub_info = rospy.Publisher(self.out_info_topic, CameraInfo, queue_size=10)
        
        # 3. CameraInfo 로드 (Bag 파일에 없는 경우를 대비해 수동 로드 유지)
        rospy.loginfo(f"Loading Camera Info from {self.intrinsic_path}...")
        self.camera_info = self.load_camera_info(self.intrinsic_path)

    def _apply_source_config_from_json(self):
        if not os.path.exists(self.sources_json):
            rospy.logwarn(f"sources_json not found: {self.sources_json}. Using ROS params only.")
            return
        try:
            with open(self.sources_json, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            sources = payload.get('sources', {}) if isinstance(payload, dict) else {}
            selected = self.source_name or payload.get('active_source', '')
            if not selected:
                return
            src = sources.get(selected)
            if not isinstance(src, dict):
                rospy.logwarn(f"source '{selected}' not found in {self.sources_json}. Using ROS params only.")
                return

            self.bag_path = _resolve_ros_path(src.get('bag_path', self.bag_path))
            self.intrinsic_path = _resolve_ros_path(src.get('intrinsic_path', self.intrinsic_path))
            self.bag_image_topic = src.get('bag_image_topic', self.bag_image_topic)
            self.bag_radar_topic = src.get('bag_radar_topic', self.bag_radar_topic)
            self.out_image_topic = src.get('out_image_topic', self.out_image_topic)
            self.out_radar_topic = src.get('out_radar_topic', self.out_radar_topic)
            self.out_info_topic = src.get('out_info_topic', self.out_info_topic)
            rospy.loginfo(f"Loaded offline source '{selected}' from {self.sources_json}")
        except Exception as exc:
            rospy.logwarn(f"Failed to load sources_json '{self.sources_json}': {exc}")
            
    def load_camera_info(self, yaml_path):
        """camera_intrinsic.yaml 파일을 읽어 CameraInfo 메시지 생성"""
        ci = CameraInfo()
        try:
            with open(yaml_path, 'r') as f:
                calib_data = yaml.safe_load(f)
                
            info_sec = calib_data.get('camera_info', {})
            ci.width = calib_data.get('camera_spec', {}).get('resolution', {}).get('width', 1920)
            ci.height = calib_data.get('camera_spec', {}).get('resolution', {}).get('height', 1200)
            ci.distortion_model = info_sec.get('distortion_model', 'plumb_bob')
            ci.D = info_sec.get('D', [0]*5)
            ci.K = info_sec.get('K', [0]*9)
            ci.R = info_sec.get('R', [1,0,0, 0,1,0, 0,0,1])
            ci.P = info_sec.get('P', [0]*12)
        except Exception as e:
            rospy.logwarn(f"Failed to parse camera_intrinsic.yaml: {e}. Using empty CameraInfo.")
        return ci

    def run(self):
        if not os.path.exists(self.bag_path):
            rospy.logerr(f"Bag file not found: {self.bag_path}")
            return

        rospy.loginfo(f"Starting playback from: {self.bag_path}")
        
        # 무한 반복 루프
        while not rospy.is_shutdown():
            bag = rosbag.Bag(self.bag_path)
            
            # 재생 속도 조절을 위한 시간 계산 변수
            start_time_real = None
            start_time_ros = None
            
            # Bag 파일 읽기
            for topic, msg, t in bag.read_messages(topics=[self.bag_image_topic, self.bag_radar_topic]):
                if rospy.is_shutdown():
                    break

                # 1. 시간 동기화 (재생 속도 맞추기)
                if start_time_real is None:
                    start_time_real = rospy.Time.now()
                    start_time_ros = t
                
                # 현재 메시지가 재생되어야 할 시간 계산
                time_diff = t - start_time_ros
                target_real_time = start_time_real + time_diff
                
                # 시간이 될 때까지 대기
                sleep_duration = target_real_time - rospy.Time.now()
                if sleep_duration.to_sec() > 0:
                    rospy.sleep(sleep_duration)

                # 2. 메시지 발행 (Timestamp는 현재 시간으로 갱신하여 발행)
                # 이렇게 해야 rviz나 다른 노드에서 시간 차이 경고가 안 뜹니다.
                current_time = rospy.Time.now()
                msg.header.stamp = current_time

                if topic == self.bag_image_topic:
                    msg.header.frame_id = "camera_optical_frame"
                    self.pub_img.publish(msg)
                    
                    # [중요] 이미지가 나갈 때 CameraInfo도 같이 쏴줍니다.
                    self.camera_info.header = msg.header
                    self.pub_info.publish(self.camera_info)
                    
                elif topic == self.bag_radar_topic:
                    # msg.header.frame_id = "retina_link" # 원본 유지
                    self.pub_radar.publish(msg)
            
            bag.close()
            rospy.loginfo("Looping playback...")

if __name__ == '__main__':
    try:
        node = OfflineProvider()
        node.run()
    except rospy.ROSInterruptException:
        pass