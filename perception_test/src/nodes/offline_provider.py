#!/usr/bin/env python3
import os
import yaml
import rospy
import cv2
import struct
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header

# 패키지 이름이 perception_test 이므로 경로 설정 시 주의
from rospkg import RosPack

class OfflineProvider:
    def __init__(self):
        rospy.init_node('offline_provider', anonymous=True)
        
        # 1. 파라미터 로드 (launch 파일에서 넘어오는 경로들)
        # 기본값은 로컬 테스트용 경로로 설정
        r = RosPack()
        pkg_path = r.get_path('perception_test')
        
        self.video_path = rospy.get_param('~video_path', os.path.join(pkg_path, 'models/test_video.mp4'))
        self.log_path = rospy.get_param('~radar_log_path', os.path.join(pkg_path, 'models/log.txt'))
        self.intrinsic_path = rospy.get_param('~intrinsic_path', os.path.join(pkg_path, 'config/camera_intrinsic.yaml'))
        
        self.image_topic = rospy.get_param('~image_topic', '/camera/image_raw')
        self.info_topic = rospy.get_param('~camera_info_topic', '/camera/camera_info')
        self.radar_topic = rospy.get_param('~radar_topic', '/point_cloud')
        
        # 2. 퍼블리셔 설정
        self.pub_img = rospy.Publisher(self.image_topic, Image, queue_size=10)
        self.pub_info = rospy.Publisher(self.info_topic, CameraInfo, queue_size=10)
        self.pub_radar = rospy.Publisher(self.radar_topic, PointCloud2, queue_size=10)
        
        self.bridge = CvBridge()
        self.rate = rospy.Rate(30) # 비디오 FPS에 맞춤 (기본 30)

        # 3. 데이터 로드
        rospy.loginfo(f"Loading Camera Info from {self.intrinsic_path}...")
        self.camera_info = self.load_camera_info(self.intrinsic_path)
        
        rospy.loginfo(f"Loading Radar Log from {self.log_path}...")
        self.radar_msgs = self.load_radar_log(self.log_path)
        rospy.loginfo(f"Loaded {len(self.radar_msgs)} radar frames.")

    def load_camera_info(self, yaml_path):
        """camera_intrinsic.yaml 파일을 읽어 CameraInfo 메시지 생성"""
        ci = CameraInfo()
        try:
            with open(yaml_path, 'r') as f:
                calib_data = yaml.safe_load(f)
                
            # YAML 구조에 맞춰 파싱 (사용자가 올린 파일 구조 반영)
            # camera_intrinsic.yaml의 'camera_info' 섹션 참조
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

    def load_radar_log(self, log_path):
        """log.txt (YAML 형식)를 파싱하여 PointCloud2 메시지 리스트로 변환"""
        msgs = []
        try:
            with open(log_path, 'r') as f:
                # 파일 전체를 읽되, '---' 구분자가 있다면 여러 문서로 읽음
                documents = yaml.safe_load_all(f)
                
                for doc in documents:
                    if doc is None: continue
                    
                    pc = PointCloud2()
                    # Header 재구성 (나중에 publish 할 때 stamp는 덮어씀)
                    pc.header.frame_id = doc['header']['frame_id']
                    
                    pc.height = doc.get('height', 1)
                    pc.width = doc.get('width', 0)
                    
                    # Fields 재구성
                    pc.fields = []
                    for field_dict in doc.get('fields', []):
                        pf = PointField()
                        pf.name = field_dict['name']
                        pf.offset = field_dict['offset']
                        pf.datatype = field_dict['datatype']
                        pf.count = field_dict['count']
                        pc.fields.append(pf)
                    
                    pc.is_bigendian = doc.get('is_bigendian', False)
                    pc.point_step = doc.get('point_step', 20)
                    pc.row_step = doc.get('row_step', 0)
                    
                    # Data 변환: YAML 리스트 -> bytes
                    # log.txt에는 [177, 105, ...] 형태의 정수 리스트로 저장되어 있음
                    data_list = doc.get('data', [])
                    pc.data = bytes(data_list)
                    
                    msgs.append(pc)
                    
        except Exception as e:
            rospy.logerr(f"Failed to parse log.txt: {e}")
            
        return msgs

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            rospy.logerr(f"Could not open video file: {self.video_path}")
            return

        idx = 0
        total_radar_frames = len(self.radar_msgs)
        
        rospy.loginfo("Starting playback...")
        
        while not rospy.is_shutdown():
            ret, frame = cap.read()
            if not ret:
                rospy.loginfo("Video finished. Loop or Stop.")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 무한 반복
                idx = 0
                continue
                
            # 현재 시각 (동기화를 위해 모든 메시지에 동일 시간 적용)
            now = rospy.Time.now()
            
            # 1. Image Publish
            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            img_msg.header.stamp = now
            img_msg.header.frame_id = "camera_optical_frame"
            self.pub_img.publish(img_msg)
            
            # 2. CameraInfo Publish
            self.camera_info.header.stamp = now
            self.camera_info.header.frame_id = "camera_optical_frame"
            self.pub_info.publish(self.camera_info)
            
            # 3. Radar PointCloud Publish (있는 만큼만 순환)
            if total_radar_frames > 0:
                radar_msg = self.radar_msgs[idx % total_radar_frames]
                radar_msg.header.stamp = now
                # frame_id가 log.txt에 'retina_link'로 되어있으므로 그대로 둠 (TF 필요시 static_transform_publisher 사용)
                self.pub_radar.publish(radar_msg)
            
            idx += 1
            self.rate.sleep()

if __name__ == '__main__':
    try:
        node = OfflineProvider()
        node.run()
    except rospy.ROSInterruptException:
        pass