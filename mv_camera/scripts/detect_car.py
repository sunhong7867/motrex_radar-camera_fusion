#!/usr/bin/env python3
import rospy
import rospkg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
import cv2
import os
import sys

# ==========================================================
# 모드 선택 ('camera', 'file' 중 하나 선택)
# 'camera'  : 실시간 카메라 토픽을 받아 처리합니다.
# 'file' : 아래 FILE_PATH에 있는 이미지나 동영상을 처리합니다.
current_mode = 'file' 

# [Mode 1: ROS 설정] 카메라 노드에서 보내는 토픽 이름
ros_topic = "/camera/image_raw"

# [Mode 2: File 설정] 동영상 파일 경로 OR 이미지가 있는 폴더 경로
# 이미지(폴더): "/home/sukja/camera_data"
# 동영상: "/home/sukja/camera_data/video/MicrosoftTeams-video.mp4"
file_path = "/home/sukja/camera_data"

# 탐지할 대상 (COCO dataset 기준)
# 2:Car, 3:Motorcycle, 5:Bus, 7:Truck
target_classes = [2, 3, 5, 7]
# ==========================================================

class MasterDetector:
    def __init__(self):
        # 1. 노드 초기화
        rospy.init_node('master_detector', anonymous=True)
        
        # 2. 모델 경로 자동 설정 (mv_camera/weights/yolo11m.pt)
        rospack = rospkg.RosPack()
        try:
            package_path = rospack.get_path('mv_camera')
            self.model_path = os.path.join(package_path, 'weights', 'yolo11m.pt')
        except Exception as e:
            rospy.logerr(f"패키지 경로 오류: {e}")
            sys.exit(1)

        print(f"Loading YOLO model from: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.bridge = CvBridge()

    def run(self):
        if current_mode == 'camera':
            self.run_camera_mode()
        elif current_mode == 'file':
            self.run_file_mode()
        else:
            print(f"Error: 알 수 없는 모드입니다 -> {current_mode}")

    # ==========================================
    # 모드 1: 실시간 카메라 스트림 처리
    # ==========================================
    def run_camera_mode(self):
        print(f"\n[INFO] ROS 실시간 탐지 모드 시작! (Topic: {ros_topic})")
        # Subscriber 등록
        rospy.Subscriber(ros_topic, Image, self.image_callback)
        rospy.spin()

    def image_callback(self, data):
        try:
            # 카메라 이미지를 OpenCV 포맷으로 변환 (mono8)
            cv_image = self.bridge.imgmsg_to_cv2(data, "mono8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # YOLO 추론 (메모리 효율을 위해 stream=True 권장하지만 단일 프레임은 predict가 편함)
        # classes=[2] : Car만 탐지
        results = self.model.predict(source=cv_image, save=False, classes=[2], conf=0.5, verbose=False)

        # 결과 그리기
        annotated_frame = results[0].plot()

        # 화면 출력
        cv2.imshow("Real-time Camera Detection", annotated_frame)
        cv2.waitKey(1)

    # ==========================================
    # 모드 2: 저장된 파일(영상/사진) 처리
    # ==========================================
    def run_file_mode(self):
        print(f"\n[INFO] 파일 탐지 모드 시작! (Source: {file_path})")
        
        if not os.path.exists(file_path):
            print(f"[ERROR] 파일을 찾을 수 없습니다: {file_path}")
            return

        # 동영상, 이미지 폴더 모두 이 함수 하나로 해결됩니다.
        self.model.predict(
            source=file_path, 
            save=True,       # 결과 저장 여부
            # show=True,       # 화면에 실시간 출력 여부
            classes=target_classes ,
            conf=0.3,
            verbose=False
        )
        
        # (참고) 이미지 폴더의 경우 너무 빨리 지나가면 show=True 대신 
        # save=True만 쓰고 나중에 폴더 들어가서 확인하는 게 나을 수도 있습니다.

if __name__ == '__main__':
    try:
        detector = MasterDetector()
        detector.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
