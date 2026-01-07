#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import cv2
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from datetime import datetime

class ImageSaverNode:
    def __init__(self):
        # 1. 노드 초기화
        rospy.init_node('image_saver_node', anonymous=True)
        
        # 2. 토픽 구독자(Subscriber) 설정
        # /camera/image_raw 토픽이 들어오면 self.image_callback 함수가 실행됨
        self.sub = rospy.Subscriber("/camera/image_raw", Image, self.image_callback)
        
        self.bridge = CvBridge()
        self.current_frame = None
        
        # 홈 디렉토리(~) 아래에 'camera_data' 폴더를 만들어 저장
        home_path = os.path.expanduser("~") 
        self.save_dir = os.path.join(home_path, "camera_data")
        # 폴더 없으면 생성
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            rospy.loginfo(f"Created new directory: {self.save_dir}")

        rospy.loginfo(f"Image Saver Ready! Press 's' to save images in {self.save_dir}")

        self.window_name = "Image Saver"
        # 창 크기를 마우스로 조절할 수 있도록 설정
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 640, 480)

    def image_callback(self, msg):
        try:
            # ROS 이미지 메시지를 OpenCV 포맷으로 변환
            # "passthrough"를 쓰면 흑백/컬러 상관없이 원본 그대로 가져옴
            self.current_frame = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

    def run(self):
        # ROS 루프 (rospy.spin() 대신 직접 루프를 돌려서 키 입력을 받음)
        rate = rospy.Rate(30) # 30Hz 루프
        
        while not rospy.is_shutdown():
            if self.current_frame is not None:
                # 영상 띄우기
                cv2.imshow(self.window_name, self.current_frame)
                
                # 키보드 입력 대기 (1ms)
                key = cv2.waitKey(1) & 0xFF
                
                # 's' 키를 누르면 저장
                if key == ord('s'):
                    self.save_image()
                
                # 'q' 키를 누르면 종료
                elif key == ord('q'):
                    rospy.signal_shutdown("User requested shutdown")
                    break

            rate.sleep()
        
        cv2.destroyAllWindows()

    def save_image(self):
        if self.current_frame is None:
            return

        # 파일명을 날짜_시간 기반으로 생성
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
        filename = f"capture_{timestamp}.jpg"
        filepath = os.path.join(self.save_dir, filename)
        
        # OpenCV로 이미지 저장
        cv2.imwrite(filepath, self.current_frame)
        rospy.loginfo(f"Saved: {filepath}")

if __name__ == '__main__':
    try:
        node = ImageSaverNode()
        node.run()
    except rospy.ROSInterruptException:
        pass