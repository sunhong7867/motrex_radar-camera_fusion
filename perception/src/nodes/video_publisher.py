#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
video_publisher.py

역할: 동영상 파일을 읽어서 /camera/image_raw 토픽으로 쏘아줍니다.
기능:
 - 별도 경로 지정이 없으면 자동으로 패키지 내 'models' 폴더의 영상을 찾습니다.
 - 실제 카메라 없이 실내에서 알고리즘을 검증할 때 사용합니다.
"""

import rospy
import cv2
import sys
import os
import rospkg  # 패키지 경로 탐색용
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def main():
    rospy.init_node('video_publisher_node', anonymous=True)
    
    # ---------------------------------------------------------
    # [수정] 영상 경로 자동 설정 로직
    # 1순위: 실행 시 입력한 _video_path 파라미터
    # 2순위: perception 패키지 내 models/test_video.mp4
    # ---------------------------------------------------------
    default_filename = "test_video.mp4" # models 폴더에 넣을 파일명
    
    try:
        rp = rospkg.RosPack()
        pkg_path = rp.get_path('perception')
        default_path = os.path.join(pkg_path, "models", default_filename)
    except Exception as e:
        rospy.logwarn(f"[VideoPub] 패키지 경로를 찾을 수 없어 기본 경로를 사용합니다: {e}")
        default_path = f"/home/sh/Videos/{default_filename}"

    # 파라미터 로드 (입력 없으면 위에서 만든 default_path 사용)
    video_path = rospy.get_param("~video_path", default_path)
    pub_topic = rospy.get_param("~pub_topic", "/camera/image_raw")
    loop = rospy.get_param("~loop", True)

    # 파일 존재 여부 확인
    if not os.path.exists(video_path):
        rospy.logerr(f"[VideoPub] Video file not found: {video_path}")
        rospy.logerr(f" -> 'models' 폴더에 '{default_filename}' 파일을 넣었는지 확인해주세요.")
        return

    # ---------------------------------------------------------
    # 퍼블리셔 및 영상 재생 설정
    # ---------------------------------------------------------
    pub = rospy.Publisher(pub_topic, Image, queue_size=10)
    bridge = CvBridge()
    cap = cv2.VideoCapture(video_path)
    
    # 영상 원본 FPS에 맞춰 발행 속도 조절 (기본 30fps)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rate = rospy.Rate(fps)

    rospy.loginfo(f"[VideoPub] Publishing {os.path.basename(video_path)} to {pub_topic} @ {fps:.1f} FPS")

    while not rospy.is_shutdown():
        ret, frame = cap.read()

        # 영상이 끝나면 처리 (반복 or 종료)
        if not ret:
            if loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                rospy.loginfo("[VideoPub] Video finished.")
                break

        # 흑백(Mono8) 변환 (시스템 파이프라인이 흑백을 기대함)
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        try:
            # ROS 메시지로 변환 및 발행
            msg = bridge.cv2_to_imgmsg(gray, encoding="mono8")
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "camera"
            pub.publish(msg)
        except Exception as e:
            rospy.logerr(f"[VideoPub] Publish Error: {e}")

        rate.sleep()

    cap.release()

if __name__ == '__main__':
    main()