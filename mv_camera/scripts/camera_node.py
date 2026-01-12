#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import mvsdk
import platform
import yaml
import os
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

def load_camera_info(yaml_path):
    """
    YAML 파일에서 카메라 내부 파라미터(Intrinsic)를 읽어 CameraInfo 메시지를 생성합니다.
    """
    if not os.path.exists(yaml_path):
        rospy.logerr(f"Calibration file not found at: {yaml_path}")
        return None

    try:
        with open(yaml_path, 'r') as f:
            calib_data = yaml.safe_load(f)
        
        info_msg = CameraInfo()
        info_msg.width = calib_data['image_width']
        info_msg.height = calib_data['image_height']
        
        # 행렬 데이터 변환 (K, D, R, P)
        info_msg.K = calib_data['camera_matrix']['data']
        info_msg.D = calib_data['distortion_coefficients']['data']
        info_msg.R = calib_data['rectification_matrix']['data']
        info_msg.P = calib_data['projection_matrix']['data']
        info_msg.distortion_model = calib_data['distortion_model']
        
        return info_msg
    except Exception as e:
        rospy.logerr(f"Failed to parse YAML: {e}")
        return None

def main():
    # 1. ROS 노드 초기화
    rospy.init_node('mindvision_camera_node', anonymous=True)
    
    # 2. 파라미터 및 경로 설정
    # 패키지 내 config 폴더에 있는 yaml 파일을 기본값으로 설정합니다.
    default_yaml = os.path.join(os.path.dirname(__file__), '../../config/camera_intrinsic.yaml')
    yaml_path = rospy.get_param('~camera_info_url', default_yaml)
    
    # 3. 토픽 발행자(Publisher) 생성
    pub = rospy.Publisher('/camera/image_raw', Image, queue_size=10)
    info_pub = rospy.Publisher('/camera/camera_info', CameraInfo, queue_size=10)
    bridge = CvBridge()
    
    # 4. 카메라 내부 파라미터 로드
    camera_info_msg = load_camera_info(yaml_path)
    if camera_info_msg is None:
        rospy.logwarn("CameraInfo topic will be published with empty values.")
        camera_info_msg = CameraInfo()

    # 5. 마인드비전 카메라 열거 및 초기화
    DevList = mvsdk.CameraEnumerateDevice()
    nDev = len(DevList)
    if nDev < 1:
        rospy.logerr("No camera was found!")
        return

    DevInfo = DevList[0] # 첫 번째 카메라 선택
    rospy.loginfo(f"Connecting to {DevInfo.GetFriendlyName()}...")

    try:
        # 카메라 초기화
        hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
    except mvsdk.CameraException as e:
        rospy.logerr(f"CameraInit Failed({e.error_code}): {e.message}")
        return

    # 6. 카메라 속성 설정
    cap = mvsdk.CameraGetCapability(hCamera)
    
    # 출력 포맷 설정 (YOLO 추론을 고려하여 필요 시 RGB로 변경 가능)
    # 현재는 흑백(MONO8) 설정입니다.
    mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)

    # 트리거 모드: 0(Continuous), AE: 0(Manual Exposure)
    mvsdk.CameraSetTriggerMode(hCamera, 0)
    mvsdk.CameraSetAeState(hCamera, 0)
    # 노출 시간 설정 (30ms = 30000us)
    mvsdk.CameraSetExposureTime(hCamera, 30 * 1000) 
    
    # 스트리밍 시작
    mvsdk.CameraPlay(hCamera)

    # 7. 프레임 버퍼 할당
    # FHD(1920x1080) 해상도 기준으로 버퍼를 생성합니다.
    FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * 3
    pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

    rospy.loginfo("MindVision Camera Node Started!")

    # 8. 메인 루프
    while not rospy.is_shutdown():
        try:
            # 카메라로부터 이미지 버퍼 가져오기 (타임아웃 200ms)
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)
            
            # ISP 처리 (Raw -> 이미지 변환)
            mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
            # 사용이 끝난 로우 버퍼 해제
            mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

            # 운영체제에 따른 상하 반전 처리
            if platform.system() == "Windows":
                mvsdk.CameraFlipFrameBuffer(pFrameBuffer, FrameHead, 1)
            
            # 버퍼 데이터를 Numpy 배열로 변환
            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            
            # 해상도에 맞게 리쉐이프 (1920x1080)
            channels = 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3
            frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, channels))

            # 9. ROS 메시지 생성 및 발행 (타임스탬프 동기화)
            current_time = rospy.Time.now()
            
            # 이미지 메시지
            encoding = "mono8" if channels == 1 else "bgr8"
            img_msg = bridge.cv2_to_imgmsg(frame, encoding)
            img_msg.header.stamp = current_time # 동일 타임스탬프 적용
            img_msg.header.frame_id = "camera_link"
            
            # 카메라 정보 메시지
            camera_info_msg.header.stamp = current_time # 동일 타임스탬프 적용
            camera_info_msg.header.frame_id = "camera_link"
            
            # 데이터 배포
            pub.publish(img_msg)
            info_pub.publish(camera_info_msg)

        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                rospy.logerr(f"Camera error({e.error_code}): {e.message}")
            continue

    # 10. 종료 처리
    mvsdk.CameraUnInit(hCamera)
    mvsdk.CameraAlignFree(pFrameBuffer)
    rospy.loginfo("Camera Node Shutdown.")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
