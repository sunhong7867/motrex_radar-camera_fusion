#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import cv2
import numpy as np
import mvsdk
import platform
import rosparam
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge

def load_camera_info(camera_info_yaml):
    msg = CameraInfo()
    if not camera_info_yaml:
        return msg

    try:
        params, _ = rosparam.load_file(camera_info_yaml)[0]
        ci = params.get("camera_info", {})
        msg.distortion_model = ci.get("distortion_model", "plumb_bob")
        msg.D = ci.get("D", [])
        msg.K = ci.get("K", [0.0] * 9)
        msg.R = ci.get("R", [0.0] * 9)
        msg.P = ci.get("P", [0.0] * 12)
        rospy.loginfo(f"Loaded CameraInfo from {camera_info_yaml}")
    except Exception as exc:
        rospy.logwarn(f"Failed to load camera_info_yaml ({camera_info_yaml}): {exc}")

    return msg

def main():
    # 1. ROS 노드 초기화
    rospy.init_node('mindvision_camera_node', anonymous=True)
    
    # 2. 토픽 발행자(Publisher) 생성
    # 토픽 이름: /camera/image_raw
    pub = rospy.Publisher('/camera/image_raw', Image, queue_size=10)
    info_pub = rospy.Publisher('/camera/camera_info', CameraInfo, queue_size=10)
    bridge = CvBridge()
    camera_info_yaml = rospy.get_param('~camera_info_yaml', '')
    camera_optical_frame = rospy.get_param('~camera_optical_frame', 'camera_optical_frame')
    camera_info = load_camera_info(camera_info_yaml)
    
    # 카메라 열거
    DevList = mvsdk.CameraEnumerateDevice()
    nDev = len(DevList)
    if nDev < 1:
        rospy.logerr("No camera was found!")
        return

    DevInfo = DevList[0] # 첫 번째 카메라 선택
    rospy.loginfo(f"Connecting to {DevInfo.GetFriendlyName()}...")

    # 카메라 열기
    try:
        hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
    except mvsdk.CameraException as e:
        rospy.logerr(f"CameraInit Failed({e.error_code}): {e.message}")
        return

    # 설정 (데모 코드와 동일)
    cap = mvsdk.CameraGetCapability(hCamera)
    monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

    mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)

    mvsdk.CameraSetTriggerMode(hCamera, 0)
    mvsdk.CameraSetAeState(hCamera, 0)
    mvsdk.CameraSetExposureTime(hCamera, 30 * 1000) # 노출시간 30ms
    mvsdk.CameraPlay(hCamera)

    # 버퍼 할당
    FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)
    pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

    rospy.loginfo("Camera Started! Publishing to /camera/image_raw")
    # 창 크기를 마우스로 조절할 수 있도록 설정
    cv2.namedWindow("Camera View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera View", 640, 480)

    # 3. 메인 루프 (ROS가 죽을 때까지 반복)
    while not rospy.is_shutdown():
        try:
            # 카메라에서 이미지 가져오기
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)
            mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

            if platform.system() == "Windows":
                mvsdk.CameraFlipFrameBuffer(pFrameBuffer, FrameHead, 1)
            
            # 버퍼 -> Numpy 배열 변환
            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))

            # 4. ROS 메시지로 변환 후 발행
            msg = bridge.cv2_to_imgmsg(frame, "mono8")
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "camera_link"
            pub.publish(msg)

            info_msg = camera_info
            info_msg.header = msg.header
            info_msg.header.frame_id = camera_optical_frame
            if info_msg.width == 0:
                info_msg.width = FrameHead.iWidth
            if info_msg.height == 0:
                info_msg.height = FrameHead.iHeight
            info_pub.publish(info_msg)
            
            # 5. 카메라 영상 출력
            # cv2.imshow("Camera View", frame)
            # cv2.waitKey(1)

        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                rospy.logerr(f"CameraGetImageBuffer failed({e.error_code}): {e.message}")
            continue

    # 종료 처리
    mvsdk.CameraUnInit(hCamera)
    mvsdk.CameraAlignFree(pFrameBuffer)

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass