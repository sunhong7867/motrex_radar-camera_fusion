#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
calibration_manager.py

역할:
1. 카메라의 Detection(YOLO) 결과와 레이더의 PointCloud를 수집
2. 움직이는 단일 차량을 타겟으로 가정하고, 데이터 쌍(3D Radar Centroid <-> 2D BBox Center) 축적
3. 일정 샘플 이상 모이면 cv2.solvePnP로 회전(R), 이동(t) 행렬 계산
4. extrinsic.json 파일로 저장 후 종료
"""

import rospy
import cv2
import numpy as np
import json
import os
import message_filters
import rospkg
from sensor_msgs.msg import PointCloud2, CameraInfo
from perception_test.msg import DetectionArray
import sensor_msgs.point_cloud2 as pc2

class CalibrationManager:
    def __init__(self):
        rospy.init_node('calibration_manager', anonymous=True)
        
        # 파라미터
        self.min_samples = 20    # 캘리브레이션에 필요한 최소 샘플 수
        self.samples_collected = 0
        
        # 데이터 저장소
        self.obj_points = [] # 3D Radar Points (Centroids)
        self.img_points = [] # 2D Camera Points (BBox Centers)
        
        # 카메라 내부 파라미터 (CameraInfo 수신 대기)
        self.K = None
        self.D = None

        # 경로 설정
        rospack = rospkg.RosPack()
        self.save_path = os.path.join(rospack.get_path('perception_test'), 'src', 'nodes', 'extrinsic.json')

        # 토픽 구독 (Time Synchronizer 사용)
        self.sub_det = message_filters.Subscriber('/perception_test/detections', DetectionArray)
        self.sub_rad = message_filters.Subscriber('/point_cloud', PointCloud2)
        
        # 동기화 (슬롭 0.1초)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.sub_det, self.sub_rad], 10, 0.1)
        self.ts.registerCallback(self.callback)

        # Camera Info는 한 번만 받으면 됨
        self.sub_info = rospy.Subscriber('/camera/camera_info', CameraInfo, self.info_callback)

        rospy.loginfo("[CalibManager] Waiting for data... (Drive a single car in front of sensors)")

    def info_callback(self, msg):
        if self.K is None:
            self.K = np.array(msg.K).reshape(3, 3).astype(np.float64)
            self.D = np.array(msg.D).astype(np.float64)
            rospy.loginfo("[CalibManager] CameraInfo Received.")

    def callback(self, det_msg, rad_msg):
        if self.K is None:
            return # 카메라 정보 없으면 대기

        # 1. 차량 검출 확인 (가장 신뢰도 높은 1개만 사용)
        if len(det_msg.detections) == 0:
            return
            
        # Score가 가장 높은 차량 선정
        best_det = max(det_msg.detections, key=lambda x: x.score)
        if best_det.score < 0.5: # 신뢰도 필터
            return

        # BBox 중심점 (u, v)
        bb = best_det.bbox
        u = (bb.xmin + bb.xmax) / 2.0
        v = (bb.ymin + bb.ymax) / 2.0

        # 2. 레이더 데이터 파싱
        # x, y, z, power, doppler
        gen = pc2.read_points(rad_msg, field_names=("x", "y", "z", "power"), skip_nans=True)
        points = np.array(list(gen))
        
        if len(points) == 0:
            return

        # 3. 레이더 클러스터링 (단일 타겟 가정)
        # Power가 일정 이상인 점들만 사용하여 노이즈 제거
        valid_mask = points[:, 3] > 0  # Power > 0 (필요시 임계값 조정)
        valid_points = points[valid_mask, :3] # XYZ만 추출

        if len(valid_points) < 3:
            return

        # 레이더 점들의 무게중심(Centroid) 계산
        # (더 정교하게 하려면 DBSCAN 등으로 클러스터링 해야 하지만, 단일 차량 실험 환경 가정)
        centroid = np.mean(valid_points, axis=0)
        
        # 거리 필터 (너무 가까우면 오차가 큼, 5m 이상)
        dist = np.linalg.norm(centroid)
        if dist < 5.0:
            return

        # 4. 데이터 저장
        self.obj_points.append(centroid)
        self.img_points.append([u, v])
        self.samples_collected += 1
        
        rospy.loginfo(f"[CalibManager] Sample {self.samples_collected}/{self.min_samples} | Dist: {dist:.1f}m")

        # 5. 충분히 모이면 계산 시작
        if self.samples_collected >= self.min_samples:
            self.run_calibration()

    def run_calibration(self):
        rospy.loginfo("[CalibManager] Computing Extrinsic Parameters...")
        
        obj_pts = np.array(self.obj_points, dtype=np.float64)
        img_pts = np.array(self.img_points, dtype=np.float64)

        # cv2.solvePnP (Iterative Method)
        # 초기값 없이 실행
        success, rvec, tvec = cv2.solvePnP(
            obj_pts, 
            img_pts, 
            self.K, 
            self.D, 
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:
            R, _ = cv2.Rodrigues(rvec)
            
            # 결과 출력
            print("\n=== Calibration Result ===")
            print("Rotation Matrix (R):\n", R)
            print("Translation Vector (t):\n", tvec)
            
            # JSON 저장
            data = {
                "R": R.tolist(),
                "t": tvec.flatten().tolist()
            }
            
            try:
                with open(self.save_path, 'w') as f:
                    json.dump(data, f, indent=4)
                rospy.loginfo(f"[CalibManager] Saved to {self.save_path}")
            except Exception as e:
                rospy.logerr(f"Save failed: {e}")
            
            rospy.signal_shutdown("Calibration Finished")
        else:
            rospy.logerr("solvePnP Failed. Try collecting more diverse samples.")
            # 실패하면 계속 수집
            self.min_samples += 10

if __name__ == '__main__':
    try:
        CalibrationManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass