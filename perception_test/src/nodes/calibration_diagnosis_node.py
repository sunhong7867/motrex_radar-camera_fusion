#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import numpy as np
import rospy
import rospkg
import message_filters
from std_msgs.msg import Bool, String
from sensor_msgs.msg import PointCloud2, CameraInfo
import sensor_msgs.point_cloud2 as pc2
from perception_test.msg import AssociationArray

class CalibrationDiagnosisNode:
    def __init__(self):
        rospy.init_node("calibration_diagnosis_node", anonymous=False)

        # 1. 파라미터
        self.assoc_topic = rospy.get_param("~topics/association", "/perception_test/associated")
        self.radar_topic = rospy.get_param("~topics/radar", "/point_cloud")
        self.camera_info_topic = rospy.get_param("~topics/camera_info", "/camera/camera_info")
        
        self.trigger_topic = "/perception_test/diagnosis/start"
        self.result_topic = "/perception_test/diagnosis/result"

        rp = rospkg.RosPack()
        default_ext_path = os.path.join(rp.get_path("perception_test"), "config", "extrinsic.json")
        self.extrinsic_path = rospy.get_param("~extrinsic_path", default_ext_path)

        # 2. 설정값 (진단 조건)
        self.collection_duration = 30.0  # 30초
        self.min_dist_m = 30.0           # 30m 이상
        self.max_side_m = 10.0           # 10m 이내 (모든 차선)
        self.min_samples = 5             # 5개 이상이면 진단 성공 간주
        self.pixels_per_key_step = 7.2 

        # 상태 변수
        self.collecting = False
        self.start_time = rospy.Time(0)
        self.samples_u = []
        self.samples_v = []
        self.cam_K = None
        self.R = np.eye(3)
        self.t = np.zeros((3,1))

        self.load_extrinsic()

        # 3. 통신 연결
        self.pub_result = rospy.Publisher(self.result_topic, String, queue_size=1)
        rospy.Subscriber(self.trigger_topic, Bool, self.on_trigger)

        # 동기화 (Slop 0.3s)
        self.sub_assoc = message_filters.Subscriber(self.assoc_topic, AssociationArray)
        self.sub_radar = message_filters.Subscriber(self.radar_topic, PointCloud2)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.sub_assoc, self.sub_radar], 20, 0.3)
        self.ts.registerCallback(self.data_callback)
        
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self.cam_info_cb)

        rospy.loginfo("[Diagnosis] Node Ready. Waiting for trigger...")

    def on_trigger(self, msg):
        if msg.data:
            rospy.loginfo(f"[Diagnosis] START collecting ({self.collection_duration}s)...")
            self.samples_u = []
            self.samples_v = []
            self.start_time = rospy.Time.now()
            self.collecting = True

    def load_extrinsic(self):
        if os.path.exists(self.extrinsic_path):
            try:
                with open(self.extrinsic_path, "r") as f:
                    data = json.load(f)
                self.R = np.array(data["R"], dtype=np.float64)
                self.t = np.array(data["t"], dtype=np.float64).reshape(3,1)
            except: pass

    def cam_info_cb(self, msg):
        self.cam_K = np.array(msg.K).reshape(3,3)

    def project(self, pts_3d):
        if self.cam_K is None: return np.zeros((0,2))
        pts_cam = (self.R @ pts_3d.T) + self.t
        valid = pts_cam[2, :] > 0.1
        pts_cam = pts_cam[:, valid]
        if pts_cam.shape[1] == 0: return np.zeros((0,2))
        uv_h = self.cam_K @ pts_cam
        uv = (uv_h[:2, :] / uv_h[2, :]).T
        return uv

    def data_callback(self, assoc_msg, radar_msg):
        if not self.collecting: return

        # (시간 체크 로직 제거 - run 루프에서 처리함)

        gen = pc2.read_points(radar_msg, field_names=("x", "y", "z"), skip_nans=True)
        radar_np = np.array(list(gen), dtype=np.float32)
        if radar_np.shape[0] == 0: return

        mask = (radar_np[:, 0] > self.min_dist_m) & (np.abs(radar_np[:, 1]) < self.max_side_m)
        target_radar = radar_np[mask]
        if target_radar.shape[0] == 0: return
        
        uv_proj = self.project(target_radar)
        if uv_proj.shape[0] == 0: return

        for obj in assoc_msg.objects:
            if obj.dist_m < self.min_dist_m: continue
            bbox = obj.bbox
            bx_center = (bbox.xmin + bbox.xmax) / 2.0
            by_center = (bbox.ymin + bbox.ymax) / 2.0
            
            in_box = (uv_proj[:,0] >= bbox.xmin) & (uv_proj[:,0] <= bbox.xmax) & \
                     (uv_proj[:,1] >= bbox.ymin) & (uv_proj[:,1] <= bbox.ymax)
            
            matched = uv_proj[in_box]
            if matched.shape[0] > 0:
                rx = np.median(matched[:, 0])
                ry = np.median(matched[:, 1])
                self.samples_u.append(bx_center - rx)
                self.samples_v.append(by_center - ry)

    def analyze_and_report(self):
        self.collecting = False # 수집 중단
        n = len(self.samples_u)
        rospy.loginfo(f"[Diagnosis] Time's up! Analyzed {n} samples.")

        if n < self.min_samples:
            msg = f"⚠ [진단 실패] 데이터 수신량 부족 ({n}개).\n" \
                  f" - 30초간 들어온 유효 데이터가 너무 적습니다.\n" \
                  f" - Time Sync 또는 30m 거리 내 차량 인식 여부를 확인하세요."
            self.pub_result.publish(msg)
            return

        med_u = np.median(self.samples_u)
        med_v = np.median(self.samples_v)
        
        yaw_deg = abs(med_u) / 72.0
        pitch_deg = abs(med_v) / 72.0
        
        lines = []
        lines.append(f"📊 진단 완료 (샘플 {n}개)")
        
        # Yaw
        if abs(med_u) < 5.0:
            lines.append("✅ Yaw(좌우): 정상")
        elif med_u > 0:
            lines.append(f"👉 Yaw 보정: [우측(CW)]으로 {yaw_deg:.2f}° 회전")
            lines.append(f"   (레이더가 왼쪽으로 {med_u:.1f}px 치우침)")
        else:
            lines.append(f"👉 Yaw 보정: [좌측(CCW)]으로 {yaw_deg:.2f}° 회전")
            lines.append(f"   (레이더가 오른쪽으로 {abs(med_u):.1f}px 치우침)")

        # Pitch
        if abs(med_v) < 5.0:
            lines.append("✅ Pitch(상하): 정상")
        elif med_v > 0:
            lines.append(f"👉 Pitch 보정: [아래]로 {pitch_deg:.2f}° 숙임")
            lines.append(f"   (레이더가 위로 {med_v:.1f}px 뜸)")
        else:
            lines.append(f"👉 Pitch 보정: [위]로 {pitch_deg:.2f}° 듦")
            lines.append(f"   (레이더가 아래로 {abs(med_v):.1f}px 꺼짐)")

        final_msg = "\n".join(lines)
        self.pub_result.publish(final_msg)
        
        self.load_extrinsic()

    # [핵심] 와치독 루프 (spin 대신 사용)
    def run(self):
        rate = rospy.Rate(10) # 10Hz (0.1초마다 체크)
        while not rospy.is_shutdown():
            if self.collecting:
                elapsed = (rospy.Time.now() - self.start_time).to_sec()
                if elapsed > self.collection_duration:
                    # 30초 지났으면 강제 리포트
                    self.analyze_and_report()
            rate.sleep()

if __name__ == "__main__":
    node = CalibrationDiagnosisNode()
    node.run() # spin() 대신 run() 실행