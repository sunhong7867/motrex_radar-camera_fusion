#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
speed_estimator_node.py (Final)

역할:
- Association Node에서 1차 융합된 데이터(AssociationArray)를 수신
- [1단계] Gating: 너무 먼 거리, 적은 레이더 포인트, 비정상 속도 제거
- [2단계] Bias Correction: 거리에 따른 레이더 속도 오차 보정 (CARLA 로직 이식)
- [3단계] Kalman Filter: 차량 ID별 1D 칼만 필터로 속도 노이즈 제거 (Smoothing)
- 최종 정제된 속도를 담아 다시 AssociationArray로 발행

데이터 흐름:
[Association Node] (/autocal/associated)
       ⬇
[Speed Estimator] (Gating -> Bias -> KF)
       ⬇
[Final Output] (/autocal/associated_speed)
"""

import rospy
import numpy as np
from collections import deque
from autocal.msg import AssociationArray, Association

# =========================================================
# 1. 1D Kalman Filter (CARLA data_collection.py 로직 그대로 이식)
# =========================================================
class Kalman1D:
    def __init__(self, q=0.05, r=4.0):
        self.x = None  # 상태 (속도)
        self.p = 1.0   # 공분산
        self.q = q     # 프로세스 노이즈 (낮을수록 예측 믿음)
        self.r = r     # 측정 노이즈 (높을수록 측정 덜 믿음 -> 스무딩 강해짐)

    def update(self, z):
        if self.x is None:
            self.x = z
            return z
        
        # 예측 (Prediction) - 속도는 일정하다고 가정 (Constant Velocity Model)
        self.p += self.q
        
        # 보정 (Correction)
        k = self.p / (self.p + self.r)
        self.x += k * (z - self.x)
        self.p *= (1 - k)
        return self.x

# =========================================================
# 2. Speed Estimator Node
# =========================================================
class SpeedEstimatorNode:
    def __init__(self):
        rospy.init_node("speed_estimator_node")

        # --- 파라미터 설정 ---
        # 입력/출력 토픽
        self.sub_topic = rospy.get_param("~in_topic", "/autocal/associated")
        self.pub_topic = rospy.get_param("~out_topic", "/autocal/associated_speed")

        # Gating 파라미터 (튀는 값 제거용)
        self.min_dist = float(rospy.get_param("~gate/min_dist_m", 5.0))
        self.max_dist = float(rospy.get_param("~gate/max_dist_m", 150.0))
        self.min_points = int(rospy.get_param("~gate/min_points", 1)) # 최소 1개 이상 매칭되어야 함
        self.max_speed_kph = rospy.get_param("~gate/max_speed_kph", 100.0) # 100km/h 이상 노이즈
        self.min_speed_kph = rospy.get_param("~gate/min_speed_kph", 5.0)   # 5km/h 이하 노이즈 (나뭇잎 등)

        # Bias Correction 파라미터 (거리별 속도 오차 보정)
        # 예: 50m에서 -1km/h, 100m에서 -2km/h 오차가 있다면 이를 보상
        self.bias_enable = bool(rospy.get_param("~bias/enable", True))
        # yaml 예시: [{dist: 0, bias: 0.0}, {dist: 100, bias: -2.0}]
        self.bias_table = rospy.get_param("~bias/segments", []) 
        self._bias_x = []
        self._bias_y = []
        if self.bias_enable and self.bias_table:
            # 보간(Interpolation)을 위해 배열로 변환
            sorted_segs = sorted(self.bias_table, key=lambda x: x['dist'])
            self._bias_x = [s['dist'] for s in sorted_segs]
            self._bias_y = [s['bias'] for s in sorted_segs]
            rospy.loginfo(f"[SpeedEstimator] Bias correction enabled with {len(self.bias_table)} segments.")

        # Kalman Filter 파라미터
        self.kf_enable = bool(rospy.get_param("~kf/enable", True))
        self.kf_q = float(rospy.get_param("~kf/q", 0.05))
        self.kf_r = float(rospy.get_param("~kf/r", 4.0))
        
        # 내부 상태 변수
        self.track_kfs = {}   # {id: Kalman1D}
        self.track_last_seen = {} # {id: timestamp}
        self.cleanup_interval = 2.0 # 2초 이상 안 보이면 KF 삭제

        # --- Pub/Sub ---
        self.pub = rospy.Publisher(self.pub_topic, AssociationArray, queue_size=10)
        self.sub = rospy.Subscriber(self.sub_topic, AssociationArray, self.callback, queue_size=10)

        rospy.loginfo(f"[SpeedEstimator] Ready. In: {self.sub_topic}, Out: {self.pub_topic}")

    def get_bias(self, dist):
        """거리에 따른 속도 오차 반환 (선형 보간)"""
        if not self.bias_enable or not self._bias_x:
            return 0.0
        return float(np.interp(dist, self._bias_x, self._bias_y))

    def get_kf(self, track_id, init_val):
        """차량 ID별 칼만 필터 관리"""
        if track_id not in self.track_kfs:
            self.track_kfs[track_id] = Kalman1D(q=self.kf_q, r=self.kf_r)
            self.track_kfs[track_id].x = init_val # 초기값 설정
        return self.track_kfs[track_id]

    def cleanup_kfs(self, current_time):
        """오래된 트랙의 칼만 필터 삭제 (메모리 관리)"""
        dead_ids = []
        for tid, last_t in self.track_last_seen.items():
            if (current_time - last_t) > self.cleanup_interval:
                dead_ids.append(tid)
        for tid in dead_ids:
            del self.track_kfs[tid]
            del self.track_last_seen[tid]

    def callback(self, msg: AssociationArray):
        out_msg = AssociationArray()
        out_msg.header = msg.header
        out_msg.radar_header = msg.radar_header
        out_msg.objects = []

        now = rospy.Time.now().to_sec()

        for obj in msg.objects:
            # -------------------------------
            # 1. Gating (이상치 제거)
            # -------------------------------
            # 거리, 포인트 수, 속도 범위 체크
            is_valid = True
            if not (self.min_dist <= obj.dist_m <= self.max_dist): is_valid = False
            if obj.num_radar_points < self.min_points: is_valid = False
            
            raw_speed = obj.speed_kph
            # NaN이거나, 5km/h 미만(너무 느림) 또는 100km/h 초과(너무 빠름)인 경우 제거
            if np.isnan(raw_speed) or (abs(raw_speed) < self.min_speed_kph) or (abs(raw_speed) > self.max_speed_kph):
                is_valid = False

            if not is_valid:
                # 조건 불만족 시 NaN 처리하여 내보냄 (UI에서 무시하도록)
                obj.speed_mps = float('nan')
                obj.speed_kph = float('nan')
                out_msg.objects.append(obj)
                continue

            # -------------------------------
            # 2. Bias Correction (오차 보정)
            # -------------------------------
            # V_corrected = V_raw - Bias(dist)
            bias = self.get_bias(obj.dist_m)
            corr_speed = raw_speed - bias

            # -------------------------------
            # 3. Kalman Filter (스무딩)
            # -------------------------------
            final_speed = corr_speed
            if self.kf_enable and obj.id > 0: # ID가 할당된 차량만 트래킹
                kf = self.get_kf(obj.id, corr_speed)
                final_speed = kf.update(corr_speed)
                self.track_last_seen[obj.id] = now
            
            # 최종 결과 업데이트
            obj.speed_kph = float(final_speed)
            obj.speed_mps = float(final_speed / 3.6)
            
            out_msg.objects.append(obj)

        # 오래된 트랙 정리
        self.cleanup_kfs(now)
        
        # 발행
        self.pub.publish(out_msg)

def main():
    try:
        SpeedEstimatorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()