#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
역할:
- `AssociationArray`의 객체별 거리/속도를 게이팅, 바이어스 보정, 칼만 필터 스무딩 순서로 후처리해 안정적인 속도 값을 생성한다.
- 거리/포인트/속도 범위 조건을 만족하지 못한 항목은 NaN으로 마스킹해 후단 UI/로거가 이상치를 쉽게 배제하도록 한다.
- track id 기준으로 1D 칼만 필터를 관리하고, 오래 보이지 않는 트랙 필터 상태를 정리해 메모리 사용을 제어한다.

입력:
- `~in_topic` (`autocal/AssociationArray`): 연관 결과 입력.
- `~gate/*`, `~bias/*`, `~kf/*` 파라미터: 게이팅/보정/필터 설정.

출력:
- `~out_topic` (`autocal/AssociationArray`): 속도 후처리 결과.
"""

import numpy as np
import rospy
from autocal.msg import AssociationArray

# ---------------------------------------------------------
# 속도 추정 파라미터
# ---------------------------------------------------------
DEFAULT_GATE_MIN_DIST_M = 5.0                        # 유효 최소 거리(m)
DEFAULT_GATE_MAX_DIST_M = 150.0                      # 유효 최대 거리(m)
DEFAULT_GATE_MIN_POINTS = 1                          # 최소 레이더 매칭 포인트 수
DEFAULT_GATE_MAX_SPEED_KPH = 100.0                   # 최대 유효 속도(km/h)
DEFAULT_GATE_MIN_SPEED_KPH = 5.0                     # 최소 유효 속도(km/h)
DEFAULT_BIAS_ENABLE = True                           # 거리별 바이어스 보정 사용 여부
DEFAULT_BIAS_SEGMENTS = []                           # 거리-바이어스 구간 테이블
DEFAULT_KF_ENABLE = True                             # 칼만 필터 사용 여부
DEFAULT_KF_Q = 0.05                                  # 칼만 프로세스 노이즈
DEFAULT_KF_R = 4.0                                   # 칼만 측정 노이즈
DEFAULT_CLEANUP_INTERVAL = 2.0                       # 트랙 상태 정리 시간(초)

DEFAULT_IN_TOPIC = "/autocal/associated"            # 입력 연관 토픽
DEFAULT_OUT_TOPIC = "/autocal/associated_speed"     # 출력 속도 토픽

class Kalman1D:
    def __init__(self, q=0.05, r=4.0):
        """
        프로세스 노이즈와 측정 노이즈를 설정해 필터 상태를 초기화
        """
        self.q = q
        self.r = r
        self.x = None
        self.p = 1.0

    def update(self, z):
        """
        새로운 측정값을 반영해 추정 속도를 갱신
        """
        if self.x is None:
            self.x = float(z)
            return self.x

        self.p += self.q
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (float(z) - self.x)
        self.p *= (1 - k)
        return self.x


class SpeedEstimatorNode:
    def __init__(self):
        """
        파라미터, 필터 상태, 입출력 토픽을 초기화
        """
        rospy.init_node("speed_estimator_node")

        # 입력/출력 토픽
        self.sub_topic = rospy.get_param("~in_topic", DEFAULT_IN_TOPIC)
        self.pub_topic = rospy.get_param("~out_topic", DEFAULT_OUT_TOPIC)

        # 게이팅 파라미터
        self.min_dist = float(rospy.get_param("~gate/min_dist_m", DEFAULT_GATE_MIN_DIST_M))
        self.max_dist = float(rospy.get_param("~gate/max_dist_m", DEFAULT_GATE_MAX_DIST_M))
        self.min_points = int(rospy.get_param("~gate/min_points", DEFAULT_GATE_MIN_POINTS))
        self.max_speed_kph = float(rospy.get_param("~gate/max_speed_kph", DEFAULT_GATE_MAX_SPEED_KPH))
        self.min_speed_kph = float(rospy.get_param("~gate/min_speed_kph", DEFAULT_GATE_MIN_SPEED_KPH))

        # 바이어스 보정 파라미터
        self.bias_enable = bool(rospy.get_param("~bias/enable", DEFAULT_BIAS_ENABLE))
        self.bias_table = rospy.get_param("~bias/segments", DEFAULT_BIAS_SEGMENTS)
        self._bias_x = []
        self._bias_y = []
        if self.bias_enable and self.bias_table:
            sorted_segs = sorted(self.bias_table, key=lambda x: x["dist"])
            self._bias_x = [s["dist"] for s in sorted_segs]
            self._bias_y = [s["bias"] for s in sorted_segs]
            rospy.loginfo(f"[SpeedEstimator] Bias correction enabled: {len(self.bias_table)} segments")

        # 칼만 필터 파라미터
        self.kf_enable = bool(rospy.get_param("~kf/enable", DEFAULT_KF_ENABLE))
        self.kf_q = float(rospy.get_param("~kf/q", DEFAULT_KF_Q))
        self.kf_r = float(rospy.get_param("~kf/r", DEFAULT_KF_R))

        # 트랙별 필터 상태
        self.track_kfs = {}
        self.track_last_seen = {}
        self.cleanup_interval = float(rospy.get_param("~runtime/cleanup_interval", DEFAULT_CLEANUP_INTERVAL))

        self.pub = rospy.Publisher(self.pub_topic, AssociationArray, queue_size=10)
        self.sub = rospy.Subscriber(self.sub_topic, AssociationArray, self.callback, queue_size=10)

        rospy.loginfo(f"[SpeedEstimator] Ready. In={self.sub_topic}, Out={self.pub_topic}")

    def get_bias(self, dist):
        """
        거리값에 대응되는 속도 바이어스를 선형 보간으로 계산
        """
        if not self.bias_enable or not self._bias_x:
            return 0.0
        return float(np.interp(dist, self._bias_x, self._bias_y))

    def get_kf(self, track_id, init_val):
        """
        track id별 칼만 필터를 생성하거나 기존 필터를 반환
        """
        if track_id not in self.track_kfs:
            self.track_kfs[track_id] = Kalman1D(q=self.kf_q, r=self.kf_r)
            self.track_kfs[track_id].x = init_val
        return self.track_kfs[track_id]

    def cleanup_kfs(self, current_time):
        """
        마지막 관측 시각이 오래된 트랙 필터 상태를 삭제
        """
        dead_ids = []
        for tid, last_t in self.track_last_seen.items():
            if (current_time - last_t) > self.cleanup_interval:
                dead_ids.append(tid)

        for tid in dead_ids:
            del self.track_kfs[tid]
            del self.track_last_seen[tid]

    def callback(self, msg: AssociationArray):
        """
        객체별 속도를 게이팅/보정/스무딩한 뒤 결과 메시지를 발행
        """
        out_msg = AssociationArray()
        out_msg.header = msg.header
        out_msg.radar_header = msg.radar_header
        out_msg.objects = []

        now = rospy.Time.now().to_sec()

        for obj in msg.objects:
            # 거리/포인트/속도 게이팅
            is_valid = True
            if not (self.min_dist <= obj.dist_m <= self.max_dist):
                is_valid = False
            if obj.num_radar_points < self.min_points:
                is_valid = False

            raw_speed = obj.speed_kph
            if np.isnan(raw_speed) or (abs(raw_speed) < self.min_speed_kph) or (abs(raw_speed) > self.max_speed_kph):
                is_valid = False

            if not is_valid:
                obj.speed_mps = float("nan")
                obj.speed_kph = float("nan")
                out_msg.objects.append(obj)
                continue

            # 거리별 바이어스 보정
            bias = self.get_bias(obj.dist_m)
            corr_speed = raw_speed - bias

            # track id 기반 칼만 스무딩
            final_speed = corr_speed
            if self.kf_enable and obj.id > 0:
                kf = self.get_kf(obj.id, corr_speed)
                final_speed = kf.update(corr_speed)
                self.track_last_seen[obj.id] = now

            obj.speed_kph = float(final_speed)
            obj.speed_mps = float(final_speed / 3.6)
            out_msg.objects.append(obj)

        self.cleanup_kfs(now)
        self.pub.publish(out_msg)


def main():
    try:
        SpeedEstimatorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
