#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
speed_estimator_node.py

역할:
- association_node가 생성한 AssociationArray(거리/속도 raw)를 입력으로 받아,
  CARLA 코드에서 하던 것처럼
  1) 거리/품질 기반 Gate (너무 가까움, 점 부족, 비정상 속도 등)
  2) 거리 기반 speed bias 보정 (piecewise segment)
  3) track id 별 1D Kalman Filter로 속도 안정화
  를 적용한 "최종 속도"를 산출하여 publish

입력:
- ~in_topic (perception/AssociationArray)  [default: /perception/associated]

출력:
- ~out_topic (perception/AssociationArray) [default: /perception/associated_speed]

옵션 출력:
- ~debug/topic (std_msgs/String)           [default: /perception/speed_debug]

파라미터(튜닝 포인트):
- ~gate/min_dist_m (default: 25.0) : CARLA에서 가까운 거리 구간 튐 제거하던 정책 반영
- ~gate/max_dist_m (default: 200.0)
- ~gate/min_points (default: 3)
- ~gate/min_speed_kph (default: 1.0)
- ~gate/max_speed_kph (default: 250.0)
- ~use_abs_speed (default: True) : 접근/이탈 부호 확정 전이면 True 추천

- ~bias/enable (default: True)
- ~bias/segments (list of dict)
  예:
    [
      {max_dist: 25.0, bias_kph: -0.82},
      {max_dist: 35.0, bias_kph: -0.48},
      {max_dist: 1e9,  bias_kph: -0.41}
    ]
  적용: v_corr_kph = v_raw_kph - bias_kph

- ~kf/enable (default: True)
- ~kf/q (default: 0.05) : process noise
- ~kf/r (default: 6.0)  : measurement noise
- ~track_timeout_sec (default: 2.0) : 일정 시간 갱신 없으면 필터 제거
- ~fallback/hold_last_sec (default: 0.2) : 잠깐 detection이 끊겼을 때 이전 속도를 유지할지(선택)
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import rospy
from std_msgs.msg import String

from perception.msg import AssociationArray, Association


# ----------------------------
# CARLA 스타일 1D Kalman Filter
# ----------------------------
class SimpleKalman1D:
    """
    상태: speed_kph (1D)
    업데이트:
      pred: x = x, P = P + Q
      gain: K = P/(P+R)
      corr: x = x + K(z-x), P=(1-K)P
    """
    def __init__(self, x0: float, q: float, r: float):
        self.x = float(x0)
        self.P = 1.0
        self.Q = float(q)
        self.R = float(r)

    def update(self, z: float) -> float:
        z = float(z)
        # predict
        Pp = self.P + self.Q
        xp = self.x
        # update
        K = Pp / (Pp + self.R)
        self.x = xp + K * (z - xp)
        self.P = (1.0 - K) * Pp
        return self.x


@dataclass
class BiasSegment:
    max_dist: float
    bias_kph: float


def _param(name: str, default):
    return rospy.get_param(name, default) if rospy.has_param(name) else default


class SpeedEstimatorNode:
    def __init__(self):
        # topics
        self.in_topic = _param("~in_topic", "/perception/associated")
        self.out_topic = _param("~out_topic", "/perception/associated_speed")

        # gating
        self.min_dist_m = float(_param("~gate/min_dist_m", 25.0))
        self.max_dist_m = float(_param("~gate/max_dist_m", 200.0))
        self.min_points = int(_param("~gate/min_points", 3))
        self.min_speed_kph = float(_param("~gate/min_speed_kph", 1.0))
        self.max_speed_kph = float(_param("~gate/max_speed_kph", 250.0))
        self.use_abs_speed = bool(_param("~use_abs_speed", True))

        # bias correction
        self.bias_enable = bool(_param("~bias/enable", True))
        self.bias_segments: List[BiasSegment] = self._load_bias_segments()

        # Kalman filter
        self.kf_enable = bool(_param("~kf/enable", True))
        self.kf_q = float(_param("~kf/q", 0.05))
        self.kf_r = float(_param("~kf/r", 6.0))
        self.track_timeout_sec = float(_param("~track_timeout_sec", 2.0))

        # fallback hold-last (옵션)
        self.hold_last_sec = float(_param("~fallback/hold_last_sec", 0.0))

        # debug
        self.debug_enable = bool(_param("~debug/enable", True))
        self.debug_topic = _param("~debug/topic", "/perception/speed_debug")
        self.pub_debug = rospy.Publisher(self.debug_topic, String, queue_size=3) if self.debug_enable else None

        # state: track id -> filter + timestamps + last valid speed
        self._kf: Dict[int, SimpleKalman1D] = {}
        self._last_seen: Dict[int, float] = {}
        self._last_valid_speed: Dict[int, Tuple[float, float]] = {}  # id -> (ts, speed_kph)

        self.pub_out = rospy.Publisher(self.out_topic, AssociationArray, queue_size=5)
        rospy.Subscriber(self.in_topic, AssociationArray, self._on_assoc, queue_size=3)

        rospy.loginfo(f"[speed] in={self.in_topic} out={self.out_topic}")
        rospy.loginfo(f"[speed] gate: min_dist={self.min_dist_m}m max_dist={self.max_dist_m}m min_pts={self.min_points}")
        rospy.loginfo(f"[speed] bias_enable={self.bias_enable} kf_enable={self.kf_enable} (q={self.kf_q}, r={self.kf_r})")

    # -------------------------
    # Bias segments
    # -------------------------
    def _load_bias_segments(self) -> List[BiasSegment]:
        segs = _param("~bias/segments", None)
        out: List[BiasSegment] = []

        if isinstance(segs, list) and len(segs) > 0:
            for s in segs:
                try:
                    out.append(BiasSegment(max_dist=float(s["max_dist"]), bias_kph=float(s["bias_kph"])))
                except Exception:
                    pass

        # 기본값: CARLA에서 쓰던 값 계열(너가 이전에 쓰던 -0.82/-0.48/-0.41 패턴)
        if not out:
            out = [
                BiasSegment(max_dist=25.0, bias_kph=-0.82),
                BiasSegment(max_dist=35.0, bias_kph=-0.48),
                BiasSegment(max_dist=1e9,  bias_kph=-0.41),
            ]

        out.sort(key=lambda x: x.max_dist)
        return out

    def _bias_kph(self, dist_m: float) -> float:
        for seg in self.bias_segments:
            if dist_m < seg.max_dist:
                return float(seg.bias_kph)
        return float(self.bias_segments[-1].bias_kph)

    # -------------------------
    # Track cleanup & KF helper
    # -------------------------
    def _cleanup(self, now: float):
        dead = []
        for tid, ts in self._last_seen.items():
            if (now - ts) > self.track_timeout_sec:
                dead.append(tid)
        for tid in dead:
            self._last_seen.pop(tid, None)
            self._kf.pop(tid, None)
            # last_valid_speed는 hold 기능용이라 바로 제거 안 해도 되지만, 깔끔히 제거
            self._last_valid_speed.pop(tid, None)

    def _get_kf(self, tid: int, init_kph: float) -> SimpleKalman1D:
        if tid not in self._kf:
            self._kf[tid] = SimpleKalman1D(x0=init_kph, q=self.kf_q, r=self.kf_r)
        return self._kf[tid]

    # -------------------------
    # Main callback
    # -------------------------
    def _on_assoc(self, msg: AssociationArray):
        now = time.time()
        self._cleanup(now)

        out = AssociationArray()
        out.header = msg.header
        out.radar_header = msg.radar_header
        out.objects = []

        total = len(msg.objects)
        used = 0
        gated = 0
        nan_in = 0
        held = 0

        for obj in msg.objects:
            o = Association()
            o.id = int(obj.id)
            o.cls = int(obj.cls)
            o.score = float(obj.score)
            o.bbox = obj.bbox
            o.dist_m = float(obj.dist_m)
            o.num_radar_points = int(obj.num_radar_points)

            # -------------------------
            # Raw speed 입력 정리 (kph 우선, 없으면 mps->kph)
            # -------------------------
            v_kph = float(obj.speed_kph)
            if not np.isfinite(v_kph):
                v_mps = float(obj.speed_mps)
                if np.isfinite(v_mps):
                    v_kph = v_mps * 3.6

            if not np.isfinite(v_kph):
                nan_in += 1
                # hold-last (옵션)
                if self.hold_last_sec > 0 and o.id in self._last_valid_speed:
                    ts, last_kph = self._last_valid_speed[o.id]
                    if (now - ts) <= self.hold_last_sec:
                        o.speed_kph = float(last_kph)
                        o.speed_mps = float(last_kph / 3.6)
                        out.objects.append(o)
                        held += 1
                        continue

                o.speed_kph = float("nan")
                o.speed_mps = float("nan")
                out.objects.append(o)
                continue

            if self.use_abs_speed:
                v_kph = abs(v_kph)

            # -------------------------
            # Gate: points / dist / speed sanity
            # -------------------------
            if o.num_radar_points < self.min_points:
                gated += 1
                o.speed_kph = float("nan")
                o.speed_mps = float("nan")
                out.objects.append(o)
                continue

            if (not np.isfinite(o.dist_m)) or (o.dist_m < self.min_dist_m) or (o.dist_m > self.max_dist_m):
                gated += 1
                o.speed_kph = float("nan")
                o.speed_mps = float("nan")
                out.objects.append(o)
                continue

            if (v_kph < self.min_speed_kph) or (v_kph > self.max_speed_kph):
                gated += 1
                o.speed_kph = float("nan")
                o.speed_mps = float("nan")
                out.objects.append(o)
                continue

            # -------------------------
            # Bias correction (CARLA 방식)
            # v_corr = v_raw - bias(dist)
            # -------------------------
            v_corr = v_kph
            if self.bias_enable:
                v_corr = v_kph - self._bias_kph(o.dist_m)

            # -------------------------
            # Kalman smoothing (id 기반)
            # -------------------------
            v_final = v_corr
            if self.kf_enable and o.id >= 0:
                kf = self._get_kf(o.id, v_corr)
                v_final = float(kf.update(v_corr))
                self._last_seen[o.id] = now

            if self.use_abs_speed:
                v_final = abs(v_final)

            o.speed_kph = float(v_final)
            o.speed_mps = float(v_final / 3.6)

            out.objects.append(o)
            used += 1

            # hold-last 저장
            self._last_valid_speed[o.id] = (now, float(v_final))

        self.pub_out.publish(out)

        if self.pub_debug:
            self.pub_debug.publish(String(
                data=f"total={total} used={used} gated={gated} nan_in={nan_in} held={held} tracks={len(self._kf)}"
            ))


def main():
    rospy.init_node("speed_estimator_node", anonymous=False)
    SpeedEstimatorNode()
    rospy.spin()


if __name__ == "__main__":
    main()
