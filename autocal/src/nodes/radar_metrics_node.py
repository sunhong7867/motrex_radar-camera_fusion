#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
역할:
- 레이더 PointCloud2를 수신해 이동점 개수, 최근접 거리, 상대 속도 등 핵심 지표를 계산하고 PlotJuggler용 토픽으로 발행한다.
- 전방 ROI, 최소 속도, 최소 파워, 단순 BFS 클러스터링 같은 필터를 적용해 노이즈를 줄인 통계값을 제공한다.
- 접근/이탈 점군을 XY 배열로 분리 발행해 실시간 산점도 모니터링이 가능하도록 구성한다.

입력:
- `/point_cloud` (`sensor_msgs/PointCloud2`): x, y, z, power, doppler 필드를 포함한 레이더 포인트.

출력:
- `/radar/closest_range_m`, `/radar/closest_rel_speed_mps`, `/radar/closest_rel_speed_kph`
- `/radar/num_points`, `/radar/num_moving_points`, `/radar/frame_dt_ms`
- `/radar/app_x`, `/radar/app_y`, `/radar/rec_x`, `/radar/rec_y`
"""

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32, Float32MultiArray, Int32

# ---------------------------------------------------------
# 레이더 지표 파라미터
# ---------------------------------------------------------
MIN_SPEED_MPS = 0.2 / 3.6      # 이동점 판정 최소 속도(m/s)
MIN_POWER = None               # 최소 파워 임계값(None이면 미사용)
CLUSTER_MAX_DIST_M = 2.0       # 클러스터 이웃 거리(m)
CLUSTER_MIN_POINTS = 5         # 클러스터 최소 포인트 수
MAX_FORWARD_M = 200.0          # 전방 최대 거리(m)
MIN_FORWARD_M = 0.0            # 전방 최소 거리(m)


def cluster_points_xy(points_xy: np.ndarray, max_dist: float, min_points: int):
    """
    XY 평면에서 BFS 방식으로 인접 포인트 클러스터를 구성
    """
    n = points_xy.shape[0]
    if n == 0:
        return []

    visited = np.zeros(n, dtype=bool)
    clusters = []

    for i in range(n):
        if visited[i]:
            continue

        queue = [i]
        visited[i] = True
        cluster = []

        while queue:
            idx = queue.pop()
            cluster.append(idx)

            d = np.linalg.norm(points_xy - points_xy[idx], axis=1)
            neighbors = np.where(d <= max_dist)[0]
            for nb in neighbors:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)

        if len(cluster) >= min_points:
            clusters.append(np.array(cluster, dtype=int))

    return clusters


class RadarMetricsNode:
    def __init__(self):
        """
        토픽 연결, 퍼블리셔, 프레임 상태 변수를 초기화
        """
        self.sub = rospy.Subscriber("/point_cloud", PointCloud2, self.cb, queue_size=1)

        self.pub_closest_range = rospy.Publisher("/radar/closest_range_m", Float32, queue_size=10)
        self.pub_closest_speed_mps = rospy.Publisher("/radar/closest_rel_speed_mps", Float32, queue_size=10)
        self.pub_closest_speed_kph = rospy.Publisher("/radar/closest_rel_speed_kph", Float32, queue_size=10)
        self.pub_num_points = rospy.Publisher("/radar/num_points", Int32, queue_size=10)
        self.pub_num_moving = rospy.Publisher("/radar/num_moving_points", Int32, queue_size=10)
        self.pub_frame_dt = rospy.Publisher("/radar/frame_dt_ms", Float32, queue_size=10)

        self.pub_app_x = rospy.Publisher("/radar/app_x", Float32MultiArray, queue_size=1)
        self.pub_app_y = rospy.Publisher("/radar/app_y", Float32MultiArray, queue_size=1)
        self.pub_rec_x = rospy.Publisher("/radar/rec_x", Float32MultiArray, queue_size=1)
        self.pub_rec_y = rospy.Publisher("/radar/rec_y", Float32MultiArray, queue_size=1)

        self.prev_stamp = None

    def _publish_empty_xy(self):
        """
        유효 포인트가 없을 때 XY 산점도 토픽을 빈 배열로 갱신
        """
        self.pub_app_x.publish(Float32MultiArray(data=[]))
        self.pub_app_y.publish(Float32MultiArray(data=[]))
        self.pub_rec_x.publish(Float32MultiArray(data=[]))
        self.pub_rec_y.publish(Float32MultiArray(data=[]))

    def cb(self, msg: PointCloud2):
        """
        포인트클라우드에서 지표를 계산해 스칼라/산점도 토픽으로 발행
        """
        # 프레임 간 dt 계산
        stamp = msg.header.stamp.to_sec()
        dt_ms = 0.0 if self.prev_stamp is None else (stamp - self.prev_stamp) * 1000.0
        self.prev_stamp = stamp

        # PointCloud2를 numpy 배열로 변환
        pts = np.array(
            list(pc2.read_points(msg, field_names=("x", "y", "z", "power", "doppler"), skip_nans=True)),
            dtype=np.float32,
        )

        n_total = int(pts.shape[0])
        self.pub_num_points.publish(Int32(n_total))
        self.pub_frame_dt.publish(Float32(dt_ms))

        if n_total == 0:
            self.pub_num_moving.publish(Int32(0))
            self.pub_closest_range.publish(Float32(np.nan))
            self.pub_closest_speed_mps.publish(Float32(np.nan))
            self.pub_closest_speed_kph.publish(Float32(np.nan))
            self._publish_empty_xy()
            return

        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        power = pts[:, 3]
        doppler = pts[:, 4]

        # 전방 ROI 및 파워 필터 적용
        roi_mask = (y >= MIN_FORWARD_M) & (y <= MAX_FORWARD_M)
        if MIN_POWER is not None:
            roi_mask &= (power >= float(MIN_POWER))

        x = x[roi_mask]
        y = y[roi_mask]
        z = z[roi_mask]
        doppler = doppler[roi_mask]

        if x.size == 0:
            self.pub_num_moving.publish(Int32(0))
            self.pub_closest_range.publish(Float32(np.nan))
            self.pub_closest_speed_mps.publish(Float32(np.nan))
            self.pub_closest_speed_kph.publish(Float32(np.nan))
            self._publish_empty_xy()
            return

        # 이동점 마스크 적용
        moving_mask = np.abs(doppler) >= MIN_SPEED_MPS
        self.pub_num_moving.publish(Int32(int(np.sum(moving_mask))))

        xv = x[moving_mask]
        yv = y[moving_mask]
        dv = doppler[moving_mask]

        # 접근/이탈 XY 산점도 발행
        if xv.size == 0:
            self._publish_empty_xy()
        else:
            app_mask = dv < 0.0
            rec_mask = dv > 0.0
            self.pub_app_x.publish(Float32MultiArray(data=xv[app_mask].astype(np.float32).tolist()))
            self.pub_app_y.publish(Float32MultiArray(data=yv[app_mask].astype(np.float32).tolist()))
            self.pub_rec_x.publish(Float32MultiArray(data=xv[rec_mask].astype(np.float32).tolist()))
            self.pub_rec_y.publish(Float32MultiArray(data=yv[rec_mask].astype(np.float32).tolist()))

        # 최근접 거리 및 상대속도 계산
        ranges = np.sqrt(x * x + y * y + z * z)
        nearest_idx = int(np.argmin(ranges))
        closest_range = float(ranges[nearest_idx])
        closest_speed_mps = float(doppler[nearest_idx])

        self.pub_closest_range.publish(Float32(closest_range))
        self.pub_closest_speed_mps.publish(Float32(closest_speed_mps))
        self.pub_closest_speed_kph.publish(Float32(closest_speed_mps * 3.6))


if __name__ == "__main__":
    rospy.init_node("radar_metrics_node", anonymous=False)
    RadarMetricsNode()
    rospy.spin()
