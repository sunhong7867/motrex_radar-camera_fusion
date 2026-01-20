#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np

from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

from std_msgs.msg import Float32, Int32, Float32MultiArray

# -------------------------
# Tunable parameters
# -------------------------
MIN_SPEED_MPS = 0.2 / 3.6        # 0.2 km/h -> m/s (노이즈 제거용)
MIN_POWER = None                # power 필터 쓰고 싶으면 숫자로 (예: 0.0 ~ ?)
CLUSTER_MAX_DIST_M = 2.0        # 클러스터 이웃 거리
CLUSTER_MIN_POINTS = 5          # 클러스터 최소 포인트 수
MAX_FORWARD_M = 200.0           # y(전방) 너무 먼 포인트 컷
MIN_FORWARD_M = 0.0             # y(전방) 뒤쪽 컷 (0이면 뒤쪽 제거)


def cluster_points_xy(points_xy: np.ndarray, max_dist: float, min_points: int):
    """
    매우 가벼운 BFS 클러스터링
    points_xy: (N, 2)
    return: list of index arrays
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
        self.sub = rospy.Subscriber("/point_cloud", PointCloud2, self.cb, queue_size=1)

        # PlotJuggler-friendly scalar topics
        self.pub_closest_range = rospy.Publisher("/radar/closest_range_m", Float32, queue_size=10)
        self.pub_closest_speed_mps = rospy.Publisher("/radar/closest_rel_speed_mps", Float32, queue_size=10)
        self.pub_closest_speed_kph = rospy.Publisher("/radar/closest_rel_speed_kph", Float32, queue_size=10)
        self.pub_num_points = rospy.Publisher("/radar/num_points", Int32, queue_size=10)
        self.pub_num_moving = rospy.Publisher("/radar/num_moving_points", Int32, queue_size=10)
        self.pub_frame_dt = rospy.Publisher("/radar/frame_dt_ms", Float32, queue_size=10)

        # PlotJuggler XY scatter용 (접근/이탈 분리)
        self.pub_app_x = rospy.Publisher("/radar/app_x", Float32MultiArray, queue_size=1)
        self.pub_app_y = rospy.Publisher("/radar/app_y", Float32MultiArray, queue_size=1)
        self.pub_rec_x = rospy.Publisher("/radar/rec_x", Float32MultiArray, queue_size=1)
        self.pub_rec_y = rospy.Publisher("/radar/rec_y", Float32MultiArray, queue_size=1)

        self.prev_stamp = None

    def _publish_empty_xy(self):
        """프레임에 유효 포인트가 없을 때 PlotJuggler XY가 갱신되도록 빈 배열 publish"""
        self.pub_app_x.publish(Float32MultiArray(data=[]))
        self.pub_app_y.publish(Float32MultiArray(data=[]))
        self.pub_rec_x.publish(Float32MultiArray(data=[]))
        self.pub_rec_y.publish(Float32MultiArray(data=[]))

    def cb(self, msg: PointCloud2):
        # 1) frame_dt
        stamp = msg.header.stamp.to_sec()
        if self.prev_stamp is None:
            dt_ms = 0.0
        else:
            dt_ms = (stamp - self.prev_stamp) * 1000.0
        self.prev_stamp = stamp

        # 2) PointCloud2 -> numpy (x,y,z,power,doppler)
        pts = np.array(list(pc2.read_points(
            msg,
            field_names=("x", "y", "z", "power", "doppler"),
            skip_nans=True
        )), dtype=np.float32)

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
        doppler = pts[:, 4]  # m/s 가정

        # 3) ROI (전방만)
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

        # 4) moving points mask (노이즈 제거)
        moving_mask = np.abs(doppler) >= MIN_SPEED_MPS
        self.pub_num_moving.publish(Int32(int(np.sum(moving_mask))))

        # 4-1) PlotJuggler XY scatter publish (접근/이탈 분리)
        xv = x[moving_mask]
        yv = y[moving_mask]
        dv = doppler[moving_mask]

        if xv.size == 0:
            self._publish_empty_xy()
        else:
            app_mask = dv < 0.0   # 접근(음수)
            rec_mask = dv > 0.0   # 이탈(양수)

            self.pub_app_x.publish(Float32MultiArray(data=xv[app_mask].astype(np.float32).tolist()))
            self.pub_app_y.publish(Float32MultiArray(data=yv[app_mask].astype(np.float32).tolist()))
            self.pub_rec_x.publish(Float32MultiArray(data=xv[rec_mask].astype(np.float32).tolist()))
            self.pub_rec_y.publish(Float32MultiArray(data=yv[rec_mask].astype(np.float32).tolist()))

        # 5) 클러스터링은 moving 포인트로만
        mx = xv
        my = yv
        mz = z[moving_mask]
        md = dv

        if mx.size < CLUSTER_MIN_POINTS:
            # 백업: ROI 내 전체 포인트 중 가장 가까운 포인트
            r = np.sqrt(x * x + y * y + z * z)
            idx = int(np.argmin(r))
            closest_r = float(r[idx])
            closest_v = float(doppler[idx])
            self.pub_closest_range.publish(Float32(closest_r))
            self.pub_closest_speed_mps.publish(Float32(closest_v))
            self.pub_closest_speed_kph.publish(Float32(closest_v * 3.6))
            return

        pts_xy = np.column_stack((mx, my))
        clusters = cluster_points_xy(pts_xy, CLUSTER_MAX_DIST_M, CLUSTER_MIN_POINTS)

        if len(clusters) == 0:
            # 백업: 가장 가까운 moving point
            r = np.sqrt(mx * mx + my * my + mz * mz)
            idx = int(np.argmin(r))
            closest_r = float(r[idx])
            closest_v = float(md[idx])
            self.pub_closest_range.publish(Float32(closest_r))
            self.pub_closest_speed_mps.publish(Float32(closest_v))
            self.pub_closest_speed_kph.publish(Float32(closest_v * 3.6))
            return

        # 6) 각 클러스터의 대표 거리(median range)로 가장 가까운 클러스터 선택
        best_cr = None
        best_cd = None
        best_r = None

        for c in clusters:
            cx = mx[c]
            cy = my[c]
            cz = mz[c]
            cd = md[c]

            cr = np.sqrt(cx * cx + cy * cy + cz * cz)
            r_med = float(np.median(cr))

            if best_r is None or r_med < best_r:
                best_r = r_med
                best_cr = cr
                best_cd = cd

        closest_range = float(np.median(best_cr))   # m
        closest_speed = float(np.median(best_cd))   # m/s (doppler 중앙값)

        self.pub_closest_range.publish(Float32(closest_range))
        self.pub_closest_speed_mps.publish(Float32(closest_speed))
        self.pub_closest_speed_kph.publish(Float32(closest_speed * 3.6))


if __name__ == "__main__":
    rospy.init_node("radar_metrics_node", anonymous=True)
    node = RadarMetricsNode()
    rospy.loginfo("radar_metrics_node started. Publishing metrics + XY scatter for PlotJuggler.")
    rospy.spin()
