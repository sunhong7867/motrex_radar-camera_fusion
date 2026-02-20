#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
역할:
- 레이더 포인트를 전처리하고 클러스터링해 차량 후보를 추출한 뒤 bbox 마커를 생성한다.
- 속도/거리/형상 조건으로 클러스터를 선별해 차량 유사 객체만 시각화한다.
- 디버그 마커, 윤곽선, 텍스트를 함께 발행해 파라미터 튜닝과 검증을 지원한다.

입력:
- `~in_topic` (`sensor_msgs/PointCloud2`): 레이더 포인트 입력.
- `~*` 파라미터: 전처리, 클러스터링, 차량 유사 필터, 마커 렌더링 설정.

출력:
- `~pub` (`visualization_msgs/MarkerArray`): 포인트/박스/텍스트 시각화 결과.
"""

import rospy
import numpy as np

from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


# ---------------------------------------------------------
# 레이더 클러스터 bbox 파라미터
# ---------------------------------------------------------
DEFAULT_MIN_SPEED_MPS = 1.0          # 이동점으로 간주할 최소 속도(m/s)
DEFAULT_MIN_POWER = None             # 최소 파워 필터(None이면 미사용)
DEFAULT_MAX_FORWARD_M = 200.0        # 전방 최대 거리 제한(m)
DEFAULT_MIN_FORWARD_M = 0.0          # 전방 최소 거리 제한(m)
DEFAULT_CLUSTER_MAX_DIST_M = 1.0     # 클러스터 이웃 거리 임계값(m)
DEFAULT_CLUSTER_MIN_POINTS = 4       # 클러스터 최소 포인트 수
DEFAULT_USE_VEHICLE_FILTER = True    # 차량 유사 필터 사용 여부
DEFAULT_VEH_MIN_WIDTH_M  = 0.6       # 차량 폭 하한(m)
DEFAULT_VEH_MAX_WIDTH_M  = 4.0       # 차량 폭 상한(m)
DEFAULT_VEH_MIN_LENGTH_M = 1.2       # 차량 길이 하한(m)
DEFAULT_VEH_MAX_LENGTH_M = 12.0      # 차량 길이 상한(m)
DEFAULT_MIN_ASPECT_YX = 1.4          # 최소 세로/가로 비율
DEFAULT_MAX_DOPPLER_STD = 3.0        # 도플러 표준편차 상한(m/s)
DEFAULT_MIN_MEDIAN_SPEED = 0.4 / 3.6  # m/s
DEFAULT_BBOX_PADDING_X = 0.30        # bbox 가로 패딩(m)
DEFAULT_BBOX_PADDING_Y = 0.80        # bbox 세로 패딩(m)
DEFAULT_TARGET_ASPECT_YX = 2.2       # 목표 세로/가로 비율
DEFAULT_TRIM_RATIO = 0.2             # 상하 20% 제거


def trimmed_mean(arr: np.ndarray, trim_ratio: float = 0.2) -> float:
    """
    상하 trim_ratio 비율을 제거한 뒤 평균.
    - n이 작아서 자를 게 없으면 그냥 mean을 반환
    """
    if arr is None:
        return float("nan")
    arr = np.asarray(arr, dtype=np.float32)
    n = arr.size
    if n == 0:
        return float("nan")
    if trim_ratio <= 0.0:
        return float(np.mean(arr))

    k = int(np.floor(n * trim_ratio))
    # 너무 작으면 트리밍 불가 -> 그냥 평균
    if n - 2 * k <= 0:
        return float(np.mean(arr))

    a = np.sort(arr)
    a = a[k:n - k]
    if a.size == 0:
        return float(np.mean(arr))
    return float(np.mean(a))


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


class RadarClusterBBoxViz:
    def __init__(self):
        """
        파라미터, 토픽, 렌더링 옵션, 구독/발행 객체를 초기화
        """
        # 토픽 및 좌표계 설정
        self.in_topic = rospy.get_param("~in_topic", "/point_cloud")
        self.pub_topic = rospy.get_param("~pub", "/radar/cluster_viz")

        # 프레임 선택 정책
        self.frame_id_param = rospy.get_param("~frame_id", "retina_link")
        self.use_msg_frame = rospy.get_param("~use_msg_frame", True)  # 권장: True

        # 렌더링 파라미터
        self.point_scale = rospy.get_param("~point_scale", 0.8)
        self.z_value = rospy.get_param("~z", 0.0)
        self.bbox_line_width = rospy.get_param("~bbox_width", 0.12)
        self.bbox_alpha = rospy.get_param("~bbox_alpha", 0.18)
        self.draw_text = rospy.get_param("~draw_text", False)
        self.text_size = rospy.get_param("~text_size", 3)
        self.marker_lifetime = rospy.get_param("~lifetime", 0.25)  # 마커 유지 시간(초)

        # 선택 표시 옵션
        self.only_nearest = rospy.get_param("~only_nearest", False)
        self.max_boxes = rospy.get_param("~max_boxes", 20)

        # 디버그 및 시각화 옵션
        self.debug = rospy.get_param("~debug", True)
        self.show_all_clusters = rospy.get_param("~show_all_clusters", True)
        self.use_vehicle_filter = rospy.get_param("~use_vehicle_filter", DEFAULT_USE_VEHICLE_FILTER)

        # 전처리 파라미터
        self.min_speed_mps = float(rospy.get_param("~min_speed_mps", DEFAULT_MIN_SPEED_MPS))
        self.min_power = rospy.get_param("~min_power", DEFAULT_MIN_POWER)
        self.max_forward_m = float(rospy.get_param("~max_forward_m", DEFAULT_MAX_FORWARD_M))
        self.min_forward_m = float(rospy.get_param("~min_forward_m", DEFAULT_MIN_FORWARD_M))

        # 클러스터링 파라미터
        self.cluster_max_dist_m = float(rospy.get_param("~cluster_max_dist_m", DEFAULT_CLUSTER_MAX_DIST_M))
        self.cluster_min_points = int(rospy.get_param("~cluster_min_points", DEFAULT_CLUSTER_MIN_POINTS))

        # 차량 유사 필터 파라미터
        self.veh_min_w = float(rospy.get_param("~veh_min_width_m", DEFAULT_VEH_MIN_WIDTH_M))
        self.veh_max_w = float(rospy.get_param("~veh_max_width_m", DEFAULT_VEH_MAX_WIDTH_M))
        self.veh_min_l = float(rospy.get_param("~veh_min_length_m", DEFAULT_VEH_MIN_LENGTH_M))
        self.veh_max_l = float(rospy.get_param("~veh_max_length_m", DEFAULT_VEH_MAX_LENGTH_M))
        self.min_aspect_yx = float(rospy.get_param("~min_aspect_yx", DEFAULT_MIN_ASPECT_YX))
        self.max_doppler_std = float(rospy.get_param("~max_doppler_std", DEFAULT_MAX_DOPPLER_STD))
        self.min_median_speed = float(rospy.get_param("~min_median_speed_mps", DEFAULT_MIN_MEDIAN_SPEED))

        # 도플러 강건 통계 파라미터
        self.trim_ratio = float(rospy.get_param("~trim_ratio", DEFAULT_TRIM_RATIO))

        # bbox 형상 보정 파라미터
        self.pad_x = float(rospy.get_param("~bbox_padding_x", DEFAULT_BBOX_PADDING_X))
        self.pad_y = float(rospy.get_param("~bbox_padding_y", DEFAULT_BBOX_PADDING_Y))
        self.target_aspect_yx = float(rospy.get_param("~target_aspect_yx", DEFAULT_TARGET_ASPECT_YX))

        self.sub = rospy.Subscriber(self.in_topic, PointCloud2, self.cb, queue_size=1)
        self.pub = rospy.Publisher(self.pub_topic, MarkerArray, queue_size=1)

    def _header(self, msg):
        """
        마커 발행에 사용할 frame_id와 stamp를 결정
        """
        frame_id = msg.header.frame_id if self.use_msg_frame and msg.header.frame_id else self.frame_id_param
        stamp = msg.header.stamp
        return frame_id, stamp

    def _mk_deleteall(self, frame_id, stamp):
        """
        이전 프레임 마커를 삭제하기 위한 DELETEALL 마커 생성
        """
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = stamp
        m.ns = ""
        m.id = 0
        m.action = Marker.DELETEALL
        return m

    def _mk_points_marker(self, frame_id, stamp, mid, ns, xs, ys, rgba):
        """
        포인트 집합을 POINTS 마커로 생성
        """
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = stamp
        m.ns = ns
        m.id = mid
        m.type = Marker.POINTS
        m.action = Marker.ADD

        m.pose.orientation.w = 1.0
        m.scale.x = self.point_scale
        m.scale.y = self.point_scale
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        m.lifetime = rospy.Duration(self.marker_lifetime)

        n = min(len(xs), len(ys))
        m.points = [Point(float(xs[i]), float(ys[i]), float(self.z_value)) for i in range(n)]
        return m

    def _mk_bbox_cube(self, frame_id, stamp, mid, ns, cx, cy, w, l, rgba):
        """
        중심/크기 기반 bbox CUBE 마커를 생성
        """
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = stamp
        m.ns = ns
        m.id = mid
        m.type = Marker.CUBE
        m.action = Marker.ADD

        m.pose.position.x = float(cx)
        m.pose.position.y = float(cy)
        m.pose.position.z = float(self.z_value)
        m.pose.orientation.w = 1.0

        m.scale.x = float(w)
        m.scale.y = float(l)
        m.scale.z = 0.05
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        m.lifetime = rospy.Duration(self.marker_lifetime)
        return m

    def _mk_bbox_outline(self, frame_id, stamp, mid, ns, xmin, xmax, ymin, ymax, rgba):
        """
        bbox 사각 외곽선을 LINE_STRIP 마커로 생성
        """
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = stamp
        m.ns = ns
        m.id = mid
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD

        m.pose.orientation.w = 1.0
        m.scale.x = self.bbox_line_width
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        m.lifetime = rospy.Duration(self.marker_lifetime)

        corners = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]
        m.points = [Point(float(x), float(y), float(self.z_value + 0.01)) for (x, y) in corners]
        return m

    def _mk_text(self, frame_id, stamp, mid, ns, x, y, text, rgba):
        """
        클러스터 정보 텍스트 마커를 생성
        """
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = stamp
        m.ns = ns
        m.id = mid
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD

        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = float(self.z_value + 0.6)
        m.pose.orientation.w = 1.0

        m.scale.z = float(self.text_size)
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        m.text = text
        m.lifetime = rospy.Duration(self.marker_lifetime)
        return m

    def _vehicle_like_filter(self, width, length, npts, v_rep, v_std):
        """
        폭/길이/속도 통계로 차량 유사 클러스터 여부를 판정
        """
        # v_rep: 대표 속도(현재는 trimmed mean 사용)
        if npts < self.cluster_min_points:
            return False
        if width < self.veh_min_w or width > self.veh_max_w:
            return False
        if length < self.veh_min_l or length > self.veh_max_l:
            return False
        if (length / max(width, 1e-3)) < self.min_aspect_yx:
            return False
        if abs(v_rep) < self.min_median_speed:
            return False
        if v_std > self.max_doppler_std:
            return False
        return True

    def _make_vertical_bbox(self, xmin, xmax, ymin, ymax):
        """
        패딩과 목표 종횡비를 반영해 세로형 bbox를 보정
        """
        xmin2 = xmin - self.pad_x
        xmax2 = xmax + self.pad_x
        ymin2 = ymin - self.pad_y
        ymax2 = ymax + self.pad_y

        w = xmax2 - xmin2
        l = ymax2 - ymin2

        target_l = max(l, self.target_aspect_yx * max(w, 1e-3))
        if target_l > l:
            cy = 0.5 * (ymin2 + ymax2)
            ymin2 = cy - 0.5 * target_l
            ymax2 = cy + 0.5 * target_l

        return xmin2, xmax2, ymin2, ymax2

    def cb(self, msg: PointCloud2):
        """
        포인트 전처리, 클러스터링, bbox 계산 후 MarkerArray를 발행
        """
        frame_id, stamp = self._header(msg)

        # 잔상 제거를 위해 DELETEALL 마커를 먼저 추가
        ma = MarkerArray()
        ma.markers.append(self._mk_deleteall(frame_id, stamp))

        pts = np.array(list(pc2.read_points(
            msg, field_names=("x", "y", "z", "power", "doppler"), skip_nans=True
        )), dtype=np.float32)

        if pts.size == 0:
            self.pub.publish(ma)
            return

        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        power = pts[:, 3]
        doppler = pts[:, 4]

        # ROI 필터 적용
        roi = (y >= self.min_forward_m) & (y <= self.max_forward_m)
        if self.min_power is not None:
            roi &= (power >= float(self.min_power))

        x = x[roi]; y = y[roi]; z = z[roi]; doppler = doppler[roi]
        roi_pts = int(x.size)

        if roi_pts == 0:
            self.pub.publish(ma)
            return

        # 이동점 필터 적용
        moving = (np.abs(doppler) >= self.min_speed_mps)
        xv = x[moving]; yv = y[moving]; zv = z[moving]; dv = doppler[moving]
        moving_pts = int(xv.size)

        if moving_pts < self.cluster_min_points:
            if self.debug:
                rospy.loginfo_throttle(
                    1.0,
                    f"[cluster_viz] roi_pts={roi_pts} moving_pts={moving_pts} clusters=0 candidates=0"
                )
            self.pub.publish(ma)
            return

        # 클러스터링 수행
        pts_xy = np.column_stack((xv, yv))
        clusters = cluster_points_xy(pts_xy, self.cluster_max_dist_m, self.cluster_min_points)
        n_clusters = int(len(clusters))

        if n_clusters == 0:
            if self.debug:
                rospy.loginfo_throttle(
                    1.0,
                    f"[cluster_viz] roi_pts={roi_pts} moving_pts={moving_pts} clusters=0 candidates=0"
                )
            self.pub.publish(ma)
            return

        # 후보 구성
        candidates = []
        all_clusters_info = []

        for c in clusters:
            cx = xv[c]; cy = yv[c]; cz = zv[c]; cd = dv[c]
            xmin = float(np.min(cx)); xmax = float(np.max(cx))
            ymin = float(np.min(cy)); ymax = float(np.max(cy))
            width = xmax - xmin
            length = ymax - ymin

            # 대표 속도는 상하 trim_ratio 제거 평균으로 계산
            v_rep = trimmed_mean(cd, trim_ratio=self.trim_ratio)
            v_std = float(np.std(cd))

            npts = int(len(c))
            r = float(np.median(np.sqrt(cx*cx + cy*cy)))

            info = (r, c, xmin, xmax, ymin, ymax, width, length, v_rep, v_std, npts)
            all_clusters_info.append(info)

            if (not self.use_vehicle_filter) or self._vehicle_like_filter(width, length, npts, v_rep, v_std):
                candidates.append(info)

        n_candidates = int(len(candidates))

        if self.debug:
            rospy.loginfo_throttle(
                1.0,
                f"[cluster_viz] roi_pts={roi_pts} moving_pts={moving_pts} clusters={n_clusters} candidates={n_candidates} "
                f"(use_vehicle_filter={self.use_vehicle_filter}, show_all={self.show_all_clusters}, trim_ratio={self.trim_ratio})"
            )

        chosen = candidates
        if len(chosen) == 0 and self.show_all_clusters:
            chosen = all_clusters_info

        if len(chosen) == 0:
            self.pub.publish(ma)
            return

        chosen.sort(key=lambda t: t[0])
        if self.only_nearest:
            chosen = chosen[:1]
        else:
            chosen = chosen[:min(self.max_boxes, len(chosen))]

        # 선택 클러스터 포인트(접근/이탈)
        pts_app_x, pts_app_y = [], []
        pts_rec_x, pts_rec_y = [], []

        bbox_id = 10
        outline_id = 200
        text_id = 400

        for (r, c, xmin, xmax, ymin, ymax, width, length, v_rep, v_std, npts) in chosen:
            cx = xv[c]; cy = yv[c]; cd = dv[c]

            app = (cd < 0.0)
            rec = (cd > 0.0)
            pts_app_x.extend(list(cx[app])); pts_app_y.extend(list(cy[app]))
            pts_rec_x.extend(list(cx[rec])); pts_rec_y.extend(list(cy[rec]))

            xmin2, xmax2, ymin2, ymax2 = self._make_vertical_bbox(xmin, xmax, ymin, ymax)
            w2 = xmax2 - xmin2
            l2 = ymax2 - ymin2
            xmid = 0.5 * (xmin2 + xmax2)
            ymid = 0.5 * (ymin2 + ymax2)

            # 녹색 bbox 마커 추가
            ma.markers.append(self._mk_bbox_cube(
                frame_id, stamp, bbox_id, "bbox_fill",
                xmid, ymid, w2, l2,
                (0.1, 1.0, 0.1, float(self.bbox_alpha))
            ))
            bbox_id += 1

            ma.markers.append(self._mk_bbox_outline(
                frame_id, stamp, outline_id, "bbox_outline",
                xmin2, xmax2, ymin2, ymax2,
                (0.1, 1.0, 0.1, 1.0)
            ))
            outline_id += 1

            if self.draw_text:
                # 텍스트에는 대표 속도 v_rep를 사용
                txt = f"r={r:.1f}m v={v_rep*3.6:.1f}km/h n={npts}"
                ma.markers.append(self._mk_text(
                    frame_id, stamp, text_id, "bbox_text",
                    xmid, ymid, txt,
                    (1.0, 1.0, 1.0, 1.0)
                ))
                text_id += 1

        # 포인트 마커 추가
        if len(pts_app_x) > 0:
            ma.markers.append(self._mk_points_marker(
                frame_id, stamp, 1, "approaching_pts",
                np.array(pts_app_x), np.array(pts_app_y),
                (1.0, 0.0, 0.0, 1.0)
            ))
        if len(pts_rec_x) > 0:
            ma.markers.append(self._mk_points_marker(
                frame_id, stamp, 2, "receding_pts",
                np.array(pts_rec_x), np.array(pts_rec_y),
                (0.0, 0.4, 1.0, 1.0)
            ))

        self.pub.publish(ma)


if __name__ == "__main__":
    rospy.init_node("radar_cluster_bbox_viz")
    node = RadarClusterBBoxViz()
    rospy.loginfo("radar_cluster_bbox_viz started. Publishing DELETEALL + cluster bboxes/points on /radar/cluster_viz")
    rospy.spin()
