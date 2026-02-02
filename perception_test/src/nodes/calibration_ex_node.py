#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
calibration_ex_node.py

레이더-카메라 외부 파라미터(R, t)를 실시간으로 추정하는 노드입니다.
DetectionArray(2D bbox)와 PointCloud2(3D 점) 데이터를 시간 동기화하여
각 프레임에서 다중 차량을 고려해 레이더 클러스터와 카메라 검출을 매칭하고,
대응점으로 solvePnP(RANSAC) 수행합니다. 초기값은 기본 R,t 또는 파라미터로
주어진 값을 사용해 시작합니다.

ROS 파라미터:
- ~detections_topic (기본: /perception_test/detections)
- ~radar_topic      (기본: /point_cloud)
- ~camera_info_topic (기본: /camera/camera_info)
- ~min_samples      (기본: 20)
- ~min_cluster_points (기본: 6)
- ~cluster_eps      (기본: 1.2m)
- ~max_assoc_angle_deg (기본: 10deg)
- ~min_pairs_per_frame (기본: 2)
- ~detection_threshold (기본: 0.5)
- ~min_range_m      (기본: 5.0)
- ~min_power        (기본: 0.0)
- ~sync_queue_size  (기본: 10)
- ~sync_slop        (기본: 0.1)
- ~radar_fields     (기본: [x, y, z, power])
- ~initial_R        (기본: set 값)
- ~initial_t        (기본: set 값)
- ~extrinsic_path    (기본: $(find perception_test)/config/extrinsic.json)
"""

import json
import os
import re
from typing import List

import cv2
import numpy as np
import rospy
import rospkg

from sensor_msgs.msg import CameraInfo, PointCloud2
import sensor_msgs.point_cloud2 as pc2
import message_filters
from perception_test.msg import DetectionArray

_FIND_RE = re.compile(r"\$\(\s*find\s+([A-Za-z0-9_]+)\s*\)")


def resolve_ros_path(path: str) -> str:
    if not isinstance(path, str):
        return path
    if "$(" not in path:
        return path
    rp = rospkg.RosPack()

    def _sub(match):
        pkg = match.group(1)
        try:
            return rp.get_path(pkg)
        except Exception:
            return match.group(0)

    return _FIND_RE.sub(_sub, path)


class ExtrinsicCalibrationManager:
    def __init__(self):
        rospy.init_node('calibration_ex_node', anonymous=False)
        self.detections_topic = rospy.get_param('~detections_topic', '/perception_test/detections')
        self.radar_topic = rospy.get_param('~radar_topic', '/point_cloud')
        self.camera_info_topic = rospy.get_param('~camera_info_topic', '/camera/camera_info')
        self.min_samples = int(rospy.get_param('~min_samples', 20))
        self.min_cluster_points = int(rospy.get_param('~min_cluster_points', 6))
        self.cluster_eps = float(rospy.get_param('~cluster_eps', 1.2))
        self.max_assoc_angle_deg = float(rospy.get_param('~max_assoc_angle_deg', 10.0))
        self.min_pairs_per_frame = int(rospy.get_param('~min_pairs_per_frame', 2))
        self.detection_thresh = float(rospy.get_param('~detection_threshold', 0.5))
        self.min_range_m = float(rospy.get_param('~min_range_m', 5.0))
        self.min_power = float(rospy.get_param('~min_power', 0.0))
        self.sync_queue = int(rospy.get_param('~sync_queue_size', 10))
        self.sync_slop = float(rospy.get_param('~sync_slop', 0.1))
        self.initial_R = np.array(rospy.get_param('~initial_R', [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]), dtype=np.float64)
        self.initial_t = np.array(
            rospy.get_param('~initial_t', [0.08, 0.0, -0.08]),
            dtype=np.float64
        ).reshape(3, 1)
        self.initial_from_extrinsic = bool(rospy.get_param('~initial_from_extrinsic', True))

        # extrinsic 저장 경로: config/extrinsic.json
        default_pkg = 'perception_test'
        rp = rospkg.RosPack()
        default_path = os.path.join(rp.get_path(default_pkg), 'config', 'extrinsic.json')
        self.save_path = resolve_ros_path(rospy.get_param('~extrinsic_path', default_path))

        self.radar_fields = rospy.get_param('~radar_fields', ['x', 'y', 'z', 'power'])

        self.samples = 0
        self.obj_pts: List[np.ndarray] = []
        self.img_pts: List[np.ndarray] = []
        self.K = None
        self.D = None
        if self.initial_from_extrinsic:
            self._load_initial_extrinsic()

        # 동기화 설정
        sub_det = message_filters.Subscriber(self.detections_topic, DetectionArray)
        sub_rad = message_filters.Subscriber(self.radar_topic, PointCloud2)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [sub_det, sub_rad],
            self.sync_queue,
            self.sync_slop,
        )
        self.ts.registerCallback(self._callback)
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self._info_callback, queue_size=1)
        rospy.loginfo(
            f"[Extrinsic] 데이터 수집 중... det={self.detections_topic}, radar={self.radar_topic}"
        )

    def _load_initial_extrinsic(self):
        if not os.path.exists(self.save_path):
            rospy.loginfo(f"[Extrinsic] 초기값: extrinsic.json 없음 ({self.save_path})")
            return
        try:
            with open(self.save_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            R = np.array(data.get('R', []), dtype=np.float64)
            t = np.array(data.get('t', []), dtype=np.float64).reshape(3, 1)
            if R.shape == (3, 3) and t.shape == (3, 1):
                self.initial_R = R
                self.initial_t = t
                rospy.loginfo(f"[Extrinsic] 초기값: extrinsic.json 로드 ({self.save_path})")
            else:
                rospy.logwarn("[Extrinsic] extrinsic.json 형식 오류, 파라미터 초기값 사용")
        except Exception as exc:
            rospy.logwarn(f"[Extrinsic] extrinsic.json 로드 실패: {exc}")

        rospy.loginfo(f"[Extrinsic] 데이터 수집 중... det={self.detections_topic}, radar={self.radar_topic}")

    def _info_callback(self, msg: CameraInfo):
        if self.K is None:
            self.K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
            self.D = np.array(msg.D, dtype=np.float64) if msg.D else None
            rospy.loginfo("[Extrinsic] CameraInfo 수신")

    def _callback(self, det_msg: DetectionArray, rad_msg: PointCloud2):
        # CameraInfo 수신 전이면 리턴
        if self.K is None:
            return
        # detection 없으면 패스
        if not det_msg.detections:
            return
        
        det_centers = []
        for det in det_msg.detections:
            if det.score < self.detection_thresh:
                continue
            bb = det.bbox
            u = (bb.xmin + bb.xmax) * 0.5
            v = (bb.ymin + bb.ymax) * 0.5
            det_centers.append((u, v))

        if not det_centers:
            return
        
        best_det = max(det_msg.detections, key=lambda d: d.score)
        if best_det.score < self.detection_thresh:
            return
        # 2D bbox 중심 (u,v)
        bb = best_det.bbox
        u = (bb.xmin + bb.xmax) * 0.5
        v = (bb.ymin + bb.ymax) * 0.5
        # 3D 레이더 포인트 추출
        pts = []
        field_names = [f.name for f in rad_msg.fields]
        use_fields = [f for f in self.radar_fields if f in field_names]
        if len(use_fields) < 3:
            rospy.logwarn_throttle(5.0, "[Extrinsic] Radar point fields missing: %s", self.radar_fields)
            return
        for p in pc2.read_points(rad_msg, field_names=tuple(use_fields), skip_nans=True):
            pts.append([float(v) for v in p])
        if not pts:
            return
        pts = np.array(pts, dtype=np.float64)
        xyz = pts[:, :3]
        if pts.shape[1] > 3:
            power = pts[:, 3]
            valid = power > self.min_power
            xyz = xyz[valid, :]
        if xyz.shape[0] < self.min_cluster_points:
            return

        clusters = self._cluster_points(xyz)
        if not clusters:
            return

        centroids = []
        for cluster in clusters:
            if len(cluster) < self.min_cluster_points:
                continue
            centroid = np.mean(cluster, axis=0)
            dist = np.linalg.norm(centroid)
            if dist < self.min_range_m:
                continue
            centroids.append(centroid)

        if not centroids:
            return

        pairs = self._associate(det_centers, centroids)
        if len(pairs) < self.min_pairs_per_frame:
            return

        for img_pt, obj_pt in pairs:
            self.obj_pts.append(obj_pt)
            self.img_pts.append(img_pt)
            self.samples += 1

        rospy.loginfo(
            f"[Extrinsic] 샘플 {self.samples}/{self.min_samples} 저장 "
            f"(pairs={len(pairs)} clusters={len(centroids)})"
        )
        if self.samples >= self.min_samples:
            self.run_calibration()

    def run_calibration(self):
        obj_pts = np.array(self.obj_pts, dtype=np.float64)
        img_pts = np.array(self.img_pts, dtype=np.float64)
        rvec_init, _ = cv2.Rodrigues(self.initial_R)
        tvec_init = self.initial_t.copy()
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts,
            img_pts,
            self.K,
            self.D,
            rvec=rvec_init,
            tvec=tvec_init,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=8.0,
            iterationsCount=200,
        )
        if not success:
            success, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, self.K, self.D,
                rvec=rvec_init, tvec=tvec_init,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        if not success:
            rospy.logwarn("[Extrinsic] solvePnP 실패, 샘플 더 모음")
            self.min_samples += 10
            return
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3, 1)
        if inliers is not None:
            rospy.loginfo(f"[Extrinsic] inliers: {len(inliers)}")
        data = {'R': R.tolist(), 't': t.flatten().tolist()}
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        rospy.loginfo(f"[Extrinsic] extrinsic.json 저장 완료: {self.save_path}")
        rospy.signal_shutdown("외부 캘리브레이션 완료")

    def _cluster_points(self, points: np.ndarray) -> List[np.ndarray]:
        clusters = []
        visited = np.zeros(len(points), dtype=bool)
        for idx in range(len(points)):
            if visited[idx]:
                continue
            queue = [idx]
            visited[idx] = True
            cluster = []
            while queue:
                i = queue.pop()
                cluster.append(points[i])
                dists = np.linalg.norm(points - points[i], axis=1)
                neighbors = np.where((dists <= self.cluster_eps) & (~visited))[0]
                for n in neighbors:
                    visited[n] = True
                    queue.append(n)
            clusters.append(cluster)
        return clusters

    def _associate(self, det_centers, centroids):
        pairs = []
        if self.K is None:
            return pairs

        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]
        max_angle = np.deg2rad(self.max_assoc_angle_deg)

        used = set()
        for (u, v) in det_centers:
            x = (u - cx) / fx
            y = (v - cy) / fy
            ray = np.array([x, y, 1.0], dtype=np.float64)
            ray = ray / np.linalg.norm(ray)
            cam_az = np.arctan2(ray[0], ray[2])
            cam_el = np.arctan2(-ray[1], ray[2])

            best_idx = None
            best_score = None
            for i, centroid in enumerate(centroids):
                if i in used:
                    continue
                c = centroid / np.linalg.norm(centroid)
                rad_az = np.arctan2(c[1], c[0])
                rad_el = np.arctan2(c[2], np.sqrt(c[0] ** 2 + c[1] ** 2))
                d_az = np.abs(cam_az - rad_az)
                d_el = np.abs(cam_el - rad_el)
                score = np.hypot(d_az, d_el)
                if score <= max_angle and (best_score is None or score < best_score):
                    best_score = score
                    best_idx = i

            if best_idx is not None:
                used.add(best_idx)
                pairs.append(([u, v], centroids[best_idx]))
        return pairs
def main():
    ExtrinsicCalibrationManager()
    rospy.spin()

if __name__ == '__main__':
    main()