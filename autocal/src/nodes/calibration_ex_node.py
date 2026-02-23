#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
역할:
- 검출/트랙 bbox와 레이더 포인트를 동기 수신해 카메라-레이더 외부파라미터(R, t)를 자동 보정한다.
- 레이더 클러스터 중심과 bbox 기준점(상단/중앙/하단)의 대응쌍을 수집한 뒤, RANSAC+SVD로 회전 R을 추정하고 solvePnP로 평행이동 t를 미세조정한다.
- 보정 결과를 extrinsic.json으로 저장하고 진단 노드 시작 토픽을 발행한다.

입력:
- `~detections_topic` (`autocal/DetectionArray`): 트랙 ID와 bbox 목록.
- `~radar_topic` (`sensor_msgs/PointCloud2`): x, y, z, doppler를 포함한 레이더 포인트.
- `~camera_info_topic` (`sensor_msgs/CameraInfo`): K, D 내부파라미터.

출력:
- `~extrinsic_path` (`json`): 보정된 `R`, `t` 저장.
- `/autocal/diagnosis/start` (`std_msgs/String`): 진단 노드 시작 트리거.
"""

import json
import os
import re
from collections import deque

import cv2
import message_filters
import numpy as np
import rospy
import rospkg
import sensor_msgs.point_cloud2 as pc2
from autocal.msg import DetectionArray
from sensor_msgs.msg import CameraInfo, PointCloud2
from std_msgs.msg import String

_FIND_RE = re.compile(r"\$\(\s*find\s+([A-Za-z0-9_]+)\s*\)") 

# ---------------------------------------------------------
# 보정 파라미터
# ---------------------------------------------------------
MIN_SAMPLES = 1000                  # 최적화 시작 최소 대응점 수
REQUIRED_DURATION_SEC = 30.0        # 데이터 수집 최소 시간(초)
MIN_TRACK_LEN = 20                  # 유효 트랙 최소 길이(프레임)
MIN_SPEED_MPS = 4.0                 # 정적 물체 제거용 최소 속도(m/s)
CLUSTER_TOL_M = 5.0                 # 레이더 XY 클러스터 반경(m)
MATCH_MAX_DIST_PX = 1000.0          # 영상 매칭 허용 거리(px)
RANSAC_ITERATIONS = 1000            # RANSAC 반복 횟수
RANSAC_THRESHOLD_DEG = 0.3          # 인라이어 각도 임계값(도)


def resolve_ros_path(path: str) -> str:
    """
    `$(find pkg)` 형식 경로를 실제 절대 경로로 변환
    """
    if not isinstance(path, str):
        return path
    if "$(" not in path:
        return path

    rp = rospkg.RosPack()

    def _sub(match):
        """
        패키지명을 rospack 경로로 치환
        """
        pkg = match.group(1)
        try:
            return rp.get_path(pkg)
        except Exception:
            return match.group(0)

    return _FIND_RE.sub(_sub, path)


class ExtrinsicCalibrationManager:
    def __init__(self):
        """
        보정 노드의 파라미터, 동기 구독, 내부 상태를 초기화
        """
        rospy.init_node("calibration_ex_node", anonymous=False)

        self.detections_topic = rospy.get_param("~detections_topic", "/autocal/tracks")
        self.radar_topic = rospy.get_param("~radar_topic", "/point_cloud")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/camera/camera_info")
        self.bbox_ref_mode = rospy.get_param("~bbox_ref_mode", "center")

        self.min_samples = int(rospy.get_param("~min_samples", MIN_SAMPLES))
        self.req_duration = float(rospy.get_param("~req_duration", REQUIRED_DURATION_SEC))
        self.min_track_len = int(rospy.get_param("~min_track_len", MIN_TRACK_LEN))
        self.min_speed_mps = float(rospy.get_param("~min_speed_mps", MIN_SPEED_MPS))
        self.cluster_tol = float(rospy.get_param("~cluster_tol", CLUSTER_TOL_M))
        self.match_max_dist_px = float(rospy.get_param("~match_max_dist_px", MATCH_MAX_DIST_PX))
        self.ransac_iterations = int(rospy.get_param("~ransac_iterations", RANSAC_ITERATIONS))
        self.ransac_threshold_deg = float(rospy.get_param("~ransac_threshold_deg", RANSAC_THRESHOLD_DEG))

        rp = rospkg.RosPack()
        default_path = os.path.join(rp.get_path("autocal"), "config", "extrinsic.json")
        self.save_path = resolve_ros_path(rospy.get_param("~extrinsic_path", default_path))

        self.track_buffer = {}
        self.K = None
        self.D = None
        self.inv_K = None
        self.curr_R = np.eye(3)
        self.curr_t = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)

        self.collection_start_time = None
        self.is_calibrating = False
        self._load_extrinsic()

        sub_det = message_filters.Subscriber(self.detections_topic, DetectionArray)
        sub_rad = message_filters.Subscriber(self.radar_topic, PointCloud2)
        self.ts = message_filters.ApproximateTimeSynchronizer([sub_det, sub_rad], 50, 0.1)
        self.ts.registerCallback(self._callback)

        rospy.Subscriber(self.camera_info_topic, CameraInfo, self._info_callback)
        self.pub_diag_start = rospy.Publisher("/autocal/diagnosis/start", String, queue_size=1)
        rospy.loginfo("[Autocal] Extrinsic calibration node started")

    def _load_extrinsic(self):
        """
        기존 extrinsic 파일을 읽어 초기값으로 반영
        """
        if not os.path.exists(self.save_path):
            return
        try:
            with open(self.save_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.curr_R = np.array(data["R"], dtype=np.float64)
            self.curr_t = np.array(data["t"], dtype=np.float64).reshape(3, 1)
        except Exception:
            pass

    def _info_callback(self, msg: CameraInfo):
        """
        CameraInfo에서 K, D, K 역행렬 초기화
        """
        if self.K is None:
            self.K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
            self.D = np.array(msg.D, dtype=np.float64)
            self.inv_K = np.linalg.inv(self.K)

    def _cluster_radar_points(self, points: np.ndarray, velocities: np.ndarray):
        """
        XY 거리 기반 레이더 클러스터링 후 최소 속도 조건 통과 클러스터만 반환
        """
        n = points.shape[0]
        if n == 0:
            return []

        visited = np.zeros(n, dtype=bool)
        clusters = []
        for i in range(n):
            if visited[i]:
                continue

            dist_sq = np.sum((points[:, :2] - points[i, :2]) ** 2, axis=1)
            neighbors = np.where((dist_sq <= self.cluster_tol ** 2) & (~visited))[0]
            if len(neighbors) < 2:
                continue

            visited[neighbors] = True
            vel_med = float(np.median(velocities[neighbors]))
            if abs(vel_med) >= self.min_speed_mps:
                clusters.append((np.median(points[neighbors], axis=0), vel_med))

        return clusters

    def _callback(self, det_msg: DetectionArray, rad_msg: PointCloud2):
        """
        동기 수신된 bbox와 레이더 데이터에서 대응쌍을 누적하고 조건 충족 시 최적화 실행
        """
        if self.K is None or self.is_calibrating:
            return

        if self.collection_start_time is None:
            self.collection_start_time = rospy.Time.now()

        radar_raw = np.array(list(pc2.read_points(rad_msg, field_names=("x", "y", "z", "doppler"), skip_nans=True)))
        if radar_raw.shape[0] == 0:
            return

        radar_objects = self._cluster_radar_points(radar_raw[:, :3], radar_raw[:, 3])

        proj_pts = []
        obj_3d_pts = []

        t_mat = np.eye(4)
        t_mat[:3, :3] = self.curr_R
        t_mat[:3, 3] = self.curr_t.flatten()

        for cent_3d, _ in radar_objects:
            pt_cam = t_mat @ np.append(cent_3d, 1.0)
            if pt_cam[2] <= 1.5:
                continue
            uv = self.K @ pt_cam[:3]
            proj_pts.append([uv[0] / uv[2], uv[1] / uv[2]])
            obj_3d_pts.append(cent_3d)

        if not proj_pts:
            return
        proj_pts = np.array(proj_pts)

        for det in det_msg.detections:
            if det.id <= 0:
                continue

            u_target, v_target = self._get_bbox_ref_point(det.bbox)
            dists = np.hypot(proj_pts[:, 0] - u_target, proj_pts[:, 1] - v_target)
            idx = int(np.argmin(dists))

            if dists[idx] >= self.match_max_dist_px:
                continue

            if det.id not in self.track_buffer:
                self.track_buffer[det.id] = deque(maxlen=60)

            nc = self.inv_K @ np.array([u_target, v_target, 1.0])
            nc /= np.linalg.norm(nc)
            self.track_buffer[det.id].append((nc, obj_3d_pts[idx], (u_target, v_target)))

        total = sum(len(d) for d in self.track_buffer.values() if len(d) >= self.min_track_len)
        elapsed = (rospy.Time.now() - self.collection_start_time).to_sec()
        rospy.loginfo_throttle(2.0, f"[Autocal] Collection progress: {total}/{self.min_samples} | {elapsed:.1f}s")

        if elapsed >= self.req_duration and total >= self.min_samples:
            self.run_robust_optimization()

    def run_robust_optimization(self):
        """
        RANSAC-SVD로 회전 R 추정 후 solvePnP로 평행이동 t 보정
        """
        self.is_calibrating = True
        try:
            rospy.loginfo("[Autocal] Start extrinsic optimization")

            unit_pairs = []
            for _, data in self.track_buffer.items():
                if len(data) < self.min_track_len:
                    continue
                for nc, rv_3d, uv_2d in data:
                    unit_pairs.append((rv_3d / np.linalg.norm(rv_3d), nc, uv_2d, rv_3d))

            if len(unit_pairs) < 100:
                rospy.logwarn("[Autocal] Insufficient valid pairs, continue collecting")
                return

            best_R = self.curr_R
            max_inliers = -1
            final_inlier_mask = None

            for _ in range(self.ransac_iterations):
                idx = np.random.choice(len(unit_pairs), 5, replace=False)
                a_sub = np.array([unit_pairs[i][0] for i in idx]).T
                b_sub = np.array([unit_pairs[i][1] for i in idx]).T

                h = a_sub @ b_sub.T
                u, _, vt = np.linalg.svd(h)
                r_cand = vt.T @ u.T
                if np.linalg.det(r_cand) < 0:
                    vt[2, :] *= -1
                    r_cand = vt.T @ u.T

                inlier_mask = []
                for rv_u, nc_u, _, _ in unit_pairs:
                    err = np.rad2deg(np.arccos(np.clip(np.dot(r_cand @ rv_u, nc_u), -1.0, 1.0)))
                    inlier_mask.append(err < self.ransac_threshold_deg)

                num_in = sum(inlier_mask)
                if num_in > max_inliers:
                    max_inliers = num_in
                    best_R = r_cand
                    final_inlier_mask = inlier_mask

            inlier_indices = np.where(final_inlier_mask)[0]
            v_obj = np.array([unit_pairs[i][3] for i in inlier_indices], dtype=np.float32)
            v_img = np.array([unit_pairs[i][2] for i in inlier_indices], dtype=np.float32)

            r_fix, _ = cv2.Rodrigues(best_R)
            t_ref = self.curr_t.copy().astype(np.float32)

            success, _, t_final = cv2.solvePnP(
                v_obj,
                v_img,
                self.K,
                self.D,
                r_fix,
                t_ref,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            if success and np.linalg.norm(t_final - self.curr_t) < 2.0:
                self.curr_t = t_final.reshape(3, 1)

            self.curr_R = best_R
            self._save_result(self.curr_R, self.curr_t)
            self.pub_diag_start.publish(String(data=str(self.bbox_ref_mode)))
            rospy.loginfo(f"[Autocal] Optimization done: inlier {max_inliers}/{len(unit_pairs)}")
            rospy.signal_shutdown("extrinsic calibration complete")

        except Exception as exc:
            rospy.logerr(f"[Autocal] Optimization error: {exc}")
        finally:
            self.is_calibrating = False
            self.collection_start_time = None

    def _save_result(self, r_mat: np.ndarray, t_vec: np.ndarray):
        """
        보정된 R, t를 JSON 파일로 저장
        """
        data = {"R": r_mat.tolist(), "t": t_vec.flatten().tolist()}
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def _get_bbox_ref_point(self, bbox):
        """
        모드 값에 따라 bbox 기준점(상단/중앙/하단) 선택
        """
        x1, y1, x2, y2 = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
        mode = str(self.bbox_ref_mode).lower()
        if mode in {"bottom", "bottom_center", "bottom-center"}:
            return (x1 + x2) / 2.0, float(y2)
        if mode in {"top", "top_center", "top-center"}:
            return (x1 + x2) / 2.0, float(y1)
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


if __name__ == "__main__":
    try:
        ExtrinsicCalibrationManager()
    except Exception as exc:
        rospy.logerr(f"Manager Crash: {exc}")
    rospy.spin()
