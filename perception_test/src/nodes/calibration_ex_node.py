#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
from collections import deque
import numpy as np
import cv2
import rospy
import rospkg
import message_filters
from std_msgs.msg import String
from sensor_msgs.msg import CameraInfo, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from perception_test.msg import DetectionArray

_FIND_RE = re.compile(r"\$\(\s*find\s+([A-Za-z0-9_]+)\s*\)")

def resolve_ros_path(path: str) -> str:
    if not isinstance(path, str): return path
    if "$(" not in path: return path
    rp = rospkg.RosPack()
    def _sub(match):
        pkg = match.group(1)
        try: return rp.get_path(pkg)
        except Exception: return match.group(0)
    return _FIND_RE.sub(_sub, path)

class ExtrinsicCalibrationManager:
    def __init__(self):
        rospy.init_node('calibration_ex_node', anonymous=False)

        # 1. 토픽 파라미터 로드
        self.detections_topic = rospy.get_param('~detections_topic', '/perception_test/tracks')
        self.radar_topic = rospy.get_param('~radar_topic', '/point_cloud')
        self.camera_info_topic = rospy.get_param('~camera_info_topic', '/camera/camera_info')
        self.bbox_ref_mode = rospy.get_param('~bbox_ref_mode', 'center')

        # -------------------------------------------------------------
        # [설정] 논문 및 실무 기반 고정밀 파라미터
        # -------------------------------------------------------------
        self.min_samples = 1000         # 800개 이상의 정교한 포인트 확보
        self.req_duration = 30.0        # 20초 이상 주행 데이터 수집
        self.min_track_len = 15         # 최소 15프레임 이상 추적된 궤적만 사용

        # [핵심] 정지 차량 배제
        self.min_speed_mps = 2

        self.cluster_tol = 2.0          # 레이더 클러스터링 거리 임계값
        self.match_max_dist_px = 800.0  # 초기 매칭 허용 범위 (전역 매칭)

        # RANSAC 설정
        self.ransac_iterations = 500
        self.ransac_threshold_deg = 3.0 # 2도 이내 오차만 인라이어 판정

        # 파일 경로 설정
        rp = rospkg.RosPack()
        default_path = os.path.join(rp.get_path('perception_test'), 'config', 'extrinsic.json')
        self.save_path = resolve_ros_path(rospy.get_param('~extrinsic_path', default_path))

        self.track_buffer = {}
        self.K = None
        self.D = None
        self.inv_K = None

        # 초기값 로드
        self.curr_R = np.eye(3)
        self.curr_t = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)
        self._load_extrinsic()

        self.collection_start_time = None
        self.is_calibrating = False

        # 구독 및 타임 싱크 설정
        sub_det = message_filters.Subscriber(self.detections_topic, DetectionArray)
        sub_rad = message_filters.Subscriber(self.radar_topic, PointCloud2)
        self.ts = message_filters.ApproximateTimeSynchronizer([sub_det, sub_rad], 50, 0.1)
        self.ts.registerCallback(self._callback)
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self._info_callback)
        self.pub_diag_start = rospy.Publisher('/perception_test/diagnosis/start', String, queue_size=1)

        rospy.loginfo("[Autocal] Robust Center-Matching Integration Started.")

    def _load_extrinsic(self):
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'r') as f:
                    data = json.load(f)
                self.curr_R = np.array(data['R'], dtype=np.float64)
                self.curr_t = np.array(data['t'], dtype=np.float64).reshape(3, 1)
            except: pass

    def _info_callback(self, msg):
        if self.K is None:
            self.K = np.array(msg.K).reshape(3, 3)
            self.D = np.array(msg.D)
            self.inv_K = np.linalg.inv(self.K)

    def _cluster_radar_points(self, points, velocities):
        n = points.shape[0]
        if n == 0: return []
        visited = np.zeros(n, dtype=bool)
        clusters = []
        for i in range(n):
            if visited[i]: continue
            dist_sq = np.sum((points[:, :2] - points[i, :2])**2, axis=1)
            neighbors = np.where((dist_sq <= self.cluster_tol**2) & (~visited))[0]
            if len(neighbors) >= 2:
                visited[neighbors] = True
                vel_med = np.median(velocities[neighbors])
                # 동적 객체 필터링: 주행 중인 차량만 최적화에 사용
                if abs(vel_med) >= self.min_speed_mps:
                    clusters.append((np.median(points[neighbors], axis=0), vel_med))
        return clusters

    def _callback(self, det_msg, rad_msg):
        # 최적화 수행 중에는 새로운 데이터 수집 중단 (무한 로딩 방지 1단계)
        if self.K is None or self.is_calibrating: return
        if self.collection_start_time is None: self.collection_start_time = rospy.Time.now()

        radar_raw = np.array(list(pc2.read_points(rad_msg, field_names=("x", "y", "z", "doppler"), skip_nans=True)))
        if radar_raw.shape[0] == 0: return
        radar_objects = self._cluster_radar_points(radar_raw[:, :3], radar_raw[:, 3])

        # 대표점 투영 (현재 R, t 기준)
        proj_pts = []
        obj_3d_pts = []
        T_mat = np.eye(4)
        T_mat[:3, :3] = self.curr_R
        T_mat[:3, 3] = self.curr_t.flatten()

        for (cent_3d, vel) in radar_objects:
            pt_cam = T_mat @ np.append(cent_3d, 1.0)
            if pt_cam[2] > 1.5:
                uv = self.K @ pt_cam[:3]
                proj_pts.append([uv[0]/uv[2], uv[1]/uv[2]])
                obj_3d_pts.append(cent_3d)

        if not proj_pts: return
        proj_pts = np.array(proj_pts)

        # -------------------------------------------------------------
        # [매칭 전략] 박스 정중앙(Center) 기준 매칭
        # 앞차 가림 현상 등으로 바닥이 안 보이는 경우를 위한 대응
        # -------------------------------------------------------------
        for det in det_msg.detections:
            if det.id <= 0: continue

            u_target, v_target = self._get_bbox_ref_point(det.bbox)

            dists = np.hypot(proj_pts[:, 0] - u_target, proj_pts[:, 1] - v_target)
            idx = np.argmin(dists)

            if dists[idx] < self.match_max_dist_px:
                if det.id not in self.track_buffer:
                    self.track_buffer[det.id] = deque(maxlen=60)

                # Camera Ray 생성 (박스 중앙 기준)
                nc = self.inv_K @ np.array([u_target, v_target, 1.0])
                nc /= np.linalg.norm(nc)
                self.track_buffer[det.id].append((nc, obj_3d_pts[idx], (u_target, v_target)))

        # 수집 진행도 체크
        total = sum(len(d) for d in self.track_buffer.values() if len(d) >= self.min_track_len)
        elapsed = (rospy.Time.now() - self.collection_start_time).to_sec()
        rospy.loginfo_throttle(2.0, f"[Autocal] Progress: {total}/{self.min_samples} pts | {elapsed:.1f}s")

        if elapsed >= self.req_duration and total >= self.min_samples:
            self.run_robust_optimization()

    def run_robust_optimization(self):
        """ RANSAC-SVD 2단계 최적화 """
        self.is_calibrating = True
        try:
            rospy.loginfo("[Autocal] Optimizing R and Refining t...")

            # 유효 데이터 페어 구성
            unit_pairs = []
            for tid, data in self.track_buffer.items():
                if len(data) < self.min_track_len: continue
                for nc, rv_3d, uv_2d in data:
                    unit_pairs.append((rv_3d / np.linalg.norm(rv_3d), nc, uv_2d, rv_3d))

            if len(unit_pairs) < 100:
                rospy.logwarn("[Autocal] Insufficient track data. Continuing collection.")
                return

            # --- [Step 1] RANSAC for Rotation (R) ---
            best_R = self.curr_R
            max_inliers = -1
            final_inlier_mask = None

            for _ in range(self.ransac_iterations):
                idx = np.random.choice(len(unit_pairs), 5, replace=False)
                A_sub = np.array([unit_pairs[i][0] for i in idx]).T
                B_sub = np.array([unit_pairs[i][1] for i in idx]).T

                # SVD를 통한 회전 행렬 도출 (Kabsch)
                H = A_sub @ B_sub.T
                U, S, Vt = np.linalg.svd(H)
                R_cand = Vt.T @ U.T
                if np.linalg.det(R_cand) < 0:
                    Vt[2, :] *= -1
                    R_cand = Vt.T @ U.T

                # 인라이어 판별
                inlier_mask = []
                for rv_u, nc_u, _, _ in unit_pairs:
                    err = np.rad2deg(np.arccos(np.clip(np.dot(R_cand @ rv_u, nc_u), -1.0, 1.0)))
                    inlier_mask.append(err < self.ransac_threshold_deg)

                num_in = sum(inlier_mask)
                if num_in > max_inliers:
                    max_inliers = num_in
                    best_R = R_cand
                    final_inlier_mask = inlier_mask

            # --- [Step 2] Translation (t) Refinement ---
            inlier_indices = np.where(final_inlier_mask)[0]
            v_obj = np.array([unit_pairs[i][3] for i in inlier_indices], dtype=np.float32)
            v_img = np.array([unit_pairs[i][2] for i in inlier_indices], dtype=np.float32)

            r_fix, _ = cv2.Rodrigues(best_R)
            t_ref = self.curr_t.copy().astype(np.float32)

            # R 고정 상태에서 t 미세 조정
            success, _, t_final = cv2.solvePnP(
                v_obj, v_img, self.K, self.D,
                r_fix, t_ref,
                useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success and np.linalg.norm(t_final - self.curr_t) < 2.0:
                self.curr_t = t_final.reshape(3, 1)

            # 결과 확정
            self.curr_R = best_R
            self._save_result(self.curr_R, self.curr_t)
            self.pub_diag_start.publish(String(data=str(self.bbox_ref_mode)))
            rospy.loginfo(f"[Autocal] SUCCESS. Final Inliers: {max_inliers}/{len(unit_pairs)}")
            rospy.signal_shutdown("Calibration Process Done")

        except Exception as e:
            rospy.logerr(f"[Autocal] Optimization Error: {e}")
        finally:
            # 무한 로딩 방지 2단계: 에러가 나더라도 플래그를 해제하여 시스템 복구
            self.is_calibrating = False
            self.collection_start_time = None

    def _save_result(self, R, t):
        data = {"R": R.tolist(), "t": t.flatten().tolist()}
        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=4)

    def _get_bbox_ref_point(self, bbox):
        x1, y1, x2, y2 = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
        mode = str(self.bbox_ref_mode).lower()
        if mode in {'bottom', 'bottom_center', 'bottom-center'}:
            return (x1 + x2) / 2.0, float(y2)
        if mode in {'top', 'top_center', 'top-center'}:
            return (x1 + x2) / 2.0, float(y1)
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

if __name__ == '__main__':
    try: ExtrinsicCalibrationManager()
    except Exception as e: rospy.logerr(f"Manager Crash: {e}")
    rospy.spin()
