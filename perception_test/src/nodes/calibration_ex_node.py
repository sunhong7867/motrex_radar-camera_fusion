#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
calibration_ex_node.py (Revised based on ITSC 2019 / Sensors 2025)

[문제 해결 접근]
1. 기존 solvePnP는 원거리 객체에 대해 R과 t간의 모호성으로 인해 t가 발산(1000m+)함.
2. 이를 해결하기 위해 'Decoupled Optimization' 전략 사용.
   - Step 1: t를 무시(0으로 가정)하고 Unit Vector 간의 SVD(Kabsch)를 통해 R을 최적화.
   - Step 2: 결정된 R을 고정하고, 잔여 오차를 줄이는 방향으로 t를 미세 조정(선택 사항).
3. 이 방식은 수학적으로 t가 발산하는 것을 원천 차단함.
"""

import json
import os
import re
from typing import List, Dict
from collections import deque

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
        
        # 1. 파라미터
        self.detections_topic = rospy.get_param('~detections_topic', '/perception_test/tracks')
        self.radar_topic = rospy.get_param('~radar_topic', '/point_cloud')
        self.camera_info_topic = rospy.get_param('~camera_info_topic', '/camera/camera_info')
        
        # -------------------------------------------------------------
        # [수정] 캘리브레이션 파라미터 (데이터 수집 조건 강화)
        # -------------------------------------------------------------
        self.min_samples = 600          # 최소 600개 이상의 포인트가 모여야 함 (기존 100 -> 600)
        self.req_duration = 15.0        # 최소 15초 동안은 무조건 수집 (시간 확보)
        
        self.min_track_len = 15         # 짧은 노이즈 트랙 제거
        self.min_cluster_points = 3     
        self.cluster_eps = 2.0          
        self.min_speed_mps = 3.0        # 3m/s (약 10km/h) 미만 정지 물체는 노이즈로 보고 제외
        self.max_assoc_angle_deg = 30.0 # 초기 매칭 허용 범위
        
        # 파일 경로 설정
        rp = rospkg.RosPack()
        default_path = os.path.join(rp.get_path('perception_test'), 'config', 'extrinsic.json')
        self.save_path = resolve_ros_path(rospy.get_param('~extrinsic_path', default_path))

        self.track_buffer = {} # {track_id: deque([(u,v), (x,y,z), ...])}
        self.K = None
        self.D = None
        self.inv_K = None      # 픽셀 -> 정규 좌표 변환용
        
        # 초기값 로드
        self.curr_R = np.eye(3)
        self.curr_t = np.zeros((3,1))
        self._load_extrinsic()

        # [수정] 수집 시작 시간 변수 초기화
        self.collection_start_time = None
        self.is_calibrating = False

        # 구독 설정
        sub_det = message_filters.Subscriber(self.detections_topic, DetectionArray)
        sub_rad = message_filters.Subscriber(self.radar_topic, PointCloud2)
        # ApproximateTimeSynchronizer: 0.1s 오차 허용 (20Hz 레이더 vs 30Hz 카메라)
        self.ts = message_filters.ApproximateTimeSynchronizer([sub_det, sub_rad], 30, 0.1)
        self.ts.registerCallback(self._callback)
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self._info_callback)

        rospy.loginfo(f"[Autocal] Started. Waiting for {self.req_duration}s & {self.min_samples} samples...")

    def _load_extrinsic(self):
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'r') as f:
                    data = json.load(f)
                self.curr_R = np.array(data['R'], dtype=np.float64)
                self.curr_t = np.array(data['t'], dtype=np.float64).reshape(3, 1)
                rospy.loginfo(f"[Autocal] Loaded R, t from {self.save_path}")
            except Exception as e:
                rospy.logwarn(f"[Autocal] Failed to load extrinsic: {e}")

    def _info_callback(self, msg):
        if self.K is None:
            self.K = np.array(msg.K).reshape(3, 3)
            self.D = np.array(msg.D)
            self.inv_K = np.linalg.inv(self.K)

    def _get_radar_centroids(self, rad_msg):
        # x, y, z, doppler 필드 읽기
        pts = list(pc2.read_points(rad_msg, field_names=("x", "y", "z", "doppler"), skip_nans=True))
        if not pts: return []
        data = np.array(pts)
        
        # 동적 객체 필터링 (정지 물체 제거)
        moving_mask = np.abs(data[:, 3]) > self.min_speed_mps
        moving_pts = data[moving_mask]
        if len(moving_pts) == 0: return []

        # 단순 클러스터링 (거리 기반)
        centroids = []
        visited = np.zeros(len(moving_pts), dtype=bool)
        for i in range(len(moving_pts)):
            if visited[i]: continue
            cluster_indices = []
            queue = deque([i])
            visited[i] = True
            while queue:
                curr = queue.popleft()
                cluster_indices.append(curr)
                # X, Y 평면 거리만 고려
                dists = np.linalg.norm(moving_pts[:, :2] - moving_pts[curr, :2], axis=1)
                neighbors = np.where((dists < self.cluster_eps) & (~visited))[0]
                for n in neighbors:
                    visited[n] = True
                    queue.append(n)
            
            if len(cluster_indices) >= self.min_cluster_points:
                cluster_pts = moving_pts[cluster_indices]
                # 클러스터 중심점 계산 (Median 사용으로 이상치 강건성 확보)
                centroids.append(np.median(cluster_pts[:, :3], axis=0))
        return centroids

    def _callback(self, det_msg, rad_msg):
        if self.K is None or self.is_calibrating: return
        
        # 1. 수집 시작 시간 기록 (최초 1회)
        if self.collection_start_time is None:
            self.collection_start_time = rospy.Time.now()
            rospy.loginfo("[Autocal] Timer Started.")

        centroids = self._get_radar_centroids(rad_msg)
        
        # 매칭 기준점: 박스 중앙
        valid_dets = [{'id': d.id, 'pt': ((d.bbox.xmin + d.bbox.xmax)/2.0, (d.bbox.ymin + d.bbox.ymax)/2.0)} 
                      for d in det_msg.detections if d.id > 0]

        if not centroids or not valid_dets: return

        # 데이터 매칭 (현재 추정된 R, t를 기준으로 각도 매칭)
        pairs = self._associate_angular(valid_dets, centroids)
        
        # 버퍼에 저장
        for tid, img_pt, obj_pt in pairs:
            if tid not in self.track_buffer:
                self.track_buffer[tid] = deque(maxlen=100) # 버퍼 크기 약간 증가
            self.track_buffer[tid].append((img_pt, obj_pt))

        # 데이터 수집 상태 확인
        total_samples = 0
        valid_tracks = 0
        for tid, pts in self.track_buffer.items():
            if len(pts) >= self.min_track_len:
                total_samples += len(pts)
                valid_tracks += 1
        
        # 경과 시간 계산
        elapsed = (rospy.Time.now() - self.collection_start_time).to_sec()
        
        # 상태 로그 출력 (2초마다)
        rospy.loginfo_throttle(1.0, f"[Autocal] Collecting... Time: {elapsed:.1f}/{self.req_duration}s | Samples: {total_samples}/{self.min_samples} (Tracks: {valid_tracks})")

        # [핵심] 종료 조건 검사: (시간 > 15초) AND (샘플 > 600개)
        if elapsed >= self.req_duration and total_samples >= self.min_samples:
            rospy.loginfo(f"[Autocal] Threshold Reached! Starting Optimization with {total_samples} samples.")
            self.run_optimization()

    def _associate_angular(self, valid_dets, centroids):
        """
        초기 R, t를 사용하여 레이더 점을 카메라 좌표계로 가상 투영한 뒤,
        카메라 픽셀의 각도(Azimuth, Elevation)와 비교하여 매칭.
        """
        results = []
        max_ang_rad = np.deg2rad(self.max_assoc_angle_deg)
        
        for det in valid_dets:
            u, v = det['pt']
            # Pixel -> Normalized Camera Ray (x, y, 1)
            nc = self.inv_K @ np.array([u, v, 1.0])
            cam_az = np.arctan2(nc[0], nc[2])
            cam_el = np.arctan2(nc[1], nc[2])

            best_idx = None
            min_diff = float('inf')
            
            for i, c in enumerate(centroids):
                # Radar(Local) -> Camera(Local) transform
                p_c = self.curr_R @ c.reshape(3, 1) + self.curr_t
                p_c = p_c.flatten()
                
                if p_c[2] <= 0: continue # 카메라 뒤쪽 점 제외

                rad_az = np.arctan2(p_c[0], p_c[2])
                rad_el = np.arctan2(p_c[1], p_c[2])
                
                # 각도 차이 계산 (L2 Norm)
                diff = np.hypot(cam_az - rad_az, cam_el - rad_el)
                
                if diff < max_ang_rad and diff < min_diff:
                    min_diff = diff
                    best_idx = i
            
            if best_idx is not None:
                results.append((det['id'], (u, v), centroids[best_idx]))
        return results

    def run_optimization(self):
        """
        [핵심 알고리즘] Kabsch Algorithm (SVD) for Rotation Alignment
        """
        self.is_calibrating = True
        rospy.loginfo("[Autocal] Running Optimization...")

        # 1. 데이터 준비
        radar_vecs = []
        cam_vecs = []
        
        for tid, points in self.track_buffer.items():
            if len(points) < self.min_track_len: continue
            
            for (u, v), (rx, ry, rz) in points:
                nc = self.inv_K @ np.array([u, v, 1.0])
                nc /= np.linalg.norm(nc)
                cam_vecs.append(nc)
                
                rv = np.array([rx, ry, rz])
                rv /= np.linalg.norm(rv)
                radar_vecs.append(rv)

        # 안전장치: 혹시라도 샘플이 부족하면 취소
        if len(radar_vecs) < self.min_samples:
            rospy.logwarn("[Autocal] Not enough valid points (Wait longer next time).")
            self.is_calibrating = False
            # 다시 모으기 위해 리셋
            self.collection_start_time = None 
            return

        A = np.array(radar_vecs).T 
        B = np.array(cam_vecs).T   

        # 2. SVD를 이용한 최적 회전 행렬 산출
        H = A @ B.T
        U, S, Vt = np.linalg.svd(H)
        R_opt = Vt.T @ U.T

        if np.linalg.det(R_opt) < 0:
            Vt[2, :] *= -1
            R_opt = Vt.T @ U.T

        rospy.loginfo(f"[Autocal] Rotation Optimized.\n{R_opt}")

        # 3. Translation은 기존 값 유지 (회전만 보정하는 것이 안전함)
        t_opt = self.curr_t 
        
        # 4. 결과 저장
        self.curr_R = R_opt
        self.curr_t = t_opt
        
        self._save_result(self.curr_R, self.curr_t)
        
        self.track_buffer.clear()
        rospy.loginfo("[Autocal] Calibration Finished & Saved.")
        
        # 캘리브레이션 완료 후 노드 종료
        self.is_calibrating = False
        rospy.signal_shutdown("Calibration Done") 

    def _save_result(self, R, t):
        data = {
            "R": R.tolist(),
            "t": t.flatten().tolist()
        }
        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=4)
            
if __name__ == '__main__':
    try: ExtrinsicCalibrationManager()
    except: pass
    rospy.spin()