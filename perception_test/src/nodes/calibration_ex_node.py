#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
from collections import defaultdict, deque
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
        self.min_samples = 800          # 800개 이상의 정교한 포인트 확보
        self.req_duration = 20.0        # 20초 이상 주행 데이터 수집
        self.min_track_len = 15         # 최소 15프레임 이상 추적된 궤적만 사용
        
        # [핵심] 정지 차량 배제: 시속 약 12.6km/h 미만은 노이즈로 간주
        self.min_speed_mps = 3.5        
        
        self.cluster_tol = 3.0          # 레이더 클러스터링 거리 임계값
        self.match_max_dist_px = 800.0  # 초기 매칭 허용 범위 (전역 매칭)
        self.cam_track_len = 60
        self.radar_track_len = 60
        self.radar_track_max_age_sec = 0.6
        self.radar_track_max_dist_m = 2.5
        self.cam_track_max_dt = float(rospy.get_param("~cam_track_max_dt", 0.15))
        self.lvalid_use_lane = bool(rospy.get_param("~lvalid/use_lane_validation", False))
        self.lane_anchor = rospy.get_param("~lvalid/lane_anchor", "bbox_bottom_center")
        self.lane_polys = {}
        self.lane_polys_path = None
        self.lane_polys_mtime = None
        if self.lvalid_use_lane:
            default_lane_path = os.path.join(rp.get_path("perception_test"), "src", "nodes", "lane_polys.json")
            self.lane_polys_path = resolve_ros_path(
                rospy.get_param("~lvalid/lane_polys_path", default_lane_path)
            )
            self._load_lane_polys(force=True)

        # RANSAC 설정
        self.ransac_iterations = 300
        self.ransac_threshold_deg = 2.0 # 2도 이내 오차만 인라이어 판정
        
        # 파일 경로 설정
        rp = rospkg.RosPack()
        default_path = os.path.join(rp.get_path('perception_test'), 'config', 'extrinsic.json')
        self.save_path = resolve_ros_path(rospy.get_param('~extrinsic_path', default_path))

        self.track_buffer = {}
        self.camera_tracks = {}
        self.radar_tracks = {}
        self.radar_track_meta = {}
        self.next_radar_track_id = 1
        self.K = None
        self.D = None 
        self.inv_K = None      
        
        # 초기값 로드
        self.curr_R = np.eye(3)
        self.curr_t = np.array([[0.08], [0.0], [-0.08]], dtype=np.float64) 
        self._load_extrinsic()

        self.collection_start_time = None
        self.is_calibrating = False

        # 구독 및 타임 싱크 설정
        sub_det = message_filters.Subscriber(self.detections_topic, DetectionArray)
        sub_rad = message_filters.Subscriber(self.radar_topic, PointCloud2)
        self.ts = message_filters.ApproximateTimeSynchronizer([sub_det, sub_rad], 50, 0.1)
        self.ts.registerCallback(self._callback)
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self._info_callback)
        self.pub_diag_start = rospy.Publisher("/perception_test/diagnosis/start", String, queue_size=1)

        rospy.loginfo("[Autocal] Robust Path-Matching Integration Started.")

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

    def _load_lane_polys(self, force: bool = False):
        if not self.lvalid_use_lane or not self.lane_polys_path:
            return
        try:
            if not os.path.exists(self.lane_polys_path):
                return
            mtime = os.path.getmtime(self.lane_polys_path)
            if (not force) and self.lane_polys_mtime is not None and mtime <= self.lane_polys_mtime:
                return
            with open(self.lane_polys_path, "r") as f:
                payload = json.load(f)
            lane_polys = {}
            for name, pts in payload.items():
                arr = np.array(pts, dtype=np.int32)
                if arr.ndim == 2 and arr.shape[0] >= 3:
                    lane_polys[name] = arr
            if lane_polys:
                self.lane_polys = lane_polys
                self.lane_polys_mtime = mtime
        except Exception as exc:
            rospy.logwarn(f"[Autocal] Failed to load lane polys: {exc}")

    def _lane_anchor_point(self, bbox):
        x1, y1, x2, y2 = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
        if self.lane_anchor == "bbox_center":
            return (x1 + x2) / 2.0, (y1 + y2) / 2.0
        return (x1 + x2) / 2.0, float(y2)

    def _assign_lane(self, bbox):
        if not self.lane_polys:
            return None
        cx, cy = self._lane_anchor_point(bbox)
        for lane_name, poly in self.lane_polys.items():
            if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                return lane_name
        return None

    def _select_lane_representatives(self, cam_paths, cam_lanes):
        lane_reps = {}
        for cid, lane in cam_lanes.items():
            if not lane:
                continue
            track_len = len(cam_paths.get(cid, []))
            prev = lane_reps.get(lane)
            if prev is None or track_len > prev[1]:
                lane_reps[lane] = (cid, track_len)
        return {lane: cid for lane, (cid, _len) in lane_reps.items()}

    def _compute_track_reprojection(self, R_opt, t_opt):
        if self.K is None:
            return None
        rvec, _ = cv2.Rodrigues(R_opt)
        tvec = t_opt.reshape(3, 1).astype(np.float32)
        summaries = []
        for tid, data in self.track_buffer.items():
            if len(data) < self.min_track_len:
                continue
            obj_pts = np.array([rv_3d for _, rv_3d, _ in data], dtype=np.float32)
            img_pts = np.array([uv for _, _, uv in data], dtype=np.float32)
            if obj_pts.size == 0:
                continue
            proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, self.K, self.D)
            proj = proj.reshape(-1, 2)
            diff = proj - img_pts
            rmse = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
            summaries.append((tid, rmse, len(data)))
        if summaries:
            rmses = [s[1] for s in summaries]
            mean_rmse = float(np.mean(rmses))
            median_rmse = float(np.median(rmses))
            rospy.loginfo(
                f"[Autocal] Track reprojection RMSE: mean={mean_rmse:.2f}px median={median_rmse:.2f}px"
            )
            worst = sorted(summaries, key=lambda x: x[1], reverse=True)[:3]
            for tid, rmse, cnt in worst:
                rospy.loginfo(f"[Autocal] Worst track {tid}: RMSE={rmse:.2f}px N={cnt}")
        return summaries
    
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

    def _update_radar_tracks(self, radar_objects, stamp_sec: float):
        if not radar_objects:
            return

        used_tracks = set()
        assignments = {}
        for cent_3d, vel in radar_objects:
            best_id = None
            best_dist = None
            for tid, track in self.radar_tracks.items():
                if tid in used_tracks or len(track) == 0:
                    continue
                last_pt = track[-1][1]
                dist = float(np.linalg.norm(last_pt[:2] - cent_3d[:2]))
                if dist <= self.radar_track_max_dist_m and (best_dist is None or dist < best_dist):
                    best_dist = dist
                    best_id = tid
            if best_id is None:
                tid = self.next_radar_track_id
                self.next_radar_track_id += 1
                self.radar_tracks[tid] = deque(maxlen=self.radar_track_len)
                self.radar_track_meta[tid] = {"last_ts": stamp_sec}
                assignments[tid] = (cent_3d, vel)
                used_tracks.add(tid)
            else:
                assignments[best_id] = (cent_3d, vel)
                used_tracks.add(best_id)

        for tid, (cent_3d, vel) in assignments.items():
            self.radar_tracks[tid].append((stamp_sec, cent_3d, vel))
            self.radar_track_meta[tid]["last_ts"] = stamp_sec

        stale = [
            tid for tid, meta in self.radar_track_meta.items()
            if (stamp_sec - meta.get("last_ts", stamp_sec)) > self.radar_track_max_age_sec
        ]
        for tid in stale:
            self.radar_tracks.pop(tid, None)
            self.radar_track_meta.pop(tid, None)

    def _project_radar_track(self, track_points):
        if self.K is None or len(track_points) == 0:
            return np.zeros((0, 2), dtype=np.float64)
        pts = np.array([p[1] for p in track_points], dtype=np.float64)
        T_mat = np.eye(4)
        T_mat[:3, :3] = self.curr_R
        T_mat[:3, 3] = self.curr_t.flatten()
        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
        pts_cam = (T_mat @ pts_h.T).T
        valid = pts_cam[:, 2] > 1.5
        pts_cam = pts_cam[valid]
        if pts_cam.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float64)
        uv = (self.K @ pts_cam[:, :3].T).T
        uv = uv[:, :2] / uv[:, 2:3]
        return uv

    def _path_avg_distance(self, path_points, query_points):
        if len(path_points) < 2 or query_points.shape[0] == 0:
            return None
        total = 0.0
        for q in query_points:
            dist, _ = self._closest_point_on_path(path_points, q)
            total += dist
        return total / float(query_points.shape[0])

    def _strip_cam_path(self, cam_path):
        return [(p[1], p[2]) for p in cam_path]

    def _match_radar_to_cam_by_time(self, radar_track, cam_track):
        pairs = []
        if not radar_track or len(cam_track) < 2:
            return pairs
        cam_times = np.array([p[0] for p in cam_track], dtype=np.float64)
        cam_uvs = np.array([[p[1], p[2]] for p in cam_track], dtype=np.float64)
        for stamp_sec, radar_pt, _vel in radar_track:
            idx = int(np.argmin(np.abs(cam_times - stamp_sec)))
            if abs(cam_times[idx] - stamp_sec) > self.cam_track_max_dt:
                continue
            pairs.append((cam_uvs[idx], radar_pt))
        return pairs
    
    def _callback(self, det_msg, rad_msg):
        # 최적화 수행 중에는 새로운 데이터 수집 중단 (무한 로딩 방지 1단계)
        if self.K is None or self.is_calibrating: return
        if self.collection_start_time is None: self.collection_start_time = rospy.Time.now()
        if self.lvalid_use_lane:
            self._load_lane_polys()

        radar_raw = np.array(list(pc2.read_points(rad_msg, field_names=("x", "y", "z", "doppler"), skip_nans=True)))
        if radar_raw.shape[0] == 0: return
        radar_objects = self._cluster_radar_points(radar_raw[:, :3], radar_raw[:, 3])
        stamp_sec = rad_msg.header.stamp.to_sec() if rad_msg.header.stamp else rospy.Time.now().to_sec()
        self._update_radar_tracks(radar_objects, stamp_sec)

        if not self.radar_tracks:
            return

        # -------------------------------------------------------------
        # [매칭 전략] 트랙 경로(Path) 기준 매칭
        # 단일 포인트가 아닌 최근 경로와의 거리로 연관
        # -------------------------------------------------------------
        cam_paths = {}
        cam_paths_xy = {}
        cam_lanes = {}
        cam_stamp = det_msg.header.stamp.to_sec() if det_msg.header.stamp else stamp_sec

        for det in det_msg.detections:
            if det.id <= 0:
                continue
            
            u_target, v_target = self._get_bbox_ref_point(det.bbox)
            
            if det.id not in self.camera_tracks:
                self.camera_tracks[det.id] = deque(maxlen=self.cam_track_len)
            self.camera_tracks[det.id].append((cam_stamp, u_target, v_target))
            cam_paths[det.id] = list(self.camera_tracks[det.id])
            cam_paths_xy[det.id] = self._strip_cam_path(cam_paths[det.id])
            if self.lvalid_use_lane:
                lane_id = getattr(det, "lane_id", "") or ""
                cam_lanes[det.id] = lane_id if lane_id else self._assign_lane(det.bbox)

        if not cam_paths:
            return

        radar_proj = {}
        radar_lself = defaultdict(dict)
        radar_ids = list(self.radar_tracks.keys())
        cam_ids = list(cam_paths.keys())
        if len(cam_ids) >= 2:
            cam_ids_sorted = sorted(cam_ids)
        else:
            cam_ids_sorted = cam_ids
        for rid in radar_ids:
            proj = self._project_radar_track(self.radar_tracks[rid])
            radar_proj[rid] = proj
            for cid in cam_ids:
                lself = self._path_avg_distance(cam_paths_xy[cid], proj)
                radar_lself[cid][rid] = lself

        pair_costs = []
        lane_reps = self._select_lane_representatives(cam_paths, cam_lanes) if self.lvalid_use_lane else {}
        for i, cid in enumerate(cam_ids_sorted):
            for rid in radar_ids:
                lself = radar_lself[cid][rid]
                if lself is None:
                    continue
                lvalid = lself
                if len(cam_ids_sorted) > 1:
                    tc_prime = None
                    if self.lvalid_use_lane:
                        lane = cam_lanes.get(cid)
                        if lane and lane_reps:
                            for other_lane, cand in lane_reps.items():
                                if other_lane != lane:
                                    tc_prime = cand
                                    break
                        if tc_prime is None:
                            for cand in cam_ids_sorted:
                                if cand == cid:
                                    continue
                                if lane is None or cam_lanes.get(cand) != lane:
                                    tc_prime = cand
                                    break
                    if tc_prime is None:
                        tc_prime = cam_ids_sorted[(i + 1) % len(cam_ids_sorted)]
                    lvalid = radar_lself.get(tc_prime, {}).get(rid, lself)
                cost = lself + lvalid
                pair_costs.append((cost, cid, rid, lself))

        pair_costs.sort(key=lambda x: x[0])
        assigned_cam = set()
        assigned_radar = set()
        cam_to_radar = {}
        for cost, cid, rid, lself in pair_costs:
            if cid in assigned_cam or rid in assigned_radar:
                continue
            if lself > self.match_max_dist_px:
                continue
            cam_to_radar[cid] = rid
            assigned_cam.add(cid)
            assigned_radar.add(rid)

        for det in det_msg.detections:
            if det.id <= 0:
                continue
            rid = cam_to_radar.get(det.id)
            if rid is None:
                continue
            cam_track = cam_paths.get(det.id, [])
            if len(cam_track) < 2:
                continue

            radar_track = list(self.radar_tracks.get(rid, []))
            if not radar_track:
                continue
            proj_pts = radar_proj.get(rid, np.zeros((0, 2), dtype=np.float64))
            if proj_pts.shape[0] == 0:
                continue

            path_points = cam_paths_xy.get(det.id, [])
            lself = self._path_avg_distance(path_points, proj_pts)
            if lself is None or lself >= self.match_max_dist_px:
                continue

            if det.id not in self.track_buffer:
                self.track_buffer[det.id] = deque(maxlen=120)

            for uv, radar_point in self._match_radar_to_cam_by_time(radar_track, cam_track):
                nc = self.inv_K @ np.array([uv[0], uv[1], 1.0])
                nc /= np.linalg.norm(nc)
                self.track_buffer[det.id].append((nc, radar_point, (uv[0], uv[1])))

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
            self._compute_track_reprojection(self.curr_R, self.curr_t)
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

    def _get_bbox_ref_point(self, bbox):
        x1, y1, x2, y2 = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
        mode = str(self.bbox_ref_mode).lower()
        if mode in {"bottom", "bottom_center", "bottom-center"}:
            return (x1 + x2) / 2.0, float(y2)
        if mode in {"top", "top_center", "top-center"}:
            return (x1 + x2) / 2.0, float(y1)
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0
    
    def _closest_point_on_path(self, path_points, query_pt):
        if len(path_points) == 1:
            pt = np.array(path_points[0], dtype=np.float64)
            return float(np.linalg.norm(pt - query_pt)), pt

        q = np.array(query_pt, dtype=np.float64)
        min_dist = float("inf")
        closest = None
        for i in range(len(path_points) - 1):
            p0 = np.array(path_points[i], dtype=np.float64)
            p1 = np.array(path_points[i + 1], dtype=np.float64)
            v = p1 - p0
            denom = np.dot(v, v)
            if denom == 0.0:
                proj = p0
            else:
                t = np.clip(np.dot(q - p0, v) / denom, 0.0, 1.0)
                proj = p0 + t * v
            dist = float(np.linalg.norm(q - proj))
            if dist < min_dist:
                min_dist = dist
                closest = proj
        return min_dist, closest
    
    def _save_result(self, R, t):
        data = {"R": R.tolist(), "t": t.flatten().tolist()}
        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=4)

if __name__ == '__main__':
    try: ExtrinsicCalibrationManager()
    except Exception as e: rospy.logerr(f"Manager Crash: {e}")
    rospy.spin()