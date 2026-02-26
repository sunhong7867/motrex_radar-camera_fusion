#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
역할:
- 트리거 수신 후 일정 시간 동안 트랙 bbox 기준점과 레이더 투영점의 대응을 수집해 외부파라미터 품질을 진단한다.
- 트랙별 RMSE와 진행 방향 차이를 계산하여 캘리브레이션 결과의 정합도를 텍스트 리포트로 발행한다.

입력:
- `~topics/detections` (`autocal/DetectionArray`): 트랙 ID와 bbox 정보.
- `~topics/radar` (`sensor_msgs/PointCloud2`): x, y, z, doppler를 포함한 레이더 포인트.
- `~topics/camera_info` (`sensor_msgs/CameraInfo`): 카메라 내부 파라미터 K.
- `/autocal/diagnosis/start` (`std_msgs/String`): 진단 시작 명령(top/center/bottom 기준점 모드).

출력:
- `/autocal/diagnosis/result` (`std_msgs/String`): 트랙 수, RMSE 통계, 상위 오차 트랙 요약.
"""

import json
import os
from collections import defaultdict, deque
from typing import Dict, Optional, Tuple, List

import message_filters
import numpy as np
import rospy
import rospkg
import sensor_msgs.point_cloud2 as pc2
from autocal.msg import DetectionArray
from sensor_msgs.msg import CameraInfo, PointCloud2
from std_msgs.msg import String

# ---------------------------------------------------------
# 진단 파라미터
# ---------------------------------------------------------
COLLECTION_DURATION_SEC = 30.0      # 진단 데이터 수집 시간(초)
MAX_COLLECTION_DURATION_SEC = 120.0 # 트랙 부족 시 최대 재시도 포함 총 수집 시간(초)
MIN_TRACK_LEN = 10                  # 유효 트랙 최소 길이(프레임)
MIN_TRACKS = 1                      # 리포트 생성 최소 트랙 수
CLUSTER_TOL_M = 3.0                 # 레이더 XY 클러스터 반경(m)
MIN_SPEED_MPS = 2.0                 # 정적 물체 제거 속도 임계값(m/s)
MATCH_MAX_DIST_PX = 160.0           # bbox 기준점-투영점 매칭 허용 거리(px)
MIN_TRACK_DISPLACEMENT_PX = 15.0    # 정지/저이동 트랙 제외를 위한 최소 기준점 이동량(px)
OCCLUSION_MARGIN_PX = 4.0           # 가림 판정 시 bbox 내부 여유 픽셀
MAX_U_BIAS_RATIO = 0.6              # bbox 폭 대비 허용 수평 편차 비율
MAX_V_BIAS_RATIO = 1.0              # bbox 높이 대비 허용 수직 편차 비율
MAX_ABS_U_ERR_PX = 80.0             # 수평 편차 상한(px)
MAX_ABS_V_ERR_PX = 100.0            # 수직 편차 상한(px)
MAX_RESIDUAL_JUMP_PX = 40.0         # 동일 트랙 잔차(투영-기준점) 급변 허용치(px)
PROJ_EMA_ALPHA = 0.35               # 트랙별 투영점 EMA 계수(지터 완화)
BOTTOM_ANCHOR_RATIO = 1.0           # bottom 모드 기준점 비율(1.0=하단, 0.75=하단에서 75%)
DEFAULT_LANE_NAMES = ["IN1", "IN2", "IN3", "OUT1", "OUT2", "OUT3"]


class TrajectoryEvaluationNode:
    def __init__(self):
        rospy.init_node("trajectory_evaluation_node", anonymous=False)

        self.detections_topic = rospy.get_param("~topics/detections", "/autocal/tracks")
        self.radar_topic = rospy.get_param("~topics/radar", "/point_cloud")
        self.camera_info_topic = rospy.get_param("~topics/camera_info", "/camera/camera_info")

        self.trigger_topic = "/autocal/diagnosis/start"
        self.result_topic = "/autocal/diagnosis/result"

        rp = rospkg.RosPack()
        default_ext_path = os.path.join(rp.get_path("autocal"), "config", "extrinsic.json")
        self.extrinsic_path = rospy.get_param("~extrinsic_path", default_ext_path)

        self.collection_duration = float(rospy.get_param("~collection_duration", COLLECTION_DURATION_SEC))
        self.max_collection_duration = float(rospy.get_param("~max_collection_duration", MAX_COLLECTION_DURATION_SEC))
        self.keep_collecting_until_min_tracks = bool(rospy.get_param("~keep_collecting_until_min_tracks", True))
        if self.max_collection_duration <= self.collection_duration:
            self.max_collection_duration = max(self.collection_duration * 4.0, self.collection_duration + 30.0)
            rospy.logwarn(f"[TrajectoryEval] max_collection_duration must be > collection_duration; auto-adjust to {self.max_collection_duration:.1f}s")
        self.min_track_len = int(rospy.get_param("~min_track_len", MIN_TRACK_LEN))
        self.min_tracks = int(rospy.get_param("~min_tracks", MIN_TRACKS))
        self.cluster_tol = float(rospy.get_param("~cluster_tol", CLUSTER_TOL_M))
        self.min_speed_mps = float(rospy.get_param("~min_speed_mps", MIN_SPEED_MPS))
        self.match_max_dist_px = float(rospy.get_param("~match_max_dist_px", MATCH_MAX_DIST_PX))
        self.min_track_displacement_px = float(rospy.get_param("~min_track_displacement_px", MIN_TRACK_DISPLACEMENT_PX))
        self.occlusion_margin_px = float(rospy.get_param("~occlusion_margin_px", OCCLUSION_MARGIN_PX))
        self.skip_occluded_refs = bool(rospy.get_param("~skip_occluded_refs", True))

        self.max_u_bias_ratio = float(rospy.get_param("~max_u_bias_ratio", MAX_U_BIAS_RATIO))
        self.max_v_bias_ratio = float(rospy.get_param("~max_v_bias_ratio", MAX_V_BIAS_RATIO))
        self.max_abs_u_err_px = float(rospy.get_param("~max_abs_u_err_px", MAX_ABS_U_ERR_PX))
        self.max_abs_v_err_px = float(rospy.get_param("~max_abs_v_err_px", MAX_ABS_V_ERR_PX))
        self.max_residual_jump_px = float(rospy.get_param("~max_residual_jump_px", MAX_RESIDUAL_JUMP_PX))
        self.proj_ema_alpha = float(rospy.get_param("~proj_ema_alpha", PROJ_EMA_ALPHA))
        self.bottom_anchor_ratio = float(rospy.get_param("~bottom_anchor_ratio", BOTTOM_ANCHOR_RATIO))
        self.fallback_lane_names = rospy.get_param("~fallback_lane_names", DEFAULT_LANE_NAMES)

        self.collecting = False
        self.start_time = rospy.Time(0)
        self.ref_mode = "bottom"
        self.cam_K = None
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))

        self.track_pairs: Dict[int, deque] = defaultdict(lambda: deque(maxlen=120))
        self.track_proj_ema: Dict[int, np.ndarray] = {}
        self.pub_result = rospy.Publisher(self.result_topic, String, queue_size=1)

        self.load_extrinsic()
        rospy.Subscriber(self.trigger_topic, String, self.on_trigger)
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self.cam_info_cb)

        self.sub_det = message_filters.Subscriber(self.detections_topic, DetectionArray)
        self.sub_radar = message_filters.Subscriber(self.radar_topic, PointCloud2)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.sub_det, self.sub_radar], 30, 0.2)
        self.ts.registerCallback(self.data_callback)

        rospy.loginfo("[TrajectoryEval] Diagnosis node ready")

    def on_trigger(self, msg: String):
        """
        트리거 수신 시 수집 시작 및 기준점 모드 설정
        """
        mode = (msg.data or "").strip().lower()
        if mode in {"top", "center", "bottom"} or mode.startswith("bottom"):
            self.ref_mode = mode
        else:
            self.ref_mode = "bottom"

        rospy.loginfo(f"[TrajectoryEval] Start collecting: {self.collection_duration}s, ref={self.ref_mode}")
        self.track_pairs.clear()
        self.track_proj_ema.clear()
        self.start_time = rospy.Time.now()
        self.collecting = True
        self.load_extrinsic()

    def load_extrinsic(self):
        """
        진단에 사용할 extrinsic(R, t)을 파일에서 로드
        """
        if not os.path.exists(self.extrinsic_path):
            return

        try:
            with open(self.extrinsic_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.R = np.array(data["R"], dtype=np.float64)
            self.t = np.array(data["t"], dtype=np.float64).reshape(3, 1)
        except Exception as exc:
            rospy.logwarn(f"[TrajectoryEval] Failed to load extrinsic: {exc}")

    def cam_info_cb(self, msg: CameraInfo):
        """
        CameraInfo에서 카메라 내부행렬 K 갱신
        """
        self.cam_K = np.array(msg.K, dtype=np.float64).reshape(3, 3)

    def _get_bbox_ref_point(self, bbox) -> Tuple[float, float]:
        """
        설정된 기준 모드(top/center/bottom)에 맞는 bbox 기준점 반환
        """
        x1, y1, x2, y2 = float(bbox.xmin), float(bbox.ymin), float(bbox.xmax), float(bbox.ymax)
        if self.ref_mode == "top":
            return (x1 + x2) / 2.0, y1
        if self.ref_mode == "center":
            return (x1 + x2) / 2.0, (y1 + y2) / 2.0

        ratio = self.bottom_anchor_ratio
        if self.ref_mode.startswith("bottom") and len(self.ref_mode) > 6:
            # 예: bottom80 -> 0.80, bottom75 -> 0.75
            suffix = self.ref_mode[6:]
            if suffix.isdigit():
                ratio = float(suffix) / 100.0
        ratio = float(np.clip(ratio, 0.5, 1.0))
        y_ref = y1 + (y2 - y1) * ratio
        return (x1 + x2) / 2.0, y_ref

    def _read_radar(self, radar_msg: PointCloud2) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        레이더 메시지에서 XYZ와 선택적 Doppler 필드 추출
        """
        try:
            radar_raw = np.array(
                list(pc2.read_points(radar_msg, field_names=("x", "y", "z", "doppler"), skip_nans=True))
            )
            if radar_raw.shape[0] == 0:
                return np.zeros((0, 3)), None
            return radar_raw[:, :3], radar_raw[:, 3]
        except Exception:
            radar_xyz = np.array(list(pc2.read_points(radar_msg, field_names=("x", "y", "z"), skip_nans=True)))
            if radar_xyz.shape[0] == 0:
                return np.zeros((0, 3)), None
            return radar_xyz[:, :3], None

    def _cluster_radar_points(self, points: np.ndarray, velocities: Optional[np.ndarray]):
        """
        XY 거리 기반 클러스터링 후 최소 속도 조건을 만족하는 중심점 반환
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
            vel_med = None
            if velocities is not None and velocities.size > 0:
                vel_med = float(np.median(velocities[neighbors]))
                if abs(vel_med) < self.min_speed_mps:
                    continue

            clusters.append((np.median(points[neighbors], axis=0), vel_med))

        return clusters

    def _project_points(self, pts_3d: np.ndarray) -> np.ndarray:
        """
        레이더 3D 점을 extrinsic과 K를 이용해 영상 좌표(u, v)로 투영
        """
        if self.cam_K is None or pts_3d.size == 0:
            return np.zeros((0, 2))

        pts_cam = (self.R @ pts_3d.T) + self.t
        valid = pts_cam[2, :] > 0.5
        pts_cam = pts_cam[:, valid]
        if pts_cam.shape[1] == 0:
            return np.zeros((0, 2))

        uv_h = self.cam_K @ pts_cam
        return (uv_h[:2, :] / uv_h[2, :]).T

    def data_callback(self, det_msg: DetectionArray, radar_msg: PointCloud2):
        """
        수집 중 bbox 기준점과 레이더 투영점을 매칭해 트랙별로 누적
        """
        if not self.collecting or self.cam_K is None:
            return

        radar_xyz, radar_doppler = self._read_radar(radar_msg)
        if radar_xyz.shape[0] == 0:
            return

        radar_clusters = self._cluster_radar_points(radar_xyz, radar_doppler)
        if not radar_clusters:
            return

        centers = np.array([c[0] for c in radar_clusters], dtype=np.float64)
        proj_pts = self._project_points(centers)
        if proj_pts.shape[0] == 0:
            return

        detections = [det for det in det_msg.detections if det.id > 0]
        refs = []
        valid_dets = []

        for det in detections:
            u_ref, v_ref = self._get_bbox_ref_point(det.bbox)
            if self.skip_occluded_refs and self._is_occluded_ref_point(det, u_ref, v_ref, detections):
                continue
            refs.append((float(u_ref), float(v_ref)))
            valid_dets.append(det)

        if not valid_dets:
            return

        assigned = self._assign_pairs(valid_dets, refs, proj_pts)
        for det_idx, proj_idx, dist in assigned:
            det = valid_dets[det_idx]
            u_ref, v_ref = refs[det_idx]
            proj_u, proj_v = float(proj_pts[proj_idx, 0]), float(proj_pts[proj_idx, 1])

            # 트랙별 EMA로 투영점 지터 완화
            prev_ema = self.track_proj_ema.get(det.id)
            if prev_ema is None:
                ema_u, ema_v = proj_u, proj_v
            else:
                a = float(np.clip(self.proj_ema_alpha, 0.05, 1.0))
                ema_u = (a * proj_u) + ((1.0 - a) * float(prev_ema[0]))
                ema_v = (a * proj_v) + ((1.0 - a) * float(prev_ema[1]))

            if dist >= self.match_max_dist_px:
                continue
            if not self._is_plausible_match(det.bbox, u_ref, v_ref, ema_u, ema_v):
                continue
            if not self._is_residual_consistent(det.id, u_ref, v_ref, ema_u, ema_v):
                continue

            self.track_proj_ema[det.id] = np.array([ema_u, ema_v], dtype=np.float64)
            bbox_w = float(max(1.0, det.bbox.xmax - det.bbox.xmin))
            bbox_h = float(max(1.0, det.bbox.ymax - det.bbox.ymin))
            radar_xy_dist = float(np.linalg.norm(centers[proj_idx, :2]))
            lane_id = str(getattr(det, "lane_id", "") or "UNKNOWN")

            self.track_pairs[det.id].append(
                {
                    "ref": (u_ref, v_ref),
                    "proj": (ema_u, ema_v),
                    "bbox_w": bbox_w,
                    "bbox_h": bbox_h,
                    "radar_dist": radar_xy_dist,
                    "lane_id": lane_id,
                    "bbox_cx": float((det.bbox.xmin + det.bbox.xmax) * 0.5),
                }
            )

    def _is_occluded_ref_point(self, target_det, u_ref: float, v_ref: float, all_dets) -> bool:
        """
        기준점이 더 가까운(other bbox bottom이 더 큰) 차량 bbox 내부에 있으면 가림으로 간주
        """
        tx1, ty1, tx2, ty2 = (
            target_det.bbox.xmin,
            target_det.bbox.ymin,
            target_det.bbox.xmax,
            target_det.bbox.ymax,
        )
        t_bottom = float(ty2)

        for other in all_dets:
            if other.id == target_det.id:
                continue

            ox1, oy1, ox2, oy2 = other.bbox.xmin, other.bbox.ymin, other.bbox.xmax, other.bbox.ymax
            if float(oy2) <= t_bottom:
                continue

            if (
                (ox1 + self.occlusion_margin_px) <= u_ref <= (ox2 - self.occlusion_margin_px)
                and (oy1 + self.occlusion_margin_px) <= v_ref <= (oy2 - self.occlusion_margin_px)
            ):
                return True

            ix1 = max(tx1, ox1)
            iy1 = max(ty1, oy1)
            ix2 = min(tx2, ox2)
            iy2 = min(ty2, oy2)
            iw = max(0.0, float(ix2 - ix1))
            ih = max(0.0, float(iy2 - iy1))
            inter = iw * ih
            t_area = max(1.0, float((tx2 - tx1) * (ty2 - ty1)))
            overlap_ratio = inter / t_area
            if overlap_ratio >= 0.6 and float(oy2) > t_bottom:
                return True

        return False

    def _assign_pairs(self, detections, refs, proj_pts: np.ndarray):
        """
        det 기준점과 투영점 간 1:1 할당(가능하면 Hungarian, 실패 시 greedy fallback)
        """
        if len(detections) == 0 or proj_pts.shape[0] == 0:
            return []

        ref_arr = np.array(refs, dtype=np.float64)
        dist_mat = np.linalg.norm(ref_arr[:, None, :] - proj_pts[None, :, :], axis=2)

        assignments = []
        try:
            from scipy.optimize import linear_sum_assignment

            large = 1e6
            cost = dist_mat.copy()
            cost[cost >= self.match_max_dist_px] = large
            row_idx, col_idx = linear_sum_assignment(cost)
            for r, c in zip(row_idx, col_idx):
                d = float(dist_mat[r, c])
                if d < self.match_max_dist_px:
                    assignments.append((int(r), int(c), d))
            return assignments
        except Exception:
            used_proj = set()
            cands = []
            for i in range(dist_mat.shape[0]):
                for j in range(dist_mat.shape[1]):
                    d = float(dist_mat[i, j])
                    if d < self.match_max_dist_px:
                        cands.append((d, i, j))
            cands.sort(key=lambda x: x[0])
            used_det = set()
            for d, i, j in cands:
                if i in used_det or j in used_proj:
                    continue
                used_det.add(i)
                used_proj.add(j)
                assignments.append((int(i), int(j), float(d)))
            return assignments

    def _is_residual_consistent(self, track_id: int, u_ref: float, v_ref: float, proj_u: float, proj_v: float) -> bool:
        """
        동일 트랙에서 잔차(du,dv)가 프레임 간 급변하면 오매칭으로 보고 제외
        """
        pairs = self.track_pairs.get(track_id)
        if pairs is None or len(pairs) == 0:
            return True

        last = pairs[-1]
        prev_u_ref, prev_v_ref = last["ref"]
        prev_u_proj, prev_v_proj = last["proj"]
        prev_du = float(prev_u_proj - prev_u_ref)
        prev_dv = float(prev_v_proj - prev_v_ref)
        cur_du = float(proj_u - u_ref)
        cur_dv = float(proj_v - v_ref)

        jump = float(np.hypot(cur_du - prev_du, cur_dv - prev_dv))
        return jump <= self.max_residual_jump_px

    def _is_plausible_match(self, bbox, u_ref: float, v_ref: float, proj_u: float, proj_v: float) -> bool:
        """
        bbox 크기/기준점 모드 기반 오프셋 및 위치 타당성 검사
        """
        x1, y1, x2, y2 = float(bbox.xmin), float(bbox.ymin), float(bbox.xmax), float(bbox.ymax)
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)

        du = abs(proj_u - u_ref)
        dv = abs(proj_v - v_ref)

        u_gate = min(self.max_abs_u_err_px, max(25.0, bw * self.max_u_bias_ratio))
        v_gate = min(self.max_abs_v_err_px, max(30.0, bh * self.max_v_bias_ratio))
        if not ((du <= u_gate) and (dv <= v_gate)):
            return False

        # bbox 경계 기반 추가 게이트(가림/오매칭 억제)
        if proj_u < (x1 - 0.25 * bw) or proj_u > (x2 + 0.25 * bw):
            return False

        mode = str(self.ref_mode).lower()
        if mode == "bottom":
            # 하단 기준점일 때 투영점이 박스 중상단에 뜨는 매칭 차단
            if proj_v < (y1 + 0.45 * bh) or proj_v > (y2 + 0.20 * bh):
                return False
        elif mode == "center":
            if proj_v < (y1 - 0.10 * bh) or proj_v > (y2 + 0.10 * bh):
                return False
        else:  # top
            if proj_v < (y1 - 0.20 * bh) or proj_v > (y1 + 0.60 * bh):
                return False

        return True

    def _fit_direction(self, points: np.ndarray) -> Optional[np.ndarray]:
        """
        점 집합의 주성분 방향벡터 계산
        """
        if points.shape[0] < 2:
            return None

        mean = points.mean(axis=0)
        centered = points - mean
        cov = np.cov(centered.T)
        if cov.shape != (2, 2):
            return None

        eigvals, eigvecs = np.linalg.eigh(cov)
        vec = eigvecs[:, np.argmax(eigvals)]
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            return None
        return vec / norm

    def analyze_and_report(self, force_finalize: bool = False) -> bool:
        """
        수집된 트랙 대응쌍으로 RMSE와 방향 오차를 계산해 리포트 발행
        """
        summaries = []

        for tid, pairs in self.track_pairs.items():
            if len(pairs) < self.min_track_len:
                continue

            ref_pts = np.array([p["ref"] for p in pairs], dtype=np.float64)
            rad_pts = np.array([p["proj"] for p in pairs], dtype=np.float64)
            bbox_w = np.array([p["bbox_w"] for p in pairs], dtype=np.float64)
            bbox_h = np.array([p["bbox_h"] for p in pairs], dtype=np.float64)
            radar_dist = np.array([p["radar_dist"] for p in pairs], dtype=np.float64)
            lane_labels: List[str] = [str(p.get("lane_id", "UNKNOWN")) for p in pairs]
            bbox_cx = np.array([float(p.get("bbox_cx", 0.0)) for p in pairs], dtype=np.float64)

            ref_disp = float(np.linalg.norm(ref_pts[-1] - ref_pts[0])) if ref_pts.shape[0] >= 2 else 0.0
            if ref_disp < self.min_track_displacement_px:
                continue

            diff = ref_pts - rad_pts
            errs = np.hypot(diff[:, 0], diff[:, 1])
            med = float(np.median(errs)) if errs.size > 0 else 0.0
            mad = float(np.median(np.abs(errs - med))) if errs.size > 0 else 0.0
            robust_gate = med + max(15.0, 3.0 * mad)
            inlier_mask = errs <= robust_gate
            if int(np.sum(inlier_mask)) < max(8, self.min_track_len // 2):
                continue

            ref_in = ref_pts[inlier_mask]
            rad_in = rad_pts[inlier_mask]
            bw_in = bbox_w[inlier_mask]
            bh_in = bbox_h[inlier_mask]
            dist_in = radar_dist[inlier_mask]
            lane_in = [lane_labels[i] for i, ok in enumerate(inlier_mask) if ok]
            cx_in = bbox_cx[inlier_mask]

            diff_in = ref_in - rad_in
            rmse_abs = float(np.sqrt(np.mean(np.sum(diff_in ** 2, axis=1))))
            bias_uv = np.mean(diff_in, axis=0)
            diff_db = diff_in - bias_uv
            rmse_db = float(np.sqrt(np.mean(np.sum(diff_db ** 2, axis=1))))

            du = rad_in[:, 0] - ref_in[:, 0]
            dv = rad_in[:, 1] - ref_in[:, 1]
            ru = du / np.maximum(bw_in, 1.0)
            rv = dv / np.maximum(bh_in, 1.0)
            ru_mean_abs = float(np.mean(np.abs(ru)))
            rv_mean_abs = float(np.mean(np.abs(rv)))

            dir_ref = self._fit_direction(ref_in)
            dir_rad = self._fit_direction(rad_in)
            angle = None
            if dir_ref is not None and dir_rad is not None:
                dot = float(np.clip(np.abs(np.dot(dir_ref, dir_rad)), -1.0, 1.0))
                angle = float(np.degrees(np.arccos(dot)))

            summaries.append(
                {
                    "id": tid,
                    "rmse_abs": rmse_abs,
                    "rmse_db": rmse_db,
                    "bias_u": float(bias_uv[0]),
                    "bias_v": float(bias_uv[1]),
                    "angle": angle,
                    "count": len(pairs),
                    "disp": ref_disp,
                    "inliers": int(np.sum(inlier_mask)),
                    "ru_abs": ru_mean_abs,
                    "rv_abs": rv_mean_abs,
                    "dists": dist_in.tolist(),
                    "errs_abs": np.hypot(diff_in[:, 0], diff_in[:, 1]).tolist(),
                    "lanes": lane_in,
                    "du": du.tolist(),
                    "dv": dv.tolist(),
                    "cx": cx_in.tolist(),

                }
            )

        if len(summaries) < self.min_tracks:
            if (not force_finalize) or self.keep_collecting_until_min_tracks:
                rospy.logwarn_throttle(
                    2.0,
                    f"[TrajectoryEval] valid_tracks={len(summaries)} < min_tracks={self.min_tracks}; keep collecting...",
                )
                return False

            msg = (
                f"⚠ [평가 실패] 유효 트랙 부족 ({len(summaries)}개)\n"
                f"- 최소 트랙 수: {self.min_tracks}\n"
                f"- 트랙 기준 길이: {self.min_track_len} 프레임"
            )
            self.pub_result.publish(msg)
            self.collecting = False
            return False

        rmses_abs = np.array([s["rmse_abs"] for s in summaries], dtype=np.float64)
        rmses_db = np.array([s["rmse_db"] for s in summaries], dtype=np.float64)
        angles = np.array([s["angle"] for s in summaries if s["angle"] is not None], dtype=np.float64)
        ru_abs_all = np.array([s["ru_abs"] for s in summaries], dtype=np.float64)
        rv_abs_all = np.array([s["rv_abs"] for s in summaries], dtype=np.float64)

        rms_mean = float(np.mean(rmses_abs))
        rms_median = float(np.median(rmses_abs))
        rms_db_mean = float(np.mean(rmses_db))
        rms_db_median = float(np.median(rmses_db))
        ang_mean = float(np.mean(angles)) if angles.size > 0 else float("nan")
        sorted_by_rmse = sorted(summaries, key=lambda x: x["rmse_abs"])
        worst = sorted_by_rmse[::-1][:3]
        best = sorted_by_rmse[:3]

        lines = [
            "📊 궤적 기반 평가 결과",
            f"- 기준점 모드: {self.ref_mode}",
            f"- 유효 트랙 수: {len(summaries)}",
            f"- 최소 이동량 필터(px): {self.min_track_displacement_px:.1f}",
            f"- 평균 RMSE(abs, px): {rms_mean:.2f}",
            f"- 중앙값 RMSE(abs, px): {rms_median:.2f}",
            f"- 평균 RMSE(debiased, px): {rms_db_mean:.2f}",
            f"- 중앙값 RMSE(debiased, px): {rms_db_median:.2f}",
            f"- EMA alpha: {self.proj_ema_alpha:.2f}",
            f"- bottom anchor ratio: {self.bottom_anchor_ratio:.2f}",
            f"- |ru| 평균: {float(np.mean(ru_abs_all)):.3f}",
            f"- |rv| 평균: {float(np.mean(rv_abs_all)):.3f}",
        ]
        if angles.size > 0:
            lines.append(f"- 평균 방향 각도 차(°): {ang_mean:.2f}")

        lines.append("")
        lines.append("🔎 상위 오차 트랙")
        for s in worst:
            ang_str = f"{s['angle']:.2f}°" if s["angle"] is not None else "N/A"
            lines.append(
                f"- ID {s['id']}: RMSE(abs) {s['rmse_abs']:.2f}px, RMSE(db) {s['rmse_db']:.2f}px, Δθ {ang_str}, bias=({s['bias_u']:.1f},{s['bias_v']:.1f}), |ru|={s['ru_abs']:.3f}, |rv|={s['rv_abs']:.3f}, N={s['count']}, Inlier={s['inliers']}, 이동량 {s['disp']:.1f}px"
            )

        lines.append("")
        lines.append("✅ 하위 오차 트랙")
        for s in best:
            ang_str = f"{s['angle']:.2f}°" if s["angle"] is not None else "N/A"
            lines.append(
                f"- ID {s['id']}: RMSE(abs) {s['rmse_abs']:.2f}px, RMSE(db) {s['rmse_db']:.2f}px, Δθ {ang_str}, bias=({s['bias_u']:.1f},{s['bias_v']:.1f}), |ru|={s['ru_abs']:.3f}, |rv|={s['rv_abs']:.3f}, N={s['count']}, Inlier={s['inliers']}, 이동량 {s['disp']:.1f}px"
            )


        # 거리 구간(자동 10m bin) RMSE 요약
        dist_lines = []
        all_dist = []
        all_err = []
        all_lane = []
        all_du = []
        all_dv = []
        all_cx = []
        for sm in summaries:
            all_dist.extend(sm.get("dists", []))
            all_err.extend(sm.get("errs_abs", []))
            all_lane.extend(sm.get("lanes", []))
            all_du.extend(sm.get("du", []))
            all_dv.extend(sm.get("dv", []))
            all_cx.extend(sm.get("cx", []))

        all_dist = np.array(all_dist, dtype=np.float64) if all_dist else np.array([], dtype=np.float64)
        all_err = np.array(all_err, dtype=np.float64) if all_err else np.array([], dtype=np.float64)
        all_du = np.array(all_du, dtype=np.float64) if all_du else np.array([], dtype=np.float64)
        all_dv = np.array(all_dv, dtype=np.float64) if all_dv else np.array([], dtype=np.float64)
        all_cx = np.array(all_cx, dtype=np.float64) if all_cx else np.array([], dtype=np.float64)

        if all_du.size > 0 and all_dv.size > 0:
            gb_u = float(np.mean(all_du))
            gb_v = float(np.mean(all_dv))
            lines.append("")
            lines.append(f"- 전역 Bias(u,v) px: ({gb_u:.2f}, {gb_v:.2f})")

        if all_dist.size > 0 and all_err.size > 0:
            min_d = float(np.min(all_dist))
            max_d = float(np.max(all_dist))
            start_d = int(np.floor(min_d / 10.0) * 10)
            end_d = int(np.floor(max_d / 10.0) * 10) + 10
            dist_edges = list(range(start_d, end_d, 10))

            lines.append("")
            lines.append(f"📏 거리 구간별 RMSE(abs) [{start_d}~{end_d}m]")
            for d0 in dist_edges:
                d1 = d0 + 10
                mask = (all_dist >= d0) & (all_dist < d1)
                if int(np.sum(mask)) < 5:
                    continue
                rmse_bin = float(np.sqrt(np.mean(all_err[mask] ** 2)))
                bu = float(np.mean(all_du[mask])) if all_du.size == all_err.size else float("nan")
                bv = float(np.mean(all_dv[mask])) if all_dv.size == all_err.size else float("nan")
                lines.append(f"- {d0:02d}-{d1:02d}m: RMSE {rmse_bin:.2f}px, Bias({bu:.2f},{bv:.2f}) (N={int(np.sum(mask))})")

        # 차선별 RMSE(abs) 요약
        if all_lane and len(all_lane) == len(all_err):
            lines.append("")
            lines.append("🛣️ 차선별 RMSE(abs)")

            lane_array = np.array([str(x) if str(x) else "UNKNOWN" for x in all_lane], dtype=object)
            known_mask = lane_array != "UNKNOWN"

            # lane_id가 거의 전달되지 않는 경우 bbox x-center 기반 6분할 fallback
            if int(np.sum(known_mask)) < max(10, int(0.1 * len(lane_array))) and all_cx.size == len(lane_array):
                if all_cx.size >= 12:
                    q = np.percentile(all_cx, [100.0/6, 200.0/6, 300.0/6, 400.0/6, 500.0/6])
                    fallback_names = list(self.fallback_lane_names) if isinstance(self.fallback_lane_names, list) else DEFAULT_LANE_NAMES
                    if len(fallback_names) != 6:
                        fallback_names = DEFAULT_LANE_NAMES
                    for i, cx in enumerate(all_cx):
                        b = int(np.sum(cx > q))
                        lane_array[i] = fallback_names[b]
                else:
                    lane_array[:] = "UNKNOWN"

            lane_to_errs = defaultdict(list)
            for lane, e in zip(lane_array, all_err):
                lane_to_errs[str(lane)].append(float(e))

            for lane, errs_lane in sorted(lane_to_errs.items(), key=lambda kv: kv[0]):
                if len(errs_lane) < 5:
                    continue
                arr = np.array(errs_lane, dtype=np.float64)
                rmse_lane = float(np.sqrt(np.mean(arr ** 2)))
                lane_idx = [i for i, ln in enumerate(lane_array) if str(ln) == lane]
                bu = float(np.mean(all_du[lane_idx])) if all_du.size == len(lane_array) and lane_idx else float("nan")
                bv = float(np.mean(all_dv[lane_idx])) if all_dv.size == len(lane_array) and lane_idx else float("nan")
                lines.append(f"- {lane}: RMSE {rmse_lane:.2f}px, Bias({bu:.2f},{bv:.2f}) (N={len(errs_lane)})")

        self.pub_result.publish("\n".join(lines))
        self.collecting = False
        return True

    def run(self):
        """
        수집 시간을 주기 감시하고 만료 시 자동 리포트 생성
        """
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.collecting:
                elapsed = (rospy.Time.now() - self.start_time).to_sec()
                if elapsed > self.collection_duration:
                    force_finalize = elapsed >= self.max_collection_duration
                    done = self.analyze_and_report(force_finalize=force_finalize)
                    if not done and not force_finalize:
                        rospy.loginfo_throttle(
                            2.0,
                            f"[TrajectoryEval] waiting for more valid tracks... elapsed={elapsed:.1f}s/{self.max_collection_duration:.1f}s"
                        )
                    if not done and force_finalize and self.keep_collecting_until_min_tracks:
                        rospy.logwarn_throttle(
                            2.0,
                            "[TrajectoryEval] max duration reached but still waiting for min tracks; extending collection window"
                        )
                        self.start_time = rospy.Time.now()
                        continue
                    if not done and force_finalize:
                        rospy.logwarn(
                            "[TrajectoryEval] finalize with insufficient tracks after max collection duration"
                        )
            rate.sleep()


if __name__ == "__main__":
    node = TrajectoryEvaluationNode()
    node.run()
