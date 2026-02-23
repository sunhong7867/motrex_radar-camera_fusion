#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
역할:
- 레이더 PointCloud2를 프레임 버퍼에 누적하고, DetectionArray 수신 시 카메라 타임스탬프 기준 정렬 또는 최근 스택 방식으로 매칭 대상을 구성
- 레이더 점을 extrinsic(R, t)와 카메라 내부파라미터(K, D)로 영상 좌표에 투영한 뒤 bbox 내부 점만 선택해 객체 단위 연관 수행
- 거리, 최소 range, 최소 Doppler, 최소 power, 상위 power K개, bbox 상단 비율 필터를 순차 적용해 잡음을 억제하고 대표 점 집합을 구성
- Doppler 속도에 Cosine Correction을 적용해 시선각 영향을 보정하고, power 가중 중앙값 또는 일반 중앙값으로 객체 속도를 산출
- Doppler 신뢰도가 낮거나 값이 약한 구간에서는 트랙 중심 이동량 기반 속도 추정값으로 대체해 저속/원거리 안정성을 보완
- 최종 결과를 AssociationArray로 발행하고, 옵션 활성 시 RViz 확인용 MarkerArray를 함께 발행

입력:
- `~radar/pointcloud_topic` (`sensor_msgs/PointCloud2`)
- `~camera/camera_info_topic` (`sensor_msgs/CameraInfo`)
- `~bbox_topic` (`autocal/DetectionArray`)
- `~extrinsic_source/path` (`json`, R/t)

출력:
- `~associated_topic` (`autocal/AssociationArray`)
- `~debug/marker_topic` (`visualization_msgs/MarkerArray`, 선택)
"""

import json
import os
import re
from collections import deque
from typing import Optional, Tuple

import numpy as np
import rospy
import rospkg
from geometry_msgs.msg import Point
import sensor_msgs.point_cloud2 as pc2
from autocal.msg import Association, AssociationArray, BoundingBox, DetectionArray
from sensor_msgs.msg import CameraInfo, PointCloud2
from visualization_msgs.msg import Marker, MarkerArray

try:
    import cv2
except Exception:
    cv2 = None

_FIND_RE = re.compile(r"\$\(\s*find\s+([A-Za-z0-9_]+)\s*\)")

# ---------------------------------------------------------
# 투영 및 연관 파라미터
# ---------------------------------------------------------
DEFAULT_USE_DISTORTION = False             # 카메라 왜곡 계수 사용 여부
DEFAULT_PROJECTION_Z_MIN_M = 0.10          # 카메라 전방 최소 깊이(m)
DEFAULT_RADAR_HISTORY_FRAMES = 5           # 레이더 스택 프레임 수
DEFAULT_USE_BEV_ASSOCIATION = True         # 이전 BEV 중심 기반 게이팅 사용
DEFAULT_BEV_GATE_M = 3.5                   # BEV 게이팅 거리(m)
DEFAULT_USE_TEMPORAL_ALIGNMENT = True      # 검출 시각 기준 레이더 프레임 정렬
DEFAULT_MIN_RANGE_M = 0.0                  # 최소 거리(m)
DEFAULT_MIN_ABS_DOPPLER_MPS = 0.0          # 최소 절대 Doppler(m/s)
DEFAULT_MIN_POWER = 0.0                    # 최소 power
DEFAULT_TOPK_BY_POWER = 0                  # power 상위 K개만 사용(0=전체)
DEFAULT_USE_POWER_WEIGHTED_SPEED = True    # power 가중 중앙값 속도 사용
DEFAULT_USE_ABS_SPEED = True               # 속도 절대값 사용
DEFAULT_BBOX_TOP_HEIGHT_RATIO = 0.3        # bbox 상단 사용 비율

# ---------------------------------------------------------
# 트랙 기반 속도 보완 파라미터
# ---------------------------------------------------------
DEFAULT_TRACK_SPEED_ENABLE = True                      # 트랙 기반 속도 보완 사용
DEFAULT_TRACK_SPEED_USE_CAMERA_STAMP = False           # 트랙 속도 시간축에 카메라 stamp 사용
DEFAULT_TRACK_SPEED_USE_LATEST_RADAR_ONLY = True       # 최신 레이더 프레임만으로 트랙 속도 계산
DEFAULT_TRACK_SPEED_MIN_DT = 0.02                      # 트랙 속도 계산 최소 dt(s)
DEFAULT_TRACK_SPEED_MAX_DT = 0.80                      # 트랙 속도 계산 최대 dt(s)
DEFAULT_TRACK_SPEED_MIN_DISP_M = 0.03                  # 최소 이동량(m)
DEFAULT_TRACK_SPEED_MAX_MPS = 60.0                     # 트랙 속도 상한(m/s)
DEFAULT_TRACK_SPEED_EMA_ALPHA = 0.40                   # 트랙 속도 EMA 계수
DEFAULT_TRACK_SPEED_KEY_BUCKET_PX = 200.0              # det.id 중복 완화용 x 버킷 크기(px)
DEFAULT_TRACK_SPEED_FALLBACK_ABS_DOPPLER_LT_MPS = 1.0  # Doppler 절대값이 이 값보다 작으면 트랙 속도로 대체

def resolve_ros_path(path: str) -> str:
    """
    `$(find pkg)` 형식의 ROS 경로를 실제 절대 경로로 변환
    """
    if not isinstance(path, str):
        return path
    if "$(" not in path:
        return path
    rp = rospkg.RosPack()

    def _sub(m):
        pkg = m.group(1)
        try:
            return rp.get_path(pkg)
        except Exception:
            return m.group(0)

    return _FIND_RE.sub(_sub, path)


def load_extrinsic_json(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    extrinsic JSON 파일에서 회전행렬 R과 평행이동 t를 로드
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    R = np.array(payload["R"], dtype=np.float64)
    t = np.array(payload["t"], dtype=np.float64).reshape(3, 1)
    return R, t

def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """
    가중치를 고려한 중앙값 계산
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    m = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[m], weights[m]
    if values.size == 0:
        return float("nan")

    idx = np.argsort(values)
    v = values[idx]
    w = weights[idx]
    c = np.cumsum(w)
    cutoff = 0.5 * c[-1]
    return float(v[np.searchsorted(c, cutoff)])

def project_radar_to_image_with_indices(
    radar_xyz: np.ndarray,   # (N,3) radar frame
    cam_K: np.ndarray,
    cam_D: Optional[np.ndarray],
    R: np.ndarray,
    t: np.ndarray,
    w: int,
    h: int,
    use_distortion: bool,
    z_min_m: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    레이더 점을 카메라 영상 좌표로 투영하고 이미지 경계 안 점만 선별
    """
    Xr = radar_xyz.astype(np.float64).T
    Xc = (R @ Xr) + t.reshape(3, 1)
    z = Xc[2, :]

    valid = z > max(1e-6, float(z_min_m))
    idx0 = np.nonzero(valid)[0]
    if idx0.size == 0:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=np.int64)

    Xc_valid = Xc[:, valid]

    if use_distortion and (cam_D is not None) and (cv2 is not None):
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)
        pts = Xc_valid.T.reshape(-1, 1, 3)
        img_pts, _ = cv2.projectPoints(pts, rvec, tvec, cam_K, cam_D)
        uv = img_pts.reshape(-1, 2)
    else:
        uvw = cam_K @ Xc_valid
        uv = (uvw[:2, :] / uvw[2:3, :]).T

    u = uv[:, 0]
    v = uv[:, 1]
    in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)

    idx1 = np.nonzero(in_img)[0]
    if idx1.size == 0:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=np.int64)

    uv = uv[idx1]
    keep_idx = idx0[idx1]

    uv_int = np.round(uv).astype(np.int32)
    return uv_int, keep_idx.astype(np.int64)


class AssociationNode:
    def __init__(self):
        # 토픽 설정
        self.radar_topic = rospy.get_param("~radar/pointcloud_topic", "/point_cloud")
        self.camera_info_topic = rospy.get_param("~camera/camera_info_topic", "/camera/camera_info")
        self.bbox_topic = rospy.get_param("~bbox_topic", "/autocal/detections")
        self.associated_topic = rospy.get_param("~associated_topic", "/autocal/associated")

        # 좌표계 및 투영 설정
        self.radar_frame_default = rospy.get_param("~frames/radar_frame", "retina_link")
        self.use_distortion = bool(rospy.get_param("~projection/use_distortion", DEFAULT_USE_DISTORTION))
        self.z_min_m = float(rospy.get_param("~projection/z_min_m", DEFAULT_PROJECTION_Z_MIN_M))

        # 연관 파라미터
        self.history_frames = int(rospy.get_param("~association/radar_history_frames", DEFAULT_RADAR_HISTORY_FRAMES))
        self.use_bev_association = bool(rospy.get_param("~association/use_bev_association", DEFAULT_USE_BEV_ASSOCIATION))
        self.bev_gate_m = float(rospy.get_param("~association/bev_gate_m", DEFAULT_BEV_GATE_M))
        self.use_temporal_alignment = bool(rospy.get_param("~association/use_temporal_alignment", DEFAULT_USE_TEMPORAL_ALIGNMENT))
        self.min_range_m = float(rospy.get_param("~association/min_range_m", DEFAULT_MIN_RANGE_M))
        self.min_abs_doppler = float(rospy.get_param("~association/min_abs_doppler_mps", DEFAULT_MIN_ABS_DOPPLER_MPS))
        self.min_power = float(rospy.get_param("~association/min_power", DEFAULT_MIN_POWER))
        self.topk_by_power = int(rospy.get_param("~association/topk_by_power", DEFAULT_TOPK_BY_POWER)) 
        self.use_power_weighted = bool(rospy.get_param("~association/use_power_weighted_speed", DEFAULT_USE_POWER_WEIGHTED_SPEED))
        self.use_abs_speed = bool(rospy.get_param("~association/use_abs_speed", DEFAULT_USE_ABS_SPEED))
        self.bbox_top_h_ratio = float(rospy.get_param("~association/bbox_top_height_ratio", DEFAULT_BBOX_TOP_HEIGHT_RATIO))

        # 트랙 기반 지면 속도 파라미터
        self.track_speed_enable = bool(rospy.get_param("~track_speed/enable", DEFAULT_TRACK_SPEED_ENABLE))
        self.track_speed_use_camera_stamp = bool(rospy.get_param("~track_speed/use_camera_stamp", DEFAULT_TRACK_SPEED_USE_CAMERA_STAMP))
        self.track_speed_use_latest_radar_only = bool(
            rospy.get_param("~track_speed/use_latest_radar_only", DEFAULT_TRACK_SPEED_USE_LATEST_RADAR_ONLY)
        )
        self.track_speed_min_dt = float(rospy.get_param("~track_speed/min_dt", DEFAULT_TRACK_SPEED_MIN_DT))
        self.track_speed_max_dt = float(rospy.get_param("~track_speed/max_dt", DEFAULT_TRACK_SPEED_MAX_DT))
        self.track_speed_min_disp = float(rospy.get_param("~track_speed/min_disp_m", DEFAULT_TRACK_SPEED_MIN_DISP_M))
        self.track_speed_max_mps = float(rospy.get_param("~track_speed/max_mps", DEFAULT_TRACK_SPEED_MAX_MPS))
        self.track_speed_ema_alpha = float(rospy.get_param("~track_speed/ema_alpha", DEFAULT_TRACK_SPEED_EMA_ALPHA))
        self.track_speed_key_bucket_px = float(rospy.get_param("~track_speed/key_bucket_px", DEFAULT_TRACK_SPEED_KEY_BUCKET_PX))
        self.track_speed_fallback_abs_doppler_lt = float(
            rospy.get_param("~track_speed/fallback_when_abs_doppler_lt_mps", DEFAULT_TRACK_SPEED_FALLBACK_ABS_DOPPLER_LT_MPS)
        )

        # 외부파라미터 설정
        extr_path = rospy.get_param("~extrinsic_source/path", "$(find autocal)/config/extrinsic.json")
        self.extrinsic_path = resolve_ros_path(extr_path)
        self.extrinsic_reload_sec = float(rospy.get_param("~extrinsic_reload_sec", 1.0))

        # 디버그 마커 설정
        self.publish_markers = bool(rospy.get_param("~debug/publish_markers", True))
        self.marker_topic = rospy.get_param("~debug/marker_topic", "/autocal/association_markers")
        self.marker_lifetime = float(rospy.get_param("~debug/marker_lifetime", 0.2))
        self.point_scale = float(rospy.get_param("~debug/point_scale", 0.18))
        self.max_points_drawn = int(rospy.get_param("~debug/max_points_drawn", 400))

        # 내부 상태
        self._cam_K: Optional[np.ndarray] = None
        self._cam_D: Optional[np.ndarray] = None
        self._img_w: int = 0
        self._img_h: int = 0
        self._R = np.eye(3, dtype=np.float64)
        self._t = np.zeros((3, 1), dtype=np.float64)
        self._extr_mtime: float = 0.0
        self._last_extr_load = rospy.Time(0)
        self._rad_hist: deque = deque(maxlen=max(1, self.history_frames))
        self._track_last_xy = {}
        self._track_speed_ema = {}
        self._track_bev_center = {}

        # 퍼블리셔
        self.pub_assoc = rospy.Publisher(self.associated_topic, AssociationArray, queue_size=5)
        self.pub_markers = rospy.Publisher(self.marker_topic, MarkerArray, queue_size=2) if self.publish_markers else None

        # 서브스크라이버
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self._on_camera_info, queue_size=1)
        
        # DetectionArray와 PointCloud2를 비동기 버퍼 방식으로 수신
        rospy.Subscriber(self.radar_topic, PointCloud2, self._on_radar_pc, queue_size=5)
        rospy.Subscriber(self.bbox_topic, DetectionArray, self._on_detections, queue_size=5)

        rospy.loginfo(f"[association] Ready. radar={self.radar_topic}, bbox={self.bbox_topic}")

    def _on_camera_info(self, msg: CameraInfo):
        self._cam_K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
        self._cam_D = np.array(msg.D, dtype=np.float64).reshape(-1) if (msg.D is not None and len(msg.D) > 0) else None
        self._img_w = int(msg.width)
        self._img_h = int(msg.height)

    def _on_radar_pc(self, msg: PointCloud2):
        """
        레이더 데이터를 수신해 내부 버퍼에 누적
        """
        pts = self._read_radar_pointcloud(msg)
        # (header, points_Nx5) 저장
        self._rad_hist.append((msg.header, pts))

    def _read_radar_pointcloud(self, msg: PointCloud2) -> np.ndarray:
        field_names = [f.name for f in msg.fields]
        
        # 실제 드라이버 필드명 확인
        has_dop = "doppler" in field_names
        has_pow = "power" in field_names

        # 모든 필드가 있는 경우
        if has_dop and has_pow:
            use_fields = ("x","y","z","doppler","power")
            out = []
            for p in pc2.read_points(msg, field_names=use_fields, skip_nans=True):
                # 순서대로 [x, y, z, doppler, power]
                out.append([float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])])
            return np.array(out, dtype=np.float64) if out else np.zeros((0,5), dtype=np.float64)

        # doppler만 있는 경우
        if has_dop:
            use_fields = ("x","y","z","doppler")
            out = []
            for p in pc2.read_points(msg, field_names=use_fields, skip_nans=True):
                out.append([float(p[0]), float(p[1]), float(p[2]), float(p[3])])
            return np.array(out, dtype=np.float64) if out else np.zeros((0,4), dtype=np.float64)

        # xyz만 있는 경우
        use_fields = ("x","y","z")
        out = []
        for p in pc2.read_points(msg, field_names=use_fields, skip_nans=True):
            out.append([float(p[0]), float(p[1]), float(p[2])])
        return np.array(out, dtype=np.float64) if out else np.zeros((0,3), dtype=np.float64)

    def _maybe_reload_extrinsic(self):
        now = rospy.Time.now()
        if (now - self._last_extr_load).to_sec() < self.extrinsic_reload_sec:
            return
        self._last_extr_load = now

        try:
            st = os.stat(self.extrinsic_path)
            mtime = st.st_mtime
        except Exception:
            # 파일이 아직 없으면 경고만 하고 패스
            return

        if mtime <= self._extr_mtime:
            return

        try:
            R, t = load_extrinsic_json(self.extrinsic_path)
            self._R, self._t = R, t
            self._extr_mtime = mtime
            rospy.loginfo(f"[association] Reloaded extrinsic from {os.path.basename(self.extrinsic_path)}")
        except Exception as e:
            rospy.logwarn(f"[association] Extrinsic load failed: {e}")

    def _build_radar_stack(self) -> Tuple[Optional[rospy.Header], np.ndarray]:
        """
        버퍼에 있는 레이더 데이터들을 하나로 합침.
        FPS 차이로 인해 카메라 1프레임 동안 레이더 패킷이 1~2개 들어올 수 있음.
        """
        if len(self._rad_hist) == 0:
            return None, np.zeros((0, 3), dtype=np.float64)

        # 최근 N개 프레임 사용 (설정에 따라)
        items = list(self._rad_hist)[-max(1, self.history_frames):]
        # 헤더는 가장 최신 것 사용
        radar_header = items[-1][0]
        
        arrays = [it[1] for it in items if it[1] is not None and it[1].size > 0]
        if not arrays:
            return radar_header, np.zeros((0, 3), dtype=np.float64)

        cols = [a.shape[1] for a in arrays]
        # 컬럼 수가 같을 때만 합침 (안전장치)
        if all(c == cols[0] for c in cols):
            return radar_header, np.vstack(arrays)

        return radar_header, np.vstack([a[:, :3] for a in arrays])

    def _build_time_aligned_radar(self, target_stamp_sec: float) -> Tuple[Optional[rospy.Header], np.ndarray]:
        """
        입력 타임스탬프와 가장 가까운 레이더 프레임 선택
        """
        if len(self._rad_hist) == 0:
            return None, np.zeros((0, 3), dtype=np.float64)
        best = None
        best_dt = None
        for h, p in self._rad_hist:
            if h is None:
                continue
            ts = h.stamp.to_sec() if h.stamp else 0.0
            dt = abs(ts - target_stamp_sec)
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best = (h, p)
        if best is None:
            return None, np.zeros((0, 3), dtype=np.float64)
        h, p = best
        if p is None or p.size == 0:
            return h, np.zeros((0, 3), dtype=np.float64)
        return h, p
    
    def _get_latest_radar_frame(self):
        """
        최근 레이더 프레임 반환
        """
        if len(self._rad_hist) == 0:
            return None, np.zeros((0, 0), dtype=np.float64)
        h, p = self._rad_hist[-1]
        if p is None or p.size == 0:
            return h, np.zeros((0, 0), dtype=np.float64)
        return h, p

    def _on_detections(self, msg: DetectionArray):
        # 1. 캘리브레이션 갱신 확인
        self._maybe_reload_extrinsic()

        # 2. CameraInfo 수신 대기
        if self._cam_K is None or self._img_w <= 0:
            return

        # 3. 레이더 데이터 스택 가져오기
        if self.use_temporal_alignment and msg.header.stamp.to_sec() > 0:
            radar_header, pts = self._build_time_aligned_radar(msg.header.stamp.to_sec())
        else:
            radar_header, pts = self._build_radar_stack()
        # track-speed는 기본적으로 최신 레이더 프레임만 쓰도록 별도 보관
        latest_header, latest_pts = self._get_latest_radar_frame()
        if radar_header is None or pts.size == 0:
            self._publish_empty(msg, radar_header)
            return

        # 4. 1차 거리 필터링
        xyz = pts[:, :3]
        rng = np.linalg.norm(xyz, axis=1)
        # min_range_m 적용
        keep = rng >= max(0.0, self.min_range_m)
        pts = pts[keep]
        
        if pts.size == 0:
            self._publish_empty(msg, radar_header)
            return

        xyz = pts[:, :3]  # 필터링된 xyz

        # 5. 투영 (3D -> 2D)
        uv, keep_idx = project_radar_to_image_with_indices(
            radar_xyz=xyz,
            cam_K=self._cam_K,
            cam_D=self._cam_D,
            R=self._R,
            t=self._t,
            w=self._img_w,
            h=self._img_h,
            use_distortion=self.use_distortion,
            z_min_m=self.z_min_m
        )

        if uv.shape[0] == 0:
            self._publish_empty(msg, radar_header)
            return

        # 투영에 성공한 점들만 남김
        pts_valid = pts[keep_idx]
        uu = uv[:, 0].astype(np.float64)
        vv = uv[:, 1].astype(np.float64)

        out = AssociationArray()
        out.header = msg.header # 카메라 타임스탬프 유지
        out.radar_header = radar_header # 레이더 타임스탬프 보존
        out.objects = []

        # 디버깅용 마커
        marker_pack = MarkerArray() if self.publish_markers else None
        marker_id = 0
        radar_frame = radar_header.frame_id if radar_header.frame_id else self.radar_frame_default

        # track-speed용 최신 레이더 프레임 별도 투영(옵션)
        latest_uv = None
        latest_pts_valid = None
        latest_uu = latest_vv = None
        latest_radar_header = latest_header if latest_header is not None else radar_header
        if self.track_speed_enable and self.track_speed_use_latest_radar_only:
            if latest_pts is not None and latest_pts.size > 0 and self._cam_K is not None:
                # 최신 레이더 프레임도 동일한 range gate 적용
                lpts = latest_pts
                lxyz0 = lpts[:, :3]
                lrng = np.linalg.norm(lxyz0, axis=1)
                lkeep = lrng >= max(0.0, self.min_range_m)
                lpts = lpts[lkeep]
                lxyz = lpts[:, :3]
                luv, lkeep_idx = project_radar_to_image_with_indices(
                    radar_xyz=lxyz,
                    cam_K=self._cam_K,
                    cam_D=self._cam_D,
                    R=self._R,
                    t=self._t,
                    w=self._img_w,
                    h=self._img_h,
                    use_distortion=self.use_distortion,
                    z_min_m=self.z_min_m,
                )
                if luv.shape[0] > 0:
                    latest_uv = luv
                    latest_pts_valid = lpts[lkeep_idx]
                    latest_uu = latest_uv[:, 0].astype(np.float64)
                    latest_vv = latest_uv[:, 1].astype(np.float64)

        # 6. bbox별 처리 루프
        for det in msg.detections:
            x1, y1 = float(det.bbox.xmin), float(det.bbox.ymin)
            x2, y2 = float(det.bbox.xmax), float(det.bbox.ymax)

            # 박스 내부 점 찾기
            inside = (uu >= x1) & (uu <= x2) & (vv >= y1) & (vv <= y2)
            
            # bbox 상단 영역 포인트를 우선 선택
            # 뒤차의 포인트는 앞차 박스의 하단부에 걸칠 확률이 높으므로 이를 배제함.
            h_box = y2 - y1
            if h_box > 0:
                # 이미지 좌표계(v)는 위쪽이 0이므로, top 영역은 y1 ~ (y1 + h*ratio)
                v_limit = y1 + (h_box * self.bbox_top_h_ratio)
                inside_top = (vv <= v_limit)
                mask_final = inside & inside_top
                sel = pts_valid[mask_final]
            else:
                sel = pts_valid[inside]

            assoc = Association()
            assoc.id = int(det.id)
            assoc.cls = int(det.cls)
            assoc.score = float(det.score)
            assoc.bbox = det.bbox

            # 매칭된 점이 없으면 NaN 처리
            if sel.shape[0] == 0:
                assoc.dist_m = float("nan")
                assoc.speed_mps = float("nan")
                assoc.speed_kph = float("nan")
                assoc.num_radar_points = 0
                out.objects.append(assoc)
                continue

            # BEV 중심 기반 연관 보강
            if self.use_bev_association and sel.shape[0] > 0:
                track_key = f"{int(det.id)}:{int((0.5 * (x1 + x2)) / max(1.0, self.track_speed_key_bucket_px))}"
                prev_bev = self._track_bev_center.get(track_key)
                if prev_bev is not None:
                    xy = sel[:, :2]
                    bev_dist = np.linalg.norm(xy - np.array(prev_bev, dtype=np.float64), axis=1)
                    sel_gate = sel[bev_dist <= self.bev_gate_m]
                    if sel_gate.shape[0] > 0:
                        sel = sel_gate

            # 거리 계산
            sel_xyz = sel[:, :3]
            sel_rng = np.linalg.norm(sel_xyz, axis=1)
            dist_m = float(np.median(sel_rng))

            # -------------------------------------------------
            # 트랙 기반 지면 속도 계산
            # -------------------------------------------------
            track_speed_mps = float("nan")
            if self.track_speed_enable:
                track_id = int(det.id)
                # det.id 중복을 완화하기 위한 bucket 기반 key
                cx_pix = 0.5 * (x1 + x2)
                bucket = int(cx_pix / max(1.0, self.track_speed_key_bucket_px))
                track_key = f"{track_id}:{bucket}"
                
                if (not self.track_speed_use_camera_stamp) and (latest_radar_header is not None) and (latest_radar_header.stamp.to_sec() > 0):
                    t_sec = latest_radar_header.stamp.to_sec()
                elif self.track_speed_use_camera_stamp and (msg.header.stamp.to_sec() > 0):
                    t_sec = msg.header.stamp.to_sec()
                else:
                    t_sec = (latest_radar_header.stamp.to_sec() if latest_radar_header is not None else rospy.Time.now().to_sec())

                xy_src = sel_xyz
                if self.track_speed_use_latest_radar_only and (latest_pts_valid is not None) and (latest_uu is not None):
                    l_inside = (latest_uu >= x1) & (latest_uu <= x2) & (latest_vv >= y1) & (latest_vv <= y2)
                    
                    # track-speed 최신 프레임에도 동일한 상단 필터 적용
                    if h_box > 0:
                        l_v_limit = y1 + (h_box * self.bbox_top_h_ratio)
                        l_inside_top = (latest_vv <= l_v_limit)
                        l_mask = l_inside & l_inside_top
                    else:
                        l_mask = l_inside
                    
                    l_sel = latest_pts_valid[l_mask]
                    if l_sel is not None and l_sel.shape[0] > 0:
                        xy_src = l_sel[:, :3]

                cx = float(np.median(xy_src[:, 0]))
                cy = float(np.median(xy_src[:, 1]))

                prev = self._track_last_xy.get(track_key, None)
                if prev is not None:
                    t_prev, x_prev, y_prev = prev
                    dt = float(t_sec - t_prev)
                    if self.track_speed_min_dt <= dt <= self.track_speed_max_dt:
                        dx = float(cx - x_prev)
                        dy = float(cy - y_prev)
                        disp = float(np.hypot(dx, dy))
                        if disp >= self.track_speed_min_disp:
                            v = disp / dt
                            if v <= self.track_speed_max_mps:
                                ema_prev = self._track_speed_ema.get(track_key, v)
                                a = self.track_speed_ema_alpha
                                track_speed_mps = float(a * v + (1.0 - a) * ema_prev)
                                self._track_speed_ema[track_key] = track_speed_mps

                if (prev is None) or (t_sec > prev[0] + 1e-6):
                    self._track_last_xy[track_key] = (t_sec, cx, cy)
                self._track_bev_center[track_key] = (cx, cy)

            # Doppler 기반 속도 계산 및 보정
            speed_mps = float("nan")
            
            # 데이터 컬럼 파싱 (x, y, z, doppler, power)
            if sel.shape[1] >= 5:
                dop = sel[:, 3]
                pwr = sel[:, 4]
            elif sel.shape[1] >= 4:
                dop = sel[:, 3]
                pwr = np.ones_like(dop) # 파워 없으면 가중치 1
            else:
                dop = np.array([])
                pwr = np.array([])

            if dop.size > 0:
                # 6-1. 최소 속도 필터
                m_valid = np.isfinite(dop) & (np.abs(dop) >= max(0.0, self.min_abs_doppler))
                dop_m = dop[m_valid]
                pwr_m = pwr[m_valid] if (pwr.shape[0] == m_valid.shape[0]) else pwr[:dop_m.shape[0]]
                xyz_m = sel_xyz[m_valid]

                # 6-2. 최소 파워 필터
                if dop_m.size > 0:
                    m_pwr = np.isfinite(pwr_m) & (pwr_m >= self.min_power)
                    dop_m = dop_m[m_pwr]
                    pwr_m = pwr_m[m_pwr]
                    xyz_m = xyz_m[m_pwr]

                # 6-3. 상위 K개 파워 필터
                if self.topk_by_power > 0 and dop_m.size > self.topk_by_power:
                    idx = np.argsort(pwr_m)[-self.topk_by_power:]
                    dop_m = dop_m[idx]
                    pwr_m = pwr_m[idx]
                    xyz_m = xyz_m[idx]

                # 6-4. 기하학적 보정(Cosine Correction)
                if dop_m.size > 0:
                    corrected_dops = []
                    valid_pwrs = []
                    
                    for i in range(dop_m.size):
                        v_raw = dop_m[i]
                        px = xyz_m[i, 0] # 전방
                        py = xyz_m[i, 1] # 좌우
                        pz = xyz_m[i, 2] # 상하
                        dist = np.sqrt(px**2 + py**2 + pz**2)

                        if dist > 0.001:
                            cos_theta = abs(px) / dist
                            
                            if cos_theta > 0.1: 
                                v_corr = v_raw / cos_theta
                                corrected_dops.append(v_corr)
                                valid_pwrs.append(pwr_m[i])
                            else:
                                pass
                        else:
                            corrected_dops.append(v_raw)
                            valid_pwrs.append(pwr_m[i])

                    corrected_dops = np.array(corrected_dops)
                    valid_pwrs = np.array(valid_pwrs)

                    if corrected_dops.size > 0:
                        if self.use_power_weighted:
                            speed_mps = weighted_median(corrected_dops, valid_pwrs)
                        else:
                            speed_mps = float(np.median(corrected_dops))

                        if self.use_abs_speed and np.isfinite(speed_mps):
                            speed_mps = float(abs(speed_mps))
                    else:
                        speed_mps = float("nan")

            # 최종 속도 선택: Doppler 또는 track-speed 대체
            doppler_speed_mps = float(speed_mps)
            used_track_fallback = False
            if self.track_speed_enable and np.isfinite(track_speed_mps):
                if (not np.isfinite(speed_mps)) or (abs(speed_mps) < self.track_speed_fallback_abs_doppler_lt):
                    speed_mps = float(track_speed_mps)
                    used_track_fallback = True
                    if self.use_abs_speed and np.isfinite(speed_mps):
                        speed_mps = float(abs(speed_mps))

            if used_track_fallback:
                pass

            speed_kph = speed_mps * 3.6 if np.isfinite(speed_mps) else float("nan")

            assoc.dist_m = dist_m
            assoc.speed_mps = float(speed_mps)
            assoc.speed_kph = float(speed_kph)
            assoc.num_radar_points = int(sel.shape[0])
            out.objects.append(assoc)

            # 마커 추가 (시각화)
            if self.publish_markers and marker_pack is not None and sel.shape[0] > 0:
                # 상단 필터링된 대표 점들만 마커로 표시
                draw_xyz = sel_xyz
                if draw_xyz.shape[0] > self.max_points_drawn:
                    idx = np.linspace(0, draw_xyz.shape[0]-1, self.max_points_drawn).astype(np.int32)
                    draw_xyz = draw_xyz[idx]

                mk = Marker()
                mk.header.frame_id = radar_frame
                mk.header.stamp = radar_header.stamp
                mk.ns = "assoc_points"
                mk.id = marker_id
                marker_id += 1
                mk.type = Marker.SPHERE_LIST
                mk.action = Marker.ADD
                mk.scale.x = self.point_scale
                mk.scale.y = self.point_scale
                mk.scale.z = self.point_scale
                mk.color.a = 0.85
                mk.color.r = 1.0  # 노란색
                mk.color.g = 1.0
                mk.color.b = 0.0
                mk.lifetime = rospy.Duration.from_sec(self.marker_lifetime)
                for p in draw_xyz:
                    mk.points.append(Point(x=p[0], y=p[1], z=p[2]))
                marker_pack.markers.append(mk)

        self.pub_assoc.publish(out)
        if self.publish_markers and self.pub_markers is not None and marker_pack is not None:
            self.pub_markers.publish(marker_pack)

    def _publish_empty(self, detection_msg, radar_header):
        """
        매칭 실패 시 NaN 기반 빈 결과 발행
        """
        out = AssociationArray()
        out.header = detection_msg.header
        if radar_header is not None:
            out.radar_header = radar_header
        
        # 원본 박스들은 유지하되 값은 NaN으로
        for det in detection_msg.detections:
            assoc = Association()
            assoc.id = int(det.id)
            assoc.cls = int(det.cls)
            assoc.score = float(det.score)
            assoc.bbox = det.bbox
            assoc.dist_m = float("nan")
            assoc.speed_mps = float("nan")
            assoc.speed_kph = float("nan")
            assoc.num_radar_points = 0
            out.objects.append(assoc)
        
        self.pub_assoc.publish(out)


def main():
    rospy.init_node("association_node", anonymous=False)
    AssociationNode()
    rospy.spin()

if __name__ == "__main__":
    main()
