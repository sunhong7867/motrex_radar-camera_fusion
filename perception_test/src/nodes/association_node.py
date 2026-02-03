#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
association_node.py

역할:
- 레이더 PointCloud2를 카메라 픽셀(u,v)로 투영
- 입력 DetectionArray(bbox 리스트)를 기준으로 bbox 내부 레이더 점을 선택
- 선택된 레이더 점으로 거리(dist)와 도플러 기반 속도(speed)를 산출하여 publish
- [핵심] CARLA 시뮬레이션과 동일한 '기하학적 속도 보정(Cosine Correction)' 적용

입력:
- ~radar/pointcloud_topic (sensor_msgs/PointCloud2)
- ~camera/camera_info_topic (sensor_msgs/CameraInfo)
- ~bbox_topic (perception_test/DetectionArray)

출력:
- ~associated_topic (perception_test/AssociationArray)
- (옵션) ~debug/marker_topic (visualization_msgs/MarkerArray)
"""

import json
import os
import re
from collections import deque
from typing import Optional, Tuple

import numpy as np
import rospy
import rospkg
import message_filters  # 타임 싱크를 위해 필수

from sensor_msgs.msg import PointCloud2, CameraInfo
import sensor_msgs.point_cloud2 as pc2

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from perception_test.msg import DetectionArray, AssociationArray, Association, BoundingBox

try:
    import cv2
except Exception:
    cv2 = None


_FIND_RE = re.compile(r"\$\(\s*find\s+([A-Za-z0-9_]+)\s*\)")

def resolve_ros_path(path: str) -> str:
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
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    R = np.array(payload["R"], dtype=np.float64)
    t = np.array(payload["t"], dtype=np.float64).reshape(3, 1)
    return R, t

def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
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
    returns:
      uv_int: (M,2) int pixels
      keep_idx: (M,) indices in original radar_xyz
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
        # topics
        self.radar_topic = rospy.get_param("~radar/pointcloud_topic", "/point_cloud")
        self.camera_info_topic = rospy.get_param("~camera/camera_info_topic", "/camera/camera_info")
        self.bbox_topic = rospy.get_param("~bbox_topic", "/perception_test/detections")
        self.associated_topic = rospy.get_param("~associated_topic", "/perception_test/associated")

        # frames / projection
        self.radar_frame_default = rospy.get_param("~frames/radar_frame", "retina_link")
        self.use_distortion = bool(rospy.get_param("~projection/use_distortion", False))
        self.z_min_m = float(rospy.get_param("~projection/z_min_m", 0.10))

        # association parameters
        self.history_frames = int(rospy.get_param("~association/radar_history_frames", 5))
        self.min_range_m = float(rospy.get_param("~association/min_range_m", 0.0))
        self.min_abs_doppler = float(rospy.get_param("~association/min_abs_doppler_mps", 0.0))
        self.min_power = float(rospy.get_param("~association/min_power", 0.0))
        self.topk_by_power = int(rospy.get_param("~association/topk_by_power", 0)) 
        self.use_power_weighted = bool(rospy.get_param("~association/use_power_weighted_speed", True))
        self.use_abs_speed = bool(rospy.get_param("~association/use_abs_speed", True))

        # extrinsic
        extr_path = rospy.get_param("~extrinsic_source/path", "$(find perception_test)/config/extrinsic.json")
        self.extrinsic_path = resolve_ros_path(extr_path)
        self.extrinsic_reload_sec = float(rospy.get_param("~extrinsic_reload_sec", 1.0))

        # debug markers
        self.publish_markers = bool(rospy.get_param("~debug/publish_markers", True))
        self.marker_topic = rospy.get_param("~debug/marker_topic", "/perception_test/association_markers")
        self.marker_lifetime = float(rospy.get_param("~debug/marker_lifetime", 0.2))
        self.point_scale = float(rospy.get_param("~debug/point_scale", 0.18))
        self.max_points_drawn = int(rospy.get_param("~debug/max_points_drawn", 400))

        # internal states
        self._cam_K: Optional[np.ndarray] = None
        self._cam_D: Optional[np.ndarray] = None
        self._img_w: int = 0
        self._img_h: int = 0
        self._R = np.eye(3, dtype=np.float64)
        self._t = np.zeros((3, 1), dtype=np.float64)
        self._extr_mtime: float = 0.0
        self._last_extr_load = rospy.Time(0)

        # radar history: deque of (header, points_array)
        self._rad_hist: deque = deque(maxlen=max(1, self.history_frames))

        # Publishers
        self.pub_assoc = rospy.Publisher(self.associated_topic, AssociationArray, queue_size=5)
        self.pub_markers = rospy.Publisher(self.marker_topic, MarkerArray, queue_size=2) if self.publish_markers else None

        # Subscribers
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self._on_camera_info, queue_size=1)
        
        # [수정] Time Sync를 위한 MessageFilters 설정 (DetectionArray + PointCloud2)
        # 만약 동기화 없이 각각 콜백을 쓴다면 기존 방식을 쓰지만,
        # 여기서는 bbox가 들어왔을 때 가장 최신의 레이더 버퍼를 조회하는 방식(Async)을 유지합니다.
        # (이유: 레이더는 20fps, 카메라는 30fps로 서로 주기가 다르므로, 버퍼링 방식이 더 유리할 수 있음)
        rospy.Subscriber(self.radar_topic, PointCloud2, self._on_radar_pc, queue_size=5)
        rospy.Subscriber(self.bbox_topic, DetectionArray, self._on_detections, queue_size=5)

        rospy.loginfo(f"[association] Ready. radar={self.radar_topic}, bbox={self.bbox_topic}")

    def _on_camera_info(self, msg: CameraInfo):
        self._cam_K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
        self._cam_D = np.array(msg.D, dtype=np.float64).reshape(-1) if (msg.D is not None and len(msg.D) > 0) else None
        self._img_w = int(msg.width)
        self._img_h = int(msg.height)

    def _on_radar_pc(self, msg: PointCloud2):
        """레이더 데이터를 받아서 내부 버퍼에 쌓음 (비동기 수신)"""
        pts = self._read_radar_pointcloud(msg)
        # (header, points_Nx5) 저장
        self._rad_hist.append((msg.header, pts))

    def _read_radar_pointcloud(self, msg: PointCloud2) -> np.ndarray:
        field_names = [f.name for f in msg.fields]
        
        # [중요] 실제 드라이버 필드명 확인 (x, y, z, doppler, power)
        has_dop = "doppler" in field_names
        has_pow = "power" in field_names

        # 모든 필드가 다 있는 경우 (Best)
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

    def _on_detections(self, msg: DetectionArray):
        # 1. 캘리브레이션 갱신 확인
        self._maybe_reload_extrinsic()

        # 2. CameraInfo 대기
        if self._cam_K is None or self._img_w <= 0:
            return

        # 3. 레이더 데이터 스택 가져오기
        radar_header, pts = self._build_radar_stack()
        if radar_header is None or pts.size == 0:
            self._publish_empty(msg, radar_header)
            return

        # 4. 1차 필터링 (Range)
        xyz = pts[:, :3]
        rng = np.linalg.norm(xyz, axis=1)
        # min_range_m 적용
        keep = rng >= max(0.0, self.min_range_m)
        pts = pts[keep]
        
        if pts.size == 0:
            self._publish_empty(msg, radar_header)
            return

        xyz = pts[:, :3] # 필터링된 xyz

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

        # 6. 각 차량 박스(bbox)에 대해 루프
        for det in msg.detections:
            x1, y1 = float(det.bbox.xmin), float(det.bbox.ymin)
            x2, y2 = float(det.bbox.xmax), float(det.bbox.ymax)

            # 박스 내부 점 찾기
            inside = (uu >= x1) & (uu <= x2) & (vv >= y1) & (vv <= y2)
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

            # --- [핵심] 거리 계산 ---
            sel_xyz = sel[:, :3]
            sel_rng = np.linalg.norm(sel_xyz, axis=1)
            dist_m = float(np.median(sel_rng))

# --- Tracking-based speed (ground-plane) ---
# Use cluster center movement across frames (in radar XY) to estimate speed,
# so receding targets with small/zero Doppler still get speed.
track_id = int(det.id)
t_sec = radar_header.stamp.to_sec() if radar_header is not None else detection_msg.header.stamp.to_sec()
cx = float(np.median(sel_xyz[:, 0]))
cy = float(np.median(sel_xyz[:, 1]))
track_speed_mps = float("nan")

if self.track_speed_enable:
    prev = self._track_last_xy.get(track_id, None)
    if prev is not None:
        t_prev, x_prev, y_prev = prev
        dt = t_sec - t_prev
        if self.track_speed_min_dt <= dt <= self.track_speed_max_dt:
            dx = cx - x_prev
            dy = cy - y_prev
            disp = float(np.hypot(dx, dy))
            if disp >= self.track_speed_min_disp:
                v = disp / dt
                if v <= self.track_speed_max_mps:
                    # EMA smoothing per track_id
                    ema_prev = self._track_speed_ema.get(track_id, v)
                    alpha = self.track_speed_ema_alpha
                    track_speed_mps = float(alpha * v + (1.0 - alpha) * ema_prev)
                    self._track_speed_ema[track_id] = track_speed_mps

    # update last position (even if dt invalid, for next time)
    self._track_last_xy[track_id] = (t_sec, cx, cy)

            # --- [핵심] 속도 계산 및 보정 ---
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

                # 6-4. [중요] 기하학적 보정 (Cosine Correction)
                # CARLA 시뮬레이션 고정밀도의 핵심 로직
                if dop_m.size > 0:
                    corrected_dops = []
                    valid_pwrs = []
                    
                    for i in range(dop_m.size):
                        v_raw = dop_m[i]
                        # 레이더 좌표계 기준 좌표
                        px = xyz_m[i, 0] # 전방
                        py = xyz_m[i, 1] # 좌우
                        pz = xyz_m[i, 2] # 상하
                        dist = np.sqrt(px**2 + py**2 + pz**2)

                        # cos_theta = x / dist (전방 주행 방향과의 각도)
                        # v_real = v_measured / cos_theta
                        if dist > 0.001:
                            cos_theta = abs(px) / dist
                            
                            # 너무 측면(90도 근처)이면 보정이 불안정하므로 제외
                            if cos_theta > 0.1: 
                                v_corr = v_raw / cos_theta
                                corrected_dops.append(v_corr)
                                valid_pwrs.append(pwr_m[i])
                            else:
                                # 측면 데이터는 신뢰도가 낮음 -> 사용 안함
                                pass
                        else:
                            # 거리가 0에 가까우면 보정 불가
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

# --- Final speed selection: Doppler vs Track-speed fallback ---
if self.track_speed_enable and np.isfinite(track_speed_mps):
    if (not np.isfinite(speed_mps)) or (abs(speed_mps) < self.track_speed_fallback_abs_doppler_lt):
        speed_mps = float(track_speed_mps)
        if self.use_abs_speed and np.isfinite(speed_mps):
            speed_mps = float(abs(speed_mps))

speed_kph = speed_mps * 3.6 if np.isfinite(speed_mps) else float("nan")

            assoc.dist_m = dist_m
            assoc.speed_mps = float(speed_mps)
            assoc.speed_kph = float(speed_kph)
            assoc.num_radar_points = int(sel.shape[0])
            out.objects.append(assoc)

            # 마커 추가 (시각화)
            if self.publish_markers and marker_pack is not None and sel.shape[0] > 0:
                # 너무 많으면 샘플링
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
                mk.color.r = 1.0 # Yellow
                mk.color.g = 1.0
                mk.color.b = 0.0
                mk.lifetime = rospy.Duration.from_sec(self.marker_lifetime)
                for p in draw_xyz:
                    mk.points.append(Point(x=p[0], y=p[1], z=p[2]))
                marker_pack.markers.append(mk)

        self.pub_assoc.publish(out)
        rospy.loginfo_throttle(1.0, f"[association] published {len(out.objects)} objects | track_speed_enable={self.track_speed_enable}")
        if self.publish_markers and self.pub_markers is not None and marker_pack is not None:
            self.pub_markers.publish(marker_pack)

    def _publish_empty(self, detection_msg, radar_header):
        """매칭 실패 시 빈 결과 발행"""
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