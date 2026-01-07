#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
association_node.py

역할:
- 레이더 PointCloud2를 카메라 픽셀(u,v)로 투영
- 입력 DetectionArray(bbox 리스트)를 기준으로 bbox 내부 레이더 점을 선택
- 선택된 레이더 점으로 거리(dist)와 도플러 기반 속도(speed)를 산출하여 publish

입력:
- ~radar/pointcloud_topic (sensor_msgs/PointCloud2)
- ~camera/camera_info_topic (sensor_msgs/CameraInfo)
- ~bbox_topic (perception/DetectionArray)

출력:
- ~associated_topic (perception/AssociationArray)
- (옵션) ~debug/marker_topic (visualization_msgs/MarkerArray)

주의:
- YAML에 $(find pkg) 같은 치환을 쓰면 roslaunch <rosparam subst_value="true"> 권장. :contentReference[oaicite:3]{index=3}
"""

import json
import os
import re
from collections import deque
from typing import Optional, Tuple

import numpy as np
import rospy
import rospkg

from sensor_msgs.msg import PointCloud2, CameraInfo
import sensor_msgs.point_cloud2 as pc2

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from perception.msg import DetectionArray, AssociationArray, Association, BoundingBox

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
        self.bbox_topic = rospy.get_param("~bbox_topic", "/perception/detections")
        self.associated_topic = rospy.get_param("~associated_topic", "/perception/associated")

        # frames / projection
        self.radar_frame_default = rospy.get_param("~frames/radar_frame", "retina_link")
        self.use_distortion = bool(rospy.get_param("~projection/use_distortion", False))
        self.z_min_m = float(rospy.get_param("~projection/z_min_m", 0.10))

        # association
        self.history_frames = int(rospy.get_param("~association/radar_history_frames", 5))
        self.min_range_m = float(rospy.get_param("~association/min_range_m", 0.0))
        self.min_abs_doppler = float(rospy.get_param("~association/min_abs_doppler_mps", 0.0))

        # extrinsic
        extr_path = rospy.get_param("~extrinsic_source/path", "$(find perception)/src/nodes/extrinsic.json")
        self.extrinsic_path = resolve_ros_path(extr_path)
        self.extrinsic_reload_sec = float(rospy.get_param("~extrinsic_reload_sec", 1.0))

        # debug markers
        self.publish_markers = bool(rospy.get_param("~debug/publish_markers", True))
        self.marker_topic = rospy.get_param("~debug/marker_topic", "/perception/association_markers")
        self.marker_lifetime = float(rospy.get_param("~debug/marker_lifetime", 0.2))
        self.point_scale = float(rospy.get_param("~debug/point_scale", 0.18))
        self.max_points_drawn = int(rospy.get_param("~debug/max_points_drawn", 400))

        # camera intrinsics from CameraInfo
        self._cam_K: Optional[np.ndarray] = None
        self._cam_D: Optional[np.ndarray] = None
        self._img_w: int = 0
        self._img_h: int = 0

        # extrinsic
        self._R = np.eye(3, dtype=np.float64)
        self._t = np.zeros((3, 1), dtype=np.float64)
        self._extr_mtime: float = 0.0
        self._last_extr_load = rospy.Time(0)

        # radar history: each item = (header, np array)
        self._rad_hist: deque = deque(maxlen=max(1, self.history_frames))

        # pubs/subs
        self.pub_assoc = rospy.Publisher(self.associated_topic, AssociationArray, queue_size=5)
        self.pub_markers = rospy.Publisher(self.marker_topic, MarkerArray, queue_size=2) if self.publish_markers else None

        rospy.Subscriber(self.camera_info_topic, CameraInfo, self._on_camera_info, queue_size=1)
        rospy.Subscriber(self.radar_topic, PointCloud2, self._on_radar_pc, queue_size=1)
        rospy.Subscriber(self.bbox_topic, DetectionArray, self._on_detections, queue_size=3)

        rospy.loginfo(f"[association] radar={self.radar_topic}, cam_info={self.camera_info_topic}, det={self.bbox_topic}")
        rospy.loginfo(f"[association] extrinsic_path={self.extrinsic_path}, use_distortion={self.use_distortion}")

    def _on_camera_info(self, msg: CameraInfo):
        self._cam_K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
        self._cam_D = np.array(msg.D, dtype=np.float64).reshape(-1) if (msg.D is not None and len(msg.D) > 0) else None
        self._img_w = int(msg.width)
        self._img_h = int(msg.height)

    def _on_radar_pc(self, msg: PointCloud2):
        pts = self._read_radar_pointcloud(msg)
        self._rad_hist.append((msg.header, pts))

    def _read_radar_pointcloud(self, msg: PointCloud2) -> np.ndarray:
        field_names = [f.name for f in msg.fields]
        need = ("x","y","z","doppler","power")
        if not all(n in field_names for n in ("x","y","z")):
            rospy.logwarn_throttle(2.0, "[association] PointCloud2 missing x/y/z.")
            return np.zeros((0, 3), dtype=np.float64)

        has_dop = "doppler" in field_names
        has_pow = "power" in field_names

        if has_dop and has_pow:
            use_fields = ("x","y","z","doppler","power")
            out = []
            for p in pc2.read_points(msg, field_names=use_fields, skip_nans=True):
                out.append([float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])])
            return np.array(out, dtype=np.float64) if out else np.zeros((0,5), dtype=np.float64)

        # doppler만 있는 경우
        if has_dop:
            use_fields = ("x","y","z","doppler")
            out = []
            for p in pc2.read_points(msg, field_names=use_fields, skip_nans=True):
                out.append([float(p[0]), float(p[1]), float(p[2]), float(p[3])])
            return np.array(out, dtype=np.float64) if out else np.zeros((0,4), dtype=np.float64)

        # power만 있는 경우
        if has_pow:
            use_fields = ("x","y","z","power")
            out = []
            for p in pc2.read_points(msg, field_names=use_fields, skip_nans=True):
                out.append([float(p[0]), float(p[1]), float(p[2]), float(p[3])])
            return np.array(out, dtype=np.float64) if out else np.zeros((0,4), dtype=np.float64)

        # xyz만
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
            rospy.logwarn_throttle(2.0, f"[association] extrinsic not found: {self.extrinsic_path}")
            return

        if mtime <= self._extr_mtime:
            return

        try:
            R, t = load_extrinsic_json(self.extrinsic_path)
            self._R, self._t = R, t
            self._extr_mtime = mtime
            rospy.loginfo(f"[association] reloaded extrinsic: {self.extrinsic_path}")
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"[association] extrinsic load failed: {e}")

    def _build_radar_stack(self) -> Tuple[Optional[rospy.Header], np.ndarray]:
        if len(self._rad_hist) == 0:
            return None, np.zeros((0, 3), dtype=np.float64)

        items = list(self._rad_hist)[-max(1, self.history_frames):]
        radar_header = items[-1][0]
        arrays = [it[1] for it in items if it[1] is not None and it[1].size > 0]
        if not arrays:
            return radar_header, np.zeros((0, 3), dtype=np.float64)

        cols = [a.shape[1] for a in arrays]
        if all(c == cols[0] for c in cols):
            return radar_header, np.vstack(arrays)

        # doppler 유무 섞이면 xyz만 사용
        return radar_header, np.vstack([a[:, :3] for a in arrays])

    def _on_detections(self, msg: DetectionArray):
        self._maybe_reload_extrinsic()

        if self._cam_K is None or self._img_w <= 0 or self._img_h <= 0:
            rospy.logwarn_throttle(2.0, "[association] waiting CameraInfo.")
            return

        radar_header, pts = self._build_radar_stack()
        if radar_header is None or pts.size == 0:
            out = AssociationArray()
            out.header = msg.header
            if radar_header is not None:
                out.radar_header = radar_header
            out.objects = []
            self.pub_assoc.publish(out)
            return

        # ---------------------------
        # 파라미터(런타임 튜닝용)
        # ---------------------------
        min_power = float(rospy.get_param("~association/min_power", 0.0))
        topk_by_power = int(rospy.get_param("~association/topk_by_power", 0))  # 0이면 사용 안함
        use_power_weighted = bool(rospy.get_param("~association/use_power_weighted_speed", True))
        use_abs_speed = bool(rospy.get_param("~association/use_abs_speed", True))

        # range gate
        xyz = pts[:, :3]
        rng = np.linalg.norm(xyz, axis=1)
        keep = rng >= max(0.0, self.min_range_m)
        pts = pts[keep]
        if pts.size == 0:
            out = AssociationArray()
            out.header = msg.header
            out.radar_header = radar_header
            out.objects = []
            self.pub_assoc.publish(out)
            return

        xyz = pts[:, :3]

        # 1회 투영 (uv + keep_idx)
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

        out = AssociationArray()
        out.header = msg.header
        out.radar_header = radar_header
        out.objects = []

        if uv.shape[0] == 0:
            # 투영점이 하나도 없으면 전부 NaN으로
            for det in msg.detections:
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
            return

        pts_valid = pts[keep_idx]  # (M, 3/4/5 ...)
        uu = uv[:, 0].astype(np.float64)
        vv = uv[:, 1].astype(np.float64)

        # (옵션) Marker
        marker_pack = MarkerArray() if self.publish_markers else None
        marker_id = 0
        radar_frame = radar_header.frame_id if radar_header.frame_id else self.radar_frame_default

        for det in msg.detections:
            x1 = float(det.bbox.xmin)
            y1 = float(det.bbox.ymin)
            x2 = float(det.bbox.xmax)
            y2 = float(det.bbox.ymax)

            inside = (uu >= x1) & (uu <= x2) & (vv >= y1) & (vv <= y2)
            sel = pts_valid[inside]

            assoc = Association()
            assoc.id = int(det.id)
            assoc.cls = int(det.cls)
            assoc.score = float(det.score)
            assoc.bbox = det.bbox

            if sel.shape[0] == 0:
                assoc.dist_m = float("nan")
                assoc.speed_mps = float("nan")
                assoc.speed_kph = float("nan")
                assoc.num_radar_points = 0
                out.objects.append(assoc)
                continue

            # ---------------------------
            # dist: bbox 내부 점들의 range median
            # ---------------------------
            sel_xyz = sel[:, :3]
            sel_rng = np.linalg.norm(sel_xyz, axis=1)
            dist_m = float(np.median(sel_rng))

            # ---------------------------
            # speed: doppler(+power) 기반
            # ---------------------------
            speed_mps = float("nan")

            # ✅ 권장: sel 컬럼이 [x,y,z,doppler,power] (shape>=5)
            # 만약 네가 [x,y,z,power,doppler]로 쌓고 있다면
            # dop = sel[:,4], pwr = sel[:,3] 로 바꿔야 함.
            if sel.shape[1] >= 5:
                dop = sel[:, 3]
                pwr = sel[:, 4]

                # 1) 유효 doppler + min_abs_doppler 필터
                m = np.isfinite(dop) & (np.abs(dop) >= max(0.0, self.min_abs_doppler))
                dop = dop[m]
                pwr = pwr[m] if pwr.shape[0] == m.shape[0] else pwr[:dop.shape[0]]

                # 2) power gate
                if dop.size > 0:
                    mp = np.isfinite(pwr) & (pwr >= min_power)
                    dop, pwr = dop[mp], pwr[mp]

                # 3) 상위 power K개만 사용(옵션)
                if topk_by_power > 0 and dop.size > topk_by_power:
                    idx = np.argsort(pwr)[-topk_by_power:]
                    dop, pwr = dop[idx], pwr[idx]

                # 4) speed 산출
                if dop.size > 0:
                    if use_power_weighted:
                        speed_mps = weighted_median(dop, pwr)
                    else:
                        speed_mps = float(np.median(dop))

                    if use_abs_speed and np.isfinite(speed_mps):
                        speed_mps = float(abs(speed_mps))

            # doppler만 있는(>=4) 경우 fallback
            elif sel.shape[1] >= 4:
                dop = sel[:, 3]
                dop = dop[np.isfinite(dop) & (np.abs(dop) >= max(0.0, self.min_abs_doppler))]
                if dop.size > 0:
                    speed_mps = float(np.median(dop))
                    if use_abs_speed and np.isfinite(speed_mps):
                        speed_mps = float(abs(speed_mps))

            speed_kph = speed_mps * 3.6 if np.isfinite(speed_mps) else float("nan")

            # 결과 채우기
            assoc.dist_m = dist_m
            assoc.speed_mps = float(speed_mps)
            assoc.speed_kph = float(speed_kph)
            assoc.num_radar_points = int(sel.shape[0])
            out.objects.append(assoc)

            # ---------------------------
            # (옵션) marker publish
            # ---------------------------
            if self.publish_markers and marker_pack is not None:
                draw_xyz = sel_xyz
                if draw_xyz.shape[0] > self.max_points_drawn:
                    idx = np.linspace(0, draw_xyz.shape[0] - 1, self.max_points_drawn).astype(np.int32)
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
                mk.color.r = 1.0
                mk.color.g = 1.0
                mk.color.b = 0.0
                mk.lifetime = rospy.Duration.from_sec(self.marker_lifetime)
                mk.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in draw_xyz]
                marker_pack.markers.append(mk)

        # publish
        self.pub_assoc.publish(out)
        if self.publish_markers and self.pub_markers is not None and marker_pack is not None:
            self.pub_markers.publish(marker_pack)

    def _nan_objects(self, dets):
        out = []
        for det in dets:
            assoc = Association()
            assoc.id = int(det.id)
            assoc.cls = int(det.cls)
            assoc.score = float(det.score)
            assoc.bbox = BoundingBox(xmin=det.bbox.xmin, ymin=det.bbox.ymin, xmax=det.bbox.xmax, ymax=det.bbox.ymax)
            assoc.dist_m = float("nan")
            assoc.speed_mps = float("nan")
            assoc.speed_kph = float("nan")
            assoc.num_radar_points = 0
            out.append(assoc)
        return out

    def _make_output(self, header, radar_header, objects):
        out = AssociationArray()
        out.header = header
        if radar_header is not None:
            out.radar_header = radar_header
        out.objects = objects
        return out


def main():
    rospy.init_node("association_node", anonymous=False)
    AssociationNode()
    rospy.spin()


if __name__ == "__main__":
    main()
