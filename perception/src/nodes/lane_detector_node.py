#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lane_detector_node.py

역할:
- lane_polys.json(ROI 폴리곤) 기반으로 bbox(검출/트랙)를 차선별로 판정/필터링
- 선택된 bbox만 다음 단계(association_node 등)로 전달

설정(선택지 A):
- lane_polys.json은 ROI 설정/편집 시 "자동 갱신"되는 파일
- roi_lanes.yaml은 lane_polys.json 경로 + 정책(anchor point, target lanes 등)만 관리

입력:
- ~tracks_in_topic (std_msgs/String) : JSON
  {"tracks":[{"id":1,"bbox":[x1,y1,x2,y2],"cls":2,"conf":0.7}, ...]}
  {"dets":[...]} 또는 {"objects":[...]} 또는 list([...]) 도 허용

- (옵션) ~image_topic (sensor_msgs/Image) : 디버그 오버레이용

출력:
- ~tracks_out_topic (std_msgs/String) : JSON
  {"tracks":[...]}  (원본 + lane 정보 추가 + 필터링 적용)

- (옵션) ~debug_image_topic (sensor_msgs/Image)
"""

import os
import re
import json
import time
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image

try:
    import yaml
except Exception:
    yaml = None

try:
    import rospkg
except Exception:
    rospkg = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    from cv_bridge import CvBridge
except Exception:
    CvBridge = None


# -----------------------------
# Helpers
# -----------------------------
def resolve_ros_path(expr: str) -> str:
    """
    "$(find pkg)/..." 를 실제 절대경로로 치환 + ~, env 확장
    """
    if not expr:
        return expr

    s = os.path.expanduser(os.path.expandvars(expr))

    if rospkg is None:
        return s

    pattern = r"\$\(\s*find\s+([A-Za-z0-9_]+)\s*\)"
    m = re.search(pattern, s)
    if not m:
        return s

    pkg = m.group(1)
    rp = rospkg.RosPack()
    pkg_path = rp.get_path(pkg)
    return re.sub(pattern, pkg_path, s)


def load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("pyyaml is required.")
    path = resolve_ros_path(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_lane_polys_json(path: str) -> Dict[str, np.ndarray]:
    """
    lane_polys.json format:
      {
        "IN1": [[u,v], [u,v], ...],
        "IN2": [...],
        "OUT1": [...],
        "OUT2": [...]
      }

    returns dict: lane_name -> contour (N,1,2) int32
    """
    path = resolve_ros_path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    polys: Dict[str, np.ndarray] = {}
    if not isinstance(raw, dict):
        return polys

    for k, pts in raw.items():
        if not isinstance(pts, list) or len(pts) < 3:
            continue
        arr = np.array(pts, dtype=np.float32).reshape(-1, 2)
        contour = arr.reshape(-1, 1, 2).astype(np.int32)
        polys[str(k)] = contour

    return polys


def parse_tracks_json(msg_data: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    returns: (key_name, items)
      key_name: "tracks" or "dets" or "objects" (출력에 사용할 대표 키)
      items: list of dicts with at least bbox
    """
    try:
        data = json.loads(msg_data)
    except Exception:
        return "tracks", []

    if isinstance(data, dict):
        if isinstance(data.get("tracks"), list):
            return "tracks", data["tracks"]
        if isinstance(data.get("dets"), list):
            return "tracks", data["dets"]
        if isinstance(data.get("objects"), list):
            return "tracks", data["objects"]
        # dict인데 키가 애매하면 빈 리스트
        return "tracks", []
    elif isinstance(data, list):
        return "tracks", data

    return "tracks", []


def bbox_anchor(bbox: List[int], mode: str) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    if mode == "bbox_center":
        return int((x1 + x2) / 2), int((y1 + y2) / 2)
    # default: bottom center
    return int((x1 + x2) / 2), int(y2)


def point_in_poly(u: int, v: int, contour: np.ndarray) -> bool:
    if cv2 is None:
        # cv2 없으면 대충 (안 쓰는 환경이면 설치 필요)
        return False
    # pointPolygonTest: inside>=0 (경계 포함)
    return cv2.pointPolygonTest(contour, (float(u), float(v)), False) >= 0


def scale_contour(contour: np.ndarray, sx: float, sy: float) -> np.ndarray:
    pts = contour.reshape(-1, 2).astype(np.float32)
    pts[:, 0] *= sx
    pts[:, 1] *= sy
    return pts.reshape(-1, 1, 2).astype(np.int32)


# -----------------------------
# Node
# -----------------------------
class LaneDetectorNode:
    def __init__(self):
        self.lock = threading.Lock()

        # Params
        self.roi_yaml = rospy.get_param("~roi_yaml", "$(find perception)/config/roi_lanes.yaml")

        self.tracks_in_topic = rospy.get_param("~tracks_in_topic", "/perception/tracks_in")
        self.tracks_out_topic = rospy.get_param("~tracks_out_topic", "/perception/tracks")

        self.publish_debug_image = bool(rospy.get_param("~publish_debug_image", True))
        self.image_topic = rospy.get_param("~image_topic", "/camera/image_raw")
        self.debug_image_topic = rospy.get_param("~debug_image_topic", "/perception/lane_roi_debug")

        # Reload control
        self.reload_interval = float(rospy.get_param("~reload_interval", 1.0))
        self._last_reload_check = 0.0
        self._lane_polys_mtime = 0.0

        # Loaded configs
        self.lane_polys_path = ""
        self.target_lanes: List[str] = []
        self.anchor_mode: str = "bbox_bottom_center"
        self.min_bbox_area: int = 1500

        # ROI image size (ROI가 만들어진 기준 해상도). 다르면 scale 적용.
        self.roi_img_w: Optional[int] = None
        self.roi_img_h: Optional[int] = None

        # Runtime image size (현재 들어오는 이미지 해상도)
        self.curr_img_w: Optional[int] = None
        self.curr_img_h: Optional[int] = None

        self.polys: Dict[str, np.ndarray] = {}

        # Debug image buffer
        self.bridge = CvBridge() if (CvBridge is not None) else None
        self.latest_img_bgr = None
        self.latest_img_stamp = None

        # Pub/Sub
        self.pub_tracks = rospy.Publisher(self.tracks_out_topic, String, queue_size=1)
        self.pub_dbg = rospy.Publisher(self.debug_image_topic, Image, queue_size=1) if self.publish_debug_image else None

        rospy.Subscriber(self.tracks_in_topic, String, self.cb_tracks, queue_size=1)
        if self.publish_debug_image:
            rospy.Subscriber(self.image_topic, Image, self.cb_image, queue_size=1)

        # Initial load
        self.load_roi_yaml()
        self.reload_lane_polys(force=True)

        rospy.loginfo("[lane_detector] ready. in=%s out=%s roi_yaml=%s",
                      self.tracks_in_topic, self.tracks_out_topic, resolve_ros_path(self.roi_yaml))

    # -------------------------
    # Config loading
    # -------------------------
    def load_roi_yaml(self):
        cfg = load_yaml(self.roi_yaml)

        roi_src = cfg.get("roi_source", {}) or {}
        path = roi_src.get("path", "")
        if not path:
            raise RuntimeError("roi_source.path is empty in roi_lanes.yaml")
        self.lane_polys_path = resolve_ros_path(path)

        pol = cfg.get("policy", {}) or {}
        self.anchor_mode = str(pol.get("anchor_point", "bbox_bottom_center"))
        self.min_bbox_area = int(pol.get("min_bbox_area", 1500))

        tl = pol.get("target_lanes", [])
        if tl is None:
            tl = []
        if isinstance(tl, list):
            self.target_lanes = [str(x) for x in tl]
        else:
            self.target_lanes = []

        # ROI 기준 이미지 해상도 (optional)
        img = cfg.get("image", {}) or {}
        w = img.get("width", None)
        h = img.get("height", None)
        self.roi_img_w = int(w) if isinstance(w, (int, float)) and int(w) > 0 else None
        self.roi_img_h = int(h) if isinstance(h, (int, float)) and int(h) > 0 else None

        rospy.loginfo("[lane_detector] roi_source=%s target_lanes=%s anchor=%s min_area=%d",
                      self.lane_polys_path, str(self.target_lanes), self.anchor_mode, self.min_bbox_area)

    def reload_lane_polys(self, force: bool = False):
        now = time.time()
        if (not force) and ((now - self._last_reload_check) < self.reload_interval):
            return
        self._last_reload_check = now

        try:
            mtime = os.path.getmtime(self.lane_polys_path)
        except Exception:
            return

        if (not force) and (mtime <= self._lane_polys_mtime):
            return

        try:
            polys = load_lane_polys_json(self.lane_polys_path)
            if not polys:
                rospy.logwarn("[lane_detector] lane_polys.json loaded but empty: %s", self.lane_polys_path)
                return

            # scale if needed (roi_img_w/h provided AND current image size known AND differs)
            with self.lock:
                curr_w = self.curr_img_w
                curr_h = self.curr_img_h

            if self.roi_img_w and self.roi_img_h and curr_w and curr_h:
                if (curr_w != self.roi_img_w) or (curr_h != self.roi_img_h):
                    sx = float(curr_w) / float(self.roi_img_w)
                    sy = float(curr_h) / float(self.roi_img_h)
                    for k in list(polys.keys()):
                        polys[k] = scale_contour(polys[k], sx, sy)
                    rospy.loginfo("[lane_detector] scaled ROI polys: (roi %dx%d) -> (img %dx%d)",
                                  self.roi_img_w, self.roi_img_h, curr_w, curr_h)

            with self.lock:
                self.polys = polys

            self._lane_polys_mtime = mtime
            rospy.loginfo("[lane_detector] lane_polys.json reloaded (mtime=%.0f): %s", mtime, self.lane_polys_path)
        except Exception as e:
            rospy.logwarn("[lane_detector] lane_polys reload failed: %s", str(e))

    # -------------------------
    # Callbacks
    # -------------------------
    def cb_image(self, msg: Image):
        if self.bridge is None or cv2 is None:
            return
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return
        with self.lock:
            self.latest_img_bgr = img
            self.latest_img_stamp = msg.header.stamp.to_sec()
            self.curr_img_h, self.curr_img_w = img.shape[:2]

    def cb_tracks(self, msg: String):
        # reload ROI if updated
        self.reload_lane_polys(force=False)

        key, items = parse_tracks_json(msg.data)
        if not items:
            self.pub_tracks.publish(String(data=json.dumps({key: []})))
            return

        with self.lock:
            polys = dict(self.polys)
            dbg_img = None
            if self.publish_debug_image and self.latest_img_bgr is not None and cv2 is not None:
                dbg_img = self.latest_img_bgr.copy()
                img_w = self.curr_img_w
                img_h = self.curr_img_h
            else:
                img_w = None
                img_h = None

        if not polys:
            rospy.logwarn_throttle(1.0, "[lane_detector] no ROI polys loaded yet.")
            # ROI 없으면 그대로 통과(필터 off)
            out = {"tracks": self._normalize_tracks(items)}
            self.pub_tracks.publish(String(data=json.dumps(out)))
            return

        # draw ROI polygons
        if dbg_img is not None and cv2 is not None:
            for ln, contour in polys.items():
                cv2.polylines(dbg_img, [contour], isClosed=True, color=(255, 255, 255), thickness=2)
                # label near first point
                p0 = contour.reshape(-1, 2)[0]
                cv2.putText(dbg_img, ln, (int(p0[0]), int(p0[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        normed = self._normalize_tracks(items)

        filtered: List[Dict[str, Any]] = []
        for tr in normed:
            bbox = tr["bbox"]
            x1, y1, x2, y2 = bbox
            area = max(0, (x2 - x1)) * max(0, (y2 - y1))
            if area < self.min_bbox_area:
                tr["lane"] = None
                tr["lane_hit"] = False
                if dbg_img is not None and cv2 is not None:
                    cv2.rectangle(dbg_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                continue

            u, v = bbox_anchor(bbox, self.anchor_mode)

            # 현재 이미지 경계 클램프(디버그 안정)
            if img_w and img_h:
                u = max(0, min(img_w - 1, u))
                v = max(0, min(img_h - 1, v))

            hit_lane = None

            # target_lanes 지정 시 그 안에서만 판정
            lanes_to_check = self.target_lanes if self.target_lanes else list(polys.keys())

            for ln in lanes_to_check:
                if ln not in polys:
                    continue
                if point_in_poly(u, v, polys[ln]):
                    hit_lane = ln
                    break

            tr["lane"] = hit_lane
            tr["lane_hit"] = (hit_lane is not None)

            if hit_lane is not None:
                filtered.append(tr)

            # debug draw
            if dbg_img is not None and cv2 is not None:
                color = (0, 255, 0) if hit_lane is not None else (0, 0, 255)
                cv2.rectangle(dbg_img, (x1, y1), (x2, y2), color, 2)
                cv2.circle(dbg_img, (u, v), 4, (255, 0, 0), -1)
                label = f"id:{tr.get('id',-1)} {hit_lane if hit_lane else 'NONE'}"
                cv2.putText(dbg_img, label, (x1, max(20, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out = {"tracks": filtered}
        self.pub_tracks.publish(String(data=json.dumps(out)))

        # publish debug image
        if self.publish_debug_image and dbg_img is not None and self.pub_dbg is not None and self.bridge is not None:
            try:
                out_img = self.bridge.cv2_to_imgmsg(dbg_img, encoding="bgr8")
                out_img.header.stamp = rospy.Time.now()
                out_img.header.frame_id = "camera"
                self.pub_dbg.publish(out_img)
            except Exception:
                pass

    # -------------------------
    # track normalize
    # -------------------------
    def _normalize_tracks(self, items: List[Any]) -> List[Dict[str, Any]]:
        """
        items를 통일된 dict 형태로 변환:
          {"id":int, "bbox":[x1,y1,x2,y2], "cls":int, "conf":float}
        """
        out: List[Dict[str, Any]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            bbox = it.get("bbox", None)
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            try:
                x1, y1, x2, y2 = [int(v) for v in bbox]
            except Exception:
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            out.append({
                "id": int(it.get("id", -1)),
                "bbox": [x1, y1, x2, y2],
                "cls": int(it.get("cls", -1)),
                "conf": float(it.get("conf", 0.0)),
            })
        return out


def main():
    rospy.init_node("lane_detector_node", anonymous=False)
    _ = LaneDetectorNode()
    rospy.spin()


if __name__ == "__main__":
    main()
