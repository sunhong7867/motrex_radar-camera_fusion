#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gui.py  —  Motrex Consumer GUI

Run:
    roslaunch autocal autocal.launch

Description:
    1. Startup Dialog: Choose AutoCal (Y/N)
    2. Y -> 5s countdown -> AutoCal x1 round -> View Mode
    3. N -> View Mode immediately
"""

import sys
import os
import json
import time
import traceback
from collections import deque
from typing import Optional

import cv2
import numpy as np

from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Qt
import message_filters
import rospy
import rospkg
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from cv_bridge import CvBridge

# ==============================================================================
# Path Setup
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
NODES_DIR = os.path.join(CURRENT_DIR, "nodes")
if NODES_DIR in sys.path:
    sys.path.remove(NODES_DIR)
sys.path.insert(0, NODES_DIR)

from perception_lib.speed_utils import (
    find_target_lane_from_bbox,
    get_bbox_reference_point,
    parse_radar_pointcloud,
    project_points,
    select_representative_point,
    update_lane_tracking,
    update_speed_estimate,
)
from autocal.msg import AssociationArray, DetectionArray
from perception_lib import calibration_utils
from perception_lib import lane_utils

os.environ["QT_LOGGING_RULES"] = "qt.gui.painting=false"

# ==============================================================================
# Settings
# ==============================================================================
REFRESH_RATE_MS     = 33       # ~30 FPS
MAX_REASONABLE_KMH  = 100.0
NOISE_MIN_SPEED_KMH = 3.0
AUTOCAL_DELAY_SEC   = 5        # Delay before calibration starts
AUTOCAL_ROUNDS      = 1        # Total calibration rounds

TOPIC_IMAGE       = "/camera/image_raw"
TOPIC_RADAR       = "/point_cloud"
TOPIC_CAMERA_INFO = "/camera/camera_info"
TOPIC_FINAL_OUT   = "/autocal/output"
TOPIC_ASSOCIATED  = "/autocal/associated"
TOPIC_TRACKS      = "/autocal/tracks"


# ==============================================================================
# Startup Dialog
# ==============================================================================
class StartupDialog(QtWidgets.QDialog):
    """Startup Options Dialog"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Motrex  -  Startup Options")
        self.setFixedSize(520, 220)
        self.setStyleSheet("background:#1a1a2e; color:white; font-size:15px;")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(18)
        layout.setContentsMargins(30, 30, 30, 30)

        title = QtWidgets.QLabel("📷  Run Radar-Camera Auto Calibration?")
        title.setWordWrap(True)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size:14px; font-weight:bold;")
        layout.addWidget(title)

        sub = QtWidgets.QLabel(
            "Y : AutoCal x1 rounds will start in 5 sec  ->  then view\n"
            "N : Skip calibration, start live view immediately"
        )
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet("font-size:13px; color:#aaaacc;")
        layout.addWidget(sub)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_yes = QtWidgets.QPushButton("Y  -  Run AutoCal")
        self.btn_yes.setStyleSheet(
            "background:#27ae60; color:white; font-size:14px; "
            "padding:10px 20px; border-radius:6px;"
        )
        self.btn_yes.clicked.connect(self.accept)

        self.btn_no = QtWidgets.QPushButton("N  -  Skip to View")
        self.btn_no.setStyleSheet(
            "background:#2980b9; color:white; font-size:14px; "
            "padding:10px 20px; border-radius:6px;"
        )
        self.btn_no.clicked.connect(self.reject)

        btn_row.addWidget(self.btn_yes)
        btn_row.addWidget(self.btn_no)
        layout.addLayout(btn_row)

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key_Y, Qt.Key_Return, Qt.Key_Enter):
            self.accept()
        elif key in (Qt.Key_N, Qt.Key_Escape):
            self.reject()
        else:
            super().keyPressEvent(event)


# ==============================================================================
# Image Viewer Widget
# ==============================================================================
class ImageCanvasViewer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = None
        self._scaled_rect = None
        self._image_size = None
        self.setStyleSheet("background:#000;")

    def update_image(self, cv_img: np.ndarray):
        h, w = cv_img.shape[:2]
        self._image_size = (w, h)
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(rgb.data, w, h, w * 3, QtGui.QImage.Format_RGB888)
        self.pixmap = QtGui.QPixmap.fromImage(qimg)
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        if self.pixmap:
            scaled = self.pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            x = (self.width()  - scaled.width())  // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
            self._scaled_rect = QtCore.QRect(x, y, scaled.width(), scaled.height())
        else:
            painter.setPen(QtCore.Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter, "Waiting for camera stream...")


# ==============================================================================
# Main Consumer GUI
# ==============================================================================
class ConsumerGUI(QtWidgets.QMainWindow):

    # ------------------------------------------------------------------
    # States
    # ------------------------------------------------------------------
    STATE_IDLE      = "idle"
    STATE_COUNTDOWN = "countdown"
    STATE_CALIBING  = "calibrating"
    STATE_VIEW      = "view"

    def __init__(self, do_autocal: bool):
        super().__init__()
        self.setWindowTitle("Motrex  |  Vehicle Speed Monitor")
        self.showMaximized()

        # ---- ROS Init ----
        rospy.init_node("consumer_gui_node", anonymous=True)
        self.bridge = CvBridge()

        pkg_path = rospkg.RosPack().get_path("autocal")
        self.extrinsic_path = os.path.join(pkg_path, "config", "extrinsic.json")
        self.lane_json_path  = os.path.join(pkg_path, "config", "lane_polys.json")

        # ---- ROS Params ----
        self.bbox_ref_mode   = rospy.get_param("~bbox_ref_mode",   "bottom")
        self.autocal_rounds  = int(rospy.get_param("~autocal_rounds", AUTOCAL_ROUNDS))

        # ---- Internal State ----
        self.do_autocal       = do_autocal
        self.state            = self.STATE_IDLE
        self.app_start_time   = time.time()
        self.cal_round        = 0
        self.cal_start_time   = 0.0
        self.cal_total_sec    = 100.0
        self.countdown_start  = 0.0
        self.extrinsic_proc   = None
        self.extrinsic_mtime  = None
        self._cal_log_buf     = []

        # ---- Image / Radar State ----
        self.latest_frame     = None
        self.cv_image         = None
        self.cam_K            = None
        self.Extr_R           = np.eye(3)
        self.Extr_t           = np.zeros((3, 1))
        self.lane_polys       = {}
        self.vis_objects      = []
        self.assoc_speed_by_id         = {}
        self.assoc_last_update_time    = 0.0
        self.last_update_time          = 0.0
        self.last_radar_pts   = None
        self.last_radar_dop   = None
        self.last_radar_ts    = 0.0
        self.radar_hold_threshold = 0.06

        # ---- Speed estimation ----
        self.lane_counters       = {n: 0 for n in ["IN1","IN2","IN3","OUT1","OUT2","OUT3"]}
        self.global_to_local_ids = {}
        self.vehicle_lane_paths  = {}
        self.track_confirmed_cnt = {}
        self.CONFIRM_FRAMES      = 20
        self.radar_track_hist    = {}
        self.track_hist_len      = 40
        self.lane_stable_cnt     = {}
        self.LANE_STABLE_FRAMES  = 15
        self.rep_pt_mem          = {}
        self.kf_vel              = {}
        self.kf_score            = {}
        self.vel_memory          = {}
        self.vel_hold_sec        = 0.6
        self.vel_decay_sec       = 0.8
        self.vel_decay_floor_kmh = 5.0
        self.vel_rate_limit_kmh_per_s = 12.0
        self.pt_alpha            = 0.15
        self.vel_gate_mps        = 1.5
        self._speed_frame_counter = 0
        self.speed_update_period  = 2
        self.prev_disp_vel = {}

        # Dummy speed params for compatibility
        self.spin_hold_sec  = type('o', (object,), {'value': lambda: 10.0})
        self.cmb_smoothing  = type('o', (object,), {'currentText': lambda: "EMA"})
        self.spin_ema_alpha = type('o', (object,), {'value': lambda: 0.35})

        # ---- Load ----
        self._load_extrinsic()
        self._load_lane_polys()

        # ---- UI ----
        self._init_ui()

        # ---- ROS Subscribers ----
        img_sub   = message_filters.Subscriber(TOPIC_IMAGE,  Image)
        radar_sub = message_filters.Subscriber(TOPIC_RADAR,  PointCloud2)
        self.ts   = message_filters.ApproximateTimeSynchronizer(
            [img_sub, radar_sub], 10, 0.1
        )
        self.ts.registerCallback(self._cb_sync)
        rospy.Subscriber(TOPIC_CAMERA_INFO, CameraInfo,       self._cb_info,             queue_size=1)
        rospy.Subscriber(TOPIC_FINAL_OUT,   AssociationArray, self._cb_final_result,      queue_size=1)
        rospy.Subscriber(TOPIC_ASSOCIATED,  AssociationArray, self._cb_association_result,queue_size=1)
        rospy.Subscriber(TOPIC_TRACKS,      DetectionArray,   self._cb_tracker_result,    queue_size=1)

        # ---- Timer ----
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update_loop)
        self.timer.start(REFRESH_RATE_MS)

        # ---- Start ----
        if self.do_autocal:
            self._start_countdown()
        else:
            self.state = self.STATE_VIEW

    # ------------------------------------------------------------------
    # UI Init
    # ------------------------------------------------------------------
    def _init_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.viewer = ImageCanvasViewer()
        layout.addWidget(self.viewer, stretch=1)

    # ------------------------------------------------------------------
    # AutoCal flow
    # ------------------------------------------------------------------
    def _start_countdown(self):
        self.state = self.STATE_COUNTDOWN
        self.countdown_start = time.time()

    def _start_calibration_round(self):
        """Run one AutoCal round via calibration_utils.run_autocalibration."""
        self.state = self.STATE_CALIBING
        self.cal_start_time = time.time()
        self._cal_log_buf = []

        if not hasattr(self, "_cal_candidates"):
            self._cal_candidates = []

        # Dummy attrs for calibration_utils compat
        if not hasattr(self, "txt_extrinsic_log"):
            self.txt_extrinsic_log = _DummyLog()
        if not hasattr(self, "pbar_extrinsic"):
            self.pbar_extrinsic = _DummyProgressBar()

        calibration_utils.run_autocalibration(self)

    def _on_extrinsic_stdout(self):
        if not self.extrinsic_proc:
            return
        data = bytes(self.extrinsic_proc.readAllStandardOutput()).decode("utf-8", errors="ignore")
        if data:
            self._cal_log_buf.append(data.rstrip())

    def _on_extrinsic_stderr(self):
        if not self.extrinsic_proc:
            return
        data = bytes(self.extrinsic_proc.readAllStandardError()).decode("utf-8", errors="ignore")
        if data:
            self._cal_log_buf.append(data.rstrip())

    def _on_extrinsic_finished(self):
        if hasattr(self, "pbar_extrinsic"):
            self.pbar_extrinsic.setVisible(False)

        # --- Read result of this round and score it ---
        result = self._read_extrinsic_file()
        if result is not None:
            R, t = result
            score = self._score_extrinsic(R, t)
            if not hasattr(self, "_cal_candidates"):
                self._cal_candidates = []
            self._cal_candidates.append((score, R, t))
            rospy.loginfo(
                f"[ConsumerGUI] AutoCal round {self.cal_round + 1}/{self.autocal_rounds} "
                f"done. inlier_score={score:.1f}"
            )
        else:
            rospy.logwarn(
                f"[ConsumerGUI] AutoCal round {self.cal_round + 1} produced no result."
            )

        self.cal_round += 1

        if self.cal_round < self.autocal_rounds:
            QtCore.QTimer.singleShot(500, self._start_calibration_round)
        else:
            # All rounds done
            self._apply_best_candidate()
            rospy.loginfo("[ConsumerGUI] All AutoCal rounds completed. Switching to VIEW mode.")
            self.state = self.STATE_VIEW

    def _read_extrinsic_file(self):
        try:
            if not os.path.exists(self.extrinsic_path):
                return None
            with open(self.extrinsic_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            R = np.array(data["R"], dtype=np.float64)
            t = np.array(data["t"], dtype=np.float64).reshape(3, 1)
            return R, t
        except Exception as e:
            rospy.logwarn(f"[ConsumerGUI] _read_extrinsic_file failed: {e}")
            return None

    def _score_extrinsic(self, R: np.ndarray, t: np.ndarray) -> float:
        det_err = abs(np.linalg.det(R) - 1.0)
        ortho_err = np.linalg.norm(R @ R.T - np.eye(3))
        t_mag = float(np.linalg.norm(t))
        t_penalty = max(0.0, t_mag - 50.0)

        inlier_score = 0.0
        for line in self._cal_log_buf:
            if "Inliers:" in line or "SUCCESS" in line:
                import re
                m = re.search(r"(\d+)\s*/\s*(\d+)", line)
                if m:
                    inlier_score = int(m.group(1)) / max(1, int(m.group(2))) * 100

        score = (
            inlier_score
            - det_err * 50.0
            - ortho_err * 30.0
            - t_penalty * 0.5
        )
        return float(score)

    def _apply_best_candidate(self):
        candidates = getattr(self, "_cal_candidates", [])
        if not candidates:
            rospy.logwarn("[ConsumerGUI] No valid calibration candidates.")
            self._load_extrinsic()
            return

        best_score, best_R, best_t = max(candidates, key=lambda x: x[0])
        rospy.loginfo(
            f"[ConsumerGUI] Selecting best calibration: score={best_score:.1f} "
            f"(from {len(candidates)} rounds)"
        )
        try:
            data = {"R": best_R.tolist(), "t": best_t.flatten().tolist()}
            with open(self.extrinsic_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            rospy.logwarn(f"[ConsumerGUI] Failed to write best extrinsic: {e}")
        self._load_extrinsic()
        self._cal_candidates = []

    # ------------------------------------------------------------------
    # ROS Callbacks
    # ------------------------------------------------------------------
    def _cb_sync(self, img_msg, radar_msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except Exception:
            return
        pts, dop = parse_radar_pointcloud(radar_msg)
        self.latest_frame = {
            "cv_image": cv_img,
            "radar_points": pts,
            "radar_doppler": dop,
        }

    def _cb_info(self, msg):
        if self.cam_K is None:
            self.cam_K = np.array(msg.K).reshape(3, 3)

    def _cb_final_result(self, msg):
        objects = []
        for obj in msg.objects:
            spd = obj.speed_kph
            if (not np.isfinite(spd)) and (time.time() - self.assoc_last_update_time <= 0.3):
                cached = self.assoc_speed_by_id.get(obj.id)
                if cached is not None and np.isfinite(cached):
                    spd = cached
            objects.append({
                "id":   obj.id,
                "bbox": [int(obj.bbox.xmin), int(obj.bbox.ymin),
                         int(obj.bbox.xmax), int(obj.bbox.ymax)],
                "vel":  spd,
            })
        self.vis_objects = objects
        self.last_update_time = time.time()

    def _cb_association_result(self, msg):
        for obj in msg.objects:
            if np.isfinite(obj.speed_kph):
                self.assoc_speed_by_id[obj.id] = obj.speed_kph
        self.assoc_last_update_time = time.time()

    def _cb_tracker_result(self, msg):
        if time.time() - self.last_update_time > 0.3:
            objects = []
            for det in msg.detections:
                objects.append({
                    "id":   det.id,
                    "bbox": [int(det.bbox.xmin), int(det.bbox.ymin),
                             int(det.bbox.xmax), int(det.bbox.ymax)],
                    "vel":  float("nan"),
                })
            self.vis_objects = objects

    # ------------------------------------------------------------------
    # Update Loop
    # ------------------------------------------------------------------
    def _update_loop(self):
        try:
            self._maybe_reload_extrinsic()
            now = time.time()
            elapsed_total = now - self.app_start_time

            # COUNTDOWN STATE
            if self.state == self.STATE_COUNTDOWN:
                
                # 1. 데이터가 들어왔는지 확인 (Warm-up Check)
                has_valid_data = False
                pts = None
                if self.latest_frame is not None:
                    pts = self.latest_frame.get("radar_points")

                # 조건: 포인트 10개 이상 AND 박스(vis_objects)가 1개 이상 감지됨
                if (pts is not None and len(pts) > 10 and 
                    self.latest_frame.get("cv_image") is not None and
                    len(self.vis_objects) > 0):  # <--- 핵심: 박스가 보여야 시작
                    has_valid_data = True

                # 2. 데이터가 불안정하면 대기
                if not has_valid_data:
                    self.countdown_start = now  # 타이머 계속 리셋
                    self._show_countdown_screen(AUTOCAL_DELAY_SEC, elapsed_total, waiting_data=True)
                    return

                # 3. 데이터가 안정적이면 카운트다운 진행
                remaining = AUTOCAL_DELAY_SEC - (now - self.countdown_start)
                
                if remaining <= 0:
                    self._start_calibration_round()
                else:
                    self._show_countdown_screen(int(np.ceil(remaining)), elapsed_total, waiting_data=False)
                    return

            # CALIBRATING STATE
            if self.state == self.STATE_CALIBING:
                self._show_calibrating_screen(elapsed_total)
                return

            # VIEW MODE
            if self.latest_frame is None:
                return

            self.cv_image = self.latest_frame.get("cv_image")
            if self.cv_image is None:
                return

            pts_raw = self.latest_frame.get("radar_points")
            dop_raw = self.latest_frame.get("radar_doppler")
            if pts_raw is not None and len(pts_raw) > 0:
                self.last_radar_pts = pts_raw
                self.last_radar_dop = dop_raw
                self.last_radar_ts  = now
            else:
                age = now - self.last_radar_ts
                if age < self.radar_hold_threshold:
                    pts_raw = self.last_radar_pts
                    dop_raw = self.last_radar_dop
                else:
                    pts_raw = self.last_radar_pts
                    dop_raw = self.last_radar_dop

            # Radar Project
            proj_uvs = proj_valid = None
            if (pts_raw is not None and self.cam_K is not None
                    and self.Extr_R is not None and self.Extr_t is not None):
                proj_uvs, proj_valid = project_points(
                    self.cam_K, self.Extr_R, self.Extr_t.flatten(), pts_raw
                )
            
            # --- [추가] 겹치는 BBox(가림 현상) 필터링 ---
            # 카메라에 더 가까운(ymax가 큰) 차량이 포인트를 선점하도록 분배
            num_pts = len(proj_uvs) if proj_uvs is not None else 0
            point_owner = np.full(num_pts, -1, dtype=int)

            if num_pts > 0 and proj_valid is not None:
                # 카메라에 가까운 차량부터 선점하되, 박스 전체가 아닌 '핵심 영역(Core)'만 사용
                sorted_objs = sorted(self.vis_objects, key=lambda o: o['bbox'][3], reverse=True)
                for s_obj in sorted_objs:
                    s_id = s_obj['id']
                    x1, y1, x2, y2 = s_obj['bbox']
                    
                    # 좌우 15%, 상단 20%를 제외한 안쪽 알맹이 영역만 자기 포인트로 강력하게 소유
                    w, h = x2 - x1, y2 - y1
                    cx1, cx2 = x1 + w * 0.15, x2 - w * 0.15
                    cy1, cy2 = y1 + h * 0.2, y2 
                    
                    u = proj_uvs[:, 0]
                    v = proj_uvs[:, 1]
                    in_box = (u >= cx1) & (u <= cx2) & (v >= cy1) & (v <= cy2)
                    unowned = (point_owner == -1)
                    point_owner[in_box & unowned & proj_valid] = s_id
            # ---------------------------------------------

            disp = self.cv_image.copy()

            # Lanes
            active_lane_polys = {
                name: poly
                for name, poly in self.lane_polys.items()
                if poly is not None and len(poly) > 2
            }
            has_lane_filter = bool(active_lane_polys)

            draw_items = []
            self._speed_frame_counter += 1

            # -- Objects --
            active_ids = set()
            for obj in self.vis_objects:
                g_id = obj["id"]
                x1, y1, x2, y2 = obj["bbox"]

                # Lane filter
                target_lane = None
                if has_lane_filter:
                    target_lane = find_target_lane_from_bbox(
                        (x1, y1, x2, y2), active_lane_polys
                    )
                    if target_lane is None:
                        self.track_confirmed_cnt.pop(g_id, None)
                        for d in [self.radar_track_hist, self.vehicle_lane_paths,
                                  self.global_to_local_ids, self.lane_stable_cnt]:
                            d.pop(g_id, None)
                        active_ids.discard(g_id)
                        continue

                # Confirmation
                self.track_confirmed_cnt[g_id] = (
                    self.track_confirmed_cnt.get(g_id, 0) + 1
                )
                if self.track_confirmed_cnt[g_id] < self.CONFIRM_FRAMES:
                    continue

                active_ids.add(g_id)

                # Stabilization
                if g_id in self.vehicle_lane_paths:
                    last_confirmed = self.vehicle_lane_paths[g_id].split("->")[-1]
                    if target_lane != last_confirmed:
                        cand, cnt = self.lane_stable_cnt.get(
                            g_id, (target_lane, 0)
                        )
                        if cand != target_lane:
                            self.lane_stable_cnt[g_id] = (target_lane, 1)
                        else:
                            self.lane_stable_cnt[g_id] = (cand, cnt + 1)

                        if self.lane_stable_cnt[g_id][1] < self.LANE_STABLE_FRAMES:
                            target_lane = last_confirmed
                        else:
                            self.lane_stable_cnt.pop(g_id, None)
                    else:
                        self.lane_stable_cnt.pop(g_id, None)
                else:
                    self.lane_stable_cnt.pop(g_id, None)

                curr_lane_str, lane_path, local_no, label = update_lane_tracking(
                    g_id, target_lane,
                    self.vehicle_lane_paths, self.lane_counters, self.global_to_local_ids,
                )

                target_pt = get_bbox_reference_point((x1, y1, x2, y2), self.bbox_ref_mode)
                meas_kmh = rep_pt = rep_vel_raw = None
                score = 0

                # 1. 내 박스로 할당된 레이더 포인트만 사용 (가림 현상 방지)
                if proj_uvs is not None and dop_raw is not None:
                    my_proj_valid = proj_valid & (point_owner == g_id)
                    
                    meas_kmh, rep_pt, rep_vel_raw, score, _ = select_representative_point(
                        proj_uvs, my_proj_valid, dop_raw,
                        (x1, y1, x2, y2), pts_raw,
                        self.vel_gate_mps, self.rep_pt_mem,
                        self.pt_alpha, now, g_id, NOISE_MIN_SPEED_KMH,
                        self.bbox_ref_mode,
                    )
                    
                    # 2. 스마트 급변 필터 (초기 진입 프리패스 + 관통 완벽 방어)
                    if meas_kmh is not None:
                        if not hasattr(self, 'rejected_cnt'): self.rejected_cnt = {}
                        if not hasattr(self, 'prev_disp_vel'): self.prev_disp_vel = {}
                        
                        current_vel = self.prev_disp_vel.get(g_id)
                        tracking_frames = self.track_confirmed_cnt.get(g_id, 0)
                        
                        if current_vel is not None:
                            speed_diff = abs(meas_kmh - current_vel)
                            
                            # [핵심 1] 화면에 2초(60프레임) 이상 존재하며 '완전 정지' 중인 차량에만 관통 방어 적용
                            if tracking_frames > 60 and current_vel <= 5.0 and meas_kmh >= 15.0:
                                self.rejected_cnt[g_id] = self.rejected_cnt.get(g_id, 0) + 1
                                # 1초(30프레임) 연속으로 잡히지 않으면 뒷차 관통 노이즈로 보고 차단
                                if self.rejected_cnt[g_id] < 30:
                                    meas_kmh = None
                                else:
                                    self.rejected_cnt[g_id] = 0
                                    
                            # [핵심 2] 주행 중 순간적으로 10km/h 이상 튀는 일반 레이더 노이즈 방어
                            # (단, 초기 진입 후 2초 이내인 차량은 한 번에 50km/h 등으로 뛸 수 있도록 예외 처리)
                            elif speed_diff > 10.0 and tracking_frames > 60:
                                self.rejected_cnt[g_id] = self.rejected_cnt.get(g_id, 0) + 1
                                if self.rejected_cnt[g_id] < 5:
                                    meas_kmh = None 
                                else:
                                    self.rejected_cnt[g_id] = 0
                                    
                            else:
                                # 막 진입한 차량의 초기 속도 점프나, 정상적인 가감속은 즉시 통과!
                                self.rejected_cnt[g_id] = 0

                # 3. 칼만 필터 및 속도 업데이트
                vel_out, score_out = update_speed_estimate(
                    g_id, meas_kmh, score, now,
                    self.vel_memory, self.kf_vel, self.kf_score,
                    hold_sec=self.vel_hold_sec,
                    decay_sec=self.vel_decay_sec,
                    decay_floor_kmh=self.vel_decay_floor_kmh,
                    rate_limit_kmh_per_s=self.vel_rate_limit_kmh_per_s,
                )

                # 4. 객체 속도 저장
                if vel_out is not None:
                    self.prev_disp_vel[g_id] = vel_out
                    obj["vel"] = float(int(round(vel_out)))
                else:
                    obj["vel"] = float("nan")

                draw_items.append({
                    "bbox":      (x1, y1, x2, y2),
                    "label":     label,
                    "vel":       obj["vel"],
                    "target_pt": target_pt,
                })

            # Draw
            for it in draw_items:
                x1, y1, x2, y2 = it["bbox"]
                cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 0, 255), 2)

                vel = it["vel"]
                speed_str = f"{int(vel)} km/h" if np.isfinite(vel) else "-- km/h"
                lines = [it["label"], speed_str]
                font = cv2.FONT_HERSHEY_SIMPLEX
                sc, th = 0.55, 1

                sizes = [cv2.getTextSize(ln, font, sc, th)[0] for ln in lines]
                pad, gap = 4, 12
                tw = max(s[0] for s in sizes) + pad * 2
                th_total = sum(s[1] for s in sizes) + gap * (len(lines) - 1) + pad * 2
                bx1, by1 = x1, y1 - th_total - 4
                if by1 < 0:
                    by1 = y1 + 4
                cv2.rectangle(disp, (bx1, by1), (bx1 + tw, by1 + th_total), (0, 0, 0), -1)

                cy = by1 + pad + sizes[0][1]
                for i, (line, sz) in enumerate(zip(lines, sizes)):
                    cv2.putText(disp, line, (bx1 + pad, cy), font, sc, (255, 255, 255), th)
                    cy += sz[1] + gap

            # Overlay
            self._draw_status_overlay(disp, elapsed_total)

            # Cleanup
            visible_ids = {o["id"] for o in self.vis_objects}
            self.track_confirmed_cnt = {k: v for k, v in self.track_confirmed_cnt.items() if k in visible_ids}
            self.radar_track_hist    = {k: v for k, v in self.radar_track_hist.items()    if k in active_ids}
            self.lane_stable_cnt     = {k: v for k, v in self.lane_stable_cnt.items()     if k in visible_ids}
            
            # [추가] 화면에서 사라진 차량의 오래된 캐시 삭제
            if hasattr(self, 'prev_disp_vel'):
                self.prev_disp_vel = {k: v for k, v in self.prev_disp_vel.items() if k in visible_ids}
            if hasattr(self, 'rejected_cnt'):
                self.rejected_cnt = {k: v for k, v in self.rejected_cnt.items() if k in visible_ids}

            self.viewer.update_image(disp)
            
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"[ConsumerGUI] update_loop error: {e}")

    # ------------------------------------------------------------------
    # Overlay Helpers
    # ------------------------------------------------------------------
    def _show_countdown_screen(self, remaining: int, elapsed_total: float, waiting_data: bool = False):
        if self.latest_frame is not None and self.latest_frame.get("cv_image") is not None:
            disp = self.latest_frame["cv_image"].copy()
        else:
            disp = np.zeros((720, 1280, 3), dtype=np.uint8)

        h, w = disp.shape[:2]
        overlay = disp.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, disp, 0.45, 0, disp)

        font = cv2.FONT_HERSHEY_SIMPLEX

        if waiting_data:
            # [수정] 대기 안내 문구 구체화
            cv2.putText(disp, "Waiting for STABLE TRACKS...", (w // 2 - 280, h // 2 - 50),
                        font, 1.2, (0, 165, 255), 3)
            # 포인트뿐만 아니라 '박스'가 잡혀야 함을 안내
            cv2.putText(disp, "Waiting for object detection...", (w // 2 - 300, h // 2 + 50),
                        font, 1.0, (200, 200, 200), 2)
        else:
            # 기존 카운트다운
            cv2.putText(disp, "Preparing AutoCal...", (w // 2 - 200, h // 2 - 80),
                        font, 1.4, (255, 255, 255), 3)
            cv2.putText(disp, f"Start in {remaining}s",
                        (w // 2 - 130, h // 2 + 10), font, 2.0, (0, 200, 255), 4)
        
        extra_msg = "Waiting for Stream" if waiting_data else f"{remaining}s until start"
        self._draw_status_overlay(disp, elapsed_total, mode="countdown", extra=extra_msg)
        self.viewer.update_image(disp)

    def _show_calibrating_screen(self, elapsed_total: float):
        """Calibration screen (ENGLISH ONLY) - Text Centered"""
        if self.latest_frame is not None and self.latest_frame.get("cv_image") is not None:
            disp = self.latest_frame["cv_image"].copy()
        else:
            disp = np.zeros((720, 1280, 3), dtype=np.uint8)

        h, w = disp.shape[:2]
        overlay = disp.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.40, disp, 0.60, 0, disp)

        font = cv2.FONT_HERSHEY_SIMPLEX
        round_str = f"Calibrating... ({self.cal_round + 1} / {self.autocal_rounds} Rounds)"
        
        # [수정] 텍스트의 실제 너비(tw)와 높이(th)를 계산해서 정중앙(Center) 좌표 구하기
        font_scale = 1.2
        thickness = 3
        (tw, th), _ = cv2.getTextSize(round_str, font, font_scale, thickness)
        
        text_x = (w - tw) // 2
        text_y = h // 2 - 40  # 진행 바 위쪽으로 살짝 띄움

        cv2.putText(disp, round_str, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

        # Progress bar
        round_elapsed = time.time() - self.cal_start_time
        frac = min(1.0, round_elapsed / max(1.0, self.cal_total_sec))
        
        # 바(Bar)도 중앙 정렬
        bar_w = 600
        bar_h = 28
        bar_x = (w - bar_w) // 2
        bar_y = h // 2 + 20
        
        cv2.rectangle(disp, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (60, 60, 60), -1)
        cv2.rectangle(disp, (bar_x, bar_y),
                      (bar_x + int(bar_w * frac), bar_y + bar_h), (0, 200, 100), -1)
        
        # 퍼센트 텍스트 중앙 정렬
        pct_str = f"{int(frac * 100)}%"
        (pw, ph), _ = cv2.getTextSize(pct_str, font, 0.65, 2)
        pct_x = bar_x + (bar_w - pw) // 2
        pct_y = bar_y + bar_h - 6
        
        cv2.putText(disp, pct_str, (pct_x, pct_y),
                    font, 0.65, (255, 255, 255), 2)

        # Last log line
        if self._cal_log_buf:
            last = self._cal_log_buf[-1][-80:]
            # 로그 텍스트도 중앙 정렬 (선택 사항)
            (lw, lh), _ = cv2.getTextSize(last, cv2.FONT_HERSHEY_PLAIN, 1.1, 1)
            log_x = (w - lw) // 2
            cv2.putText(disp, last, (log_x, h // 2 + 80),
                        cv2.FONT_HERSHEY_PLAIN, 1.1, (160, 220, 160), 1)

        self._draw_status_overlay(disp, elapsed_total, mode="calibrating",
                                  extra=f"Round {self.cal_round + 1}/{self.autocal_rounds}")
        self.viewer.update_image(disp)

    def _draw_status_overlay(self, disp: np.ndarray, elapsed_total: float,
                              mode: str = "view", extra: str = ""):
        """Top-Right Overlay (ENGLISH ONLY)"""
        h, w = disp.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        elapsed_str = _fmt_time(elapsed_total)
        # ENGLISH STRINGS
        lines = [f"Total Time: {elapsed_str}"]

        if mode == "countdown":
            lines.append("Waiting for AutoCal")
            if extra:
                lines.append(extra)
        elif mode == "calibrating":
            lines.append(f"AutoCal {self.cal_round + 1}/{self.autocal_rounds}")
            round_elapsed = time.time() - self.cal_start_time
            lines.append(f" Round Time: {_fmt_time(round_elapsed)}")
            if extra:
                lines.append(extra)
        else:
            if self.cal_round > 0:
                lines.append(f"- AutoCal Done ({self.cal_round} rounds)")
            lines.append("Monitoring Active")

        # Layout
        sc, th = 0.52, 1
        pad = 8
        sizes  = [cv2.getTextSize(ln, font, sc, th)[0] for ln in lines]
        box_w  = max(s[0] for s in sizes) + pad * 2
        line_h = max(s[1] for s in sizes) + 6
        box_h  = line_h * len(lines) + pad * 2

        x0 = w - box_w - 12
        y0 = 12
        sub = disp[y0: y0 + box_h, x0: x0 + box_w]
        if sub.shape[0] > 0 and sub.shape[1] > 0:
            black = np.zeros_like(sub)
            cv2.addWeighted(black, 0.65, sub, 0.35, 0, sub)
            disp[y0: y0 + box_h, x0: x0 + box_w] = sub

        ty = y0 + pad + sizes[0][1]
        for ln, sz in zip(lines, sizes):
            cv2.putText(disp, ln, (x0 + pad, ty), font, sc, (255, 255, 255), th)
            ty += line_h

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------
    def _load_extrinsic(self):
        calibration_utils.load_extrinsic(self)

    def _load_lane_polys(self):
        try:
            self.lane_polys = lane_utils.load_lane_polys(self.lane_json_path)
        except Exception:
            self.lane_polys = {}

    def _maybe_reload_extrinsic(self):
        if not os.path.exists(self.extrinsic_path):
            return
        try:
            mtime = os.path.getmtime(self.extrinsic_path)
        except OSError:
            return
        if self.extrinsic_mtime is None or mtime > self.extrinsic_mtime:
            self._load_extrinsic()


# ==============================================================================
# Dummy Widgets (Compatibility)
# ==============================================================================
class _DummyLog:
    def clear(self): pass
    def appendPlainText(self, _): pass
    def setVisible(self, _): pass

class _DummyProgressBar:
    def setVisible(self, _): pass
    def setRange(self, *_): pass
    def setValue(self, _): pass


# ==============================================================================
# Helpers
# ==============================================================================
def _fmt_time(sec: float) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s   = divmod(rem,  60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


# ==============================================================================
# Main
# ==============================================================================
def main():
    app = QtWidgets.QApplication(sys.argv)

    def _qt_msg_handler(mode, context, msg):
        if not msg.startswith("QPainter::end"):
            sys.stderr.write(msg + "\n")
    QtCore.qInstallMessageHandler(_qt_msg_handler)

    dlg = StartupDialog()
    result = dlg.exec()
    do_autocal = (result == QtWidgets.QDialog.Accepted)

    gui = ConsumerGUI(do_autocal=do_autocal)
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()