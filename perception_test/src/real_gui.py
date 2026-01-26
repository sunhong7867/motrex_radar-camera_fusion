#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
real_gui.py (Data-Driven Axis Fix)

[데이터 분석 결과 반영]
- log.txt 분석 결과: X축(-42~24m)이 좌우, Y축(11~97m)이 전방임이 확인됨.
- 수정 사항:
  1. BEV 그래프 매핑을 scatter(X, Y)로 설정하여 좌우/전방을 올바르게 표시.
  2. 축 범위를 데이터 분포에 맞춰 좌우(-50~50), 전방(0~100)으로 최적화.
  3. message_filters로 영상-레이더 시간 동기화 적용.

[추가 수정(이번 이슈 해결)]
- Pause 시 QTimer만 멈추는 대신, 콜백에서 들어오는 동기 프레임을 pause_queue에 쌓고
  Resume 시 큐를 먼저 소모하며 재생 -> “화면만 멈춤”이 아니라 “데이터도 멈춘 것처럼” 동작
- 그래프가 리사이즈 때만 보이거나 반영이 불안정하던 문제:
  * cluster_points 함수 스코프 문제 제거(클래스 내부 staticmethod로 구현)
  * draw()/flush_events 대신 draw_idle + tight_layout로 안정화
- 카메라/레이더 그래프 갱신 tick 일치:
  * timer_graph 제거, update_loop 한 번의 tick에서 화면+그래프를 같은 프레임으로 업데이트
"""

import sys
import os
import json
import time
import subprocess
import traceback
import numpy as np
import cv2
from collections import deque

# ==============================================================================
# [PARAMETER SETTINGS] 사용자 설정
# ==============================================================================

# 1. 시스템 및 타이밍
REFRESH_RATE_MS = 33           # GUI 갱신 주기 (33ms = 30FPS)

# 2. 레이더 필터링
SPEED_THRESHOLD_RED = -0.5       # 접근 기준 (이보다 낮으면 빨강/접근)
SPEED_THRESHOLD_BLUE = 0.5       # 이탈 기준 (이보다 높으면 파랑/이탈)
NOISE_MIN_SPEED_KMH = 0.2      # 최소 속도 필터 (정지 물체 노이즈 제거용)
MOVING_CLUSTER_MIN_SPEED_KMH = 0  # 클러스터 박스 표시용 최소 이동 속도
CLUSTER_MAX_DIST_M = 2.0       # 클러스터 이웃 거리 (m)
CLUSTER_MIN_POINTS = 4         # 클러스터 최소 포인트 수

# 3. 시각화 크기
OVERLAY_POINT_RADIUS = 6       # 영상 위 점 크기
GRAPH_POINT_SIZE = 20          # 그래프 점 크기
BBOX_LINE_THICKNESS = 2        # 박스 두께

# 4. 그래프 축 범위 (단위: m) - [데이터 분석 기반 설정]
BEV_LATERAL_LIMIT = (-20, 20)  # 그래프 가로축 (Screen X) 범위
BEV_FORWARD_LIMIT = (0, 120)   # 그래프 세로축 (Screen Y) 범위

# 4. 토글 킬 차선
START_ACTIVE_LANES = ["IN1", "IN2", "IN3"]

# 5. 토픽 이름
TOPIC_IMAGE = "/camera/image_raw"
TOPIC_RADAR = "/point_cloud"
TOPIC_CAMERA_INFO = "/camera/camera_info"
TOPIC_FINAL_OUT = "/perception_test/output"
TOPIC_TRACKS = "/perception_test/tracks"
TOPIC_ASSOCIATED = "/perception_test/associated"

# ==============================================================================
# Setup
# ==============================================================================
# Note: Matplotlib-based radar graphs have been removed.
# We no longer import or use matplotlib in this GUI since the
# radar BEV graph and pause functionality have been dropped.

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

try:
    from PySide6 import QtWidgets, QtGui, QtCore
    from PySide6.QtCore import Qt
    import rospy
    import rospkg
    import message_filters  # [동기화 핵심]
    from sensor_msgs.msg import Image, PointCloud2, CameraInfo
    from cv_bridge import CvBridge
    import sensor_msgs.point_cloud2 as pc2

    from perception_test.msg import AssociationArray, DetectionArray
    from perception_lib import perception_utils
    from perception_lib import lane_utils
except ImportError as e:
    print(f"\n[GUI Error] 라이브러리 로드 실패: {e}")
    sys.exit(1)


# ==============================================================================
# UI Classes
# ==============================================================================


# -----------------------------------------------------------------------------
# The RadarGraphCanvas class and related clustering code have been removed.  The
# GUI no longer displays a separate radar point cloud graph.  All overlay and
# visualization is handled directly within the image view in update_loop().
    
# The old CalibrationDialog (which launched separate calibration nodes and drew a
# BEV graph) has been removed. Calibration is now triggered directly from the
# main GUI via the ManualCalibWindow for extrinsic calibration, and the
# calibration nodes can be run independently if needed.


class LaneCanvas(QtWidgets.QWidget):
    def __init__(self, bg_img, lane_polys, parent=None):
        super().__init__(parent)
        self.bg_img = bg_img.copy()
        self.h, self.w = self.bg_img.shape[:2]
        self.setFixedSize(self.w, self.h)
        self._lane_polys = lane_polys
        self._current_lane_name = "IN1"
        self._editing_pts = []
        self.pen_boundary = QtGui.QPen(QtGui.QColor(0, 255, 0), 2)
        self.brush_fill = QtGui.QBrush(QtGui.QColor(0, 255, 0, 60))
        self.pen_editing = QtGui.QPen(QtGui.QColor(50, 180, 255), 2)
        self.font_label = QtGui.QFont("Arial", 12, QtGui.QFont.Bold)

    def set_current_lane(self, name):
        self._current_lane_name = name
        self._editing_pts = []
        self.update()

    def undo_last_point(self):
        if self._editing_pts:
            self._editing_pts.pop()
            self.update()

    def finish_current_polygon(self):
        if len(self._editing_pts) < 3:
            return
        arr = np.array(self._editing_pts, dtype=np.int32)
        self._lane_polys[self._current_lane_name] = arr
        self._editing_pts = []
        self.update()

    def clear_current_lane(self):
        if self._current_lane_name in self._lane_polys:
            self._lane_polys[self._current_lane_name] = np.empty((0, 2), dtype=np.int32)
        self._editing_pts = []
        self.update()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            x, y = event.position().x(), event.position().y()
            self._editing_pts.append([x, y])
            self.update()
        elif event.button() == QtCore.Qt.RightButton:
            self.undo_last_point()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        qimg = QtGui.QImage(self.bg_img.data, self.w, self.h, 3 * self.w, QtGui.QImage.Format_BGR888)
        p.drawImage(0, 0, qimg)
        for name, poly in self._lane_polys.items():
            if poly is None or len(poly) == 0:
                continue
            if name == self._current_lane_name:
                p.setPen(QtGui.QPen(QtGui.QColor(255, 50, 50), 3))
            else:
                p.setPen(self.pen_boundary)
            p.setBrush(self.brush_fill)
            path = QtGui.QPainterPath()
            pts = [QtCore.QPoint(int(x), int(y)) for x, y in poly]
            if not pts:
                continue
            path.moveTo(pts[0])
            for pt in pts[1:]:
                path.lineTo(pt)
            path.closeSubpath()
            p.drawPath(path)
            if len(poly) > 0:
                cx, cy = int(np.mean(poly[:, 0])), int(np.mean(poly[:, 1]))
                p.setPen(QtGui.QColor(255, 255, 255))
                p.setFont(self.font_label)
                p.drawText(cx, cy, name)
        if self._editing_pts:
            p.setPen(self.pen_editing)
            pts = [QtCore.QPoint(int(x), int(y)) for x, y in self._editing_pts]
            for i in range(len(pts) - 1):
                p.drawLine(pts[i], pts[i + 1])
            for pt in pts:
                p.drawEllipse(pt, 3, 3)


class LaneEditorDialog(QtWidgets.QDialog):
    def __init__(self, bg_img, current_polys, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Lane Polygon Editor")
        img_h, img_w = bg_img.shape[:2]
        self.resize(img_w + 50, img_h + 150)
        self.polys = {}
        for k, v in current_polys.items():
            self.polys[k] = v.copy() if v is not None else np.empty((0, 2), dtype=np.int32)
        layout = QtWidgets.QVBoxLayout(self)
        h_ctrl = QtWidgets.QHBoxLayout()
        self.combo = QtWidgets.QComboBox()
        self.lane_list = ["IN1", "IN2", "IN3", "OUT1", "OUT2", "OUT3"]
        self.combo.addItems(self.lane_list)
        self.combo.currentTextChanged.connect(lambda t: self.canvas.set_current_lane(t))
        btn_finish = QtWidgets.QPushButton("영역 닫기 (Close)")
        btn_finish.clicked.connect(lambda: self.canvas.finish_current_polygon())
        btn_undo = QtWidgets.QPushButton("점 취소 (Undo)")
        btn_undo.clicked.connect(lambda: self.canvas.undo_last_point())
        btn_clear = QtWidgets.QPushButton("전체 지우기")
        btn_clear.clicked.connect(lambda: self.canvas.clear_current_lane())
        h_ctrl.addWidget(QtWidgets.QLabel("Select Lane:"))
        h_ctrl.addWidget(self.combo)
        h_ctrl.addWidget(btn_finish)
        h_ctrl.addWidget(btn_undo)
        h_ctrl.addWidget(btn_clear)
        layout.addLayout(h_ctrl)
        scroll = QtWidgets.QScrollArea()
        self.canvas = LaneCanvas(bg_img, self.polys)
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        scroll.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(scroll)
        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)
        self.canvas.set_current_lane(self.combo.currentText())

    def get_polys(self):
        return self.polys


class ImageCanvasViewer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = None
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #101010;")

    def update_image(self, cv_img):
        if cv_img is None:
            return
        h, w, ch = cv_img.shape
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        q_img = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        self.pixmap = QtGui.QPixmap.fromImage(q_img)
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        if self.pixmap:
            scaled = self.pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        else:
            painter.setPen(QtCore.Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter, "Waiting for Camera Stream...")


# ==============================================================================
# 메인 GUI Class (동기화 적용)
# ==============================================================================
class RealWorldGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Motrex-SKKU Sensor Fusion GUI")
        self.resize(1900, 2000)

        rospy.init_node('real_gui_node', anonymous=True)
        self.bridge = CvBridge()
        pkg_path = rospkg.RosPack().get_path("perception_test")
        self.nodes_dir = os.path.join(CURRENT_DIR, "nodes")
        # extrinsic and lane polygon files are stored in the config folder
        self.extrinsic_path = os.path.join(pkg_path, "config", "extrinsic.json")
        self.lane_json_path = os.path.join(pkg_path, "config", "lane_polys.json")

        self.cv_image = None
        self.radar_points = None
        self.cam_K = None
        self.lane_polys = {}
        self.Extr_R = np.eye(3)
        self.Extr_t = np.zeros((3, 1))
        self.vis_objects = []
        self.assoc_speed_by_id = {}
        self.assoc_last_update_time = 0.0
        self.lane_counters = {name: 0 for name in ["IN1", "IN2", "IN3", "OUT1", "OUT2", "OUT3"]}
        self.global_to_local_ids = {}
        self.extrinsic_mtime = None
        self.extrinsic_last_loaded = None

        # Buffer for the latest synchronized frame
        self.latest_frame = None

        self.load_extrinsic()
        self.load_lane_polys()
        self.init_ui()

        # -------------------------------------------------------------
        # Time Synchronizer (타이밍 문제 해결)
        # -------------------------------------------------------------
        image_sub = message_filters.Subscriber(TOPIC_IMAGE, Image)
        radar_sub = message_filters.Subscriber(TOPIC_RADAR, PointCloud2)

        # slop=0.1s (100ms) 이내의 오차를 가진 데이터끼리 묶어서 콜백 실행
        self.ts = message_filters.ApproximateTimeSynchronizer([image_sub, radar_sub], 10, 0.1)
        self.ts.registerCallback(self.cb_sync)

        rospy.Subscriber(TOPIC_CAMERA_INFO, CameraInfo, self.cb_info, queue_size=1)
        rospy.Subscriber(TOPIC_FINAL_OUT, AssociationArray, self.cb_final_result, queue_size=1)
        rospy.Subscriber(TOPIC_ASSOCIATED, AssociationArray, self.cb_association_result, queue_size=1)
        rospy.Subscriber(TOPIC_TRACKS, DetectionArray, self.cb_tracker_result, queue_size=1)

        # 타이머는 하나만 사용(카메라/레이더 그래프 갱신을 동일 tick으로 맞춤)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(REFRESH_RATE_MS)

        self.last_update_time = time.time()

    def init_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        self.viewer = ImageCanvasViewer()
        left_layout.addWidget(self.viewer, stretch=1)
        layout.addWidget(left_widget, stretch=4)

        panel = QtWidgets.QWidget()
        panel.setFixedWidth(280)
        vbox = QtWidgets.QVBoxLayout(panel)
        vbox.setAlignment(Qt.AlignTop)

        gb_calib = QtWidgets.QGroupBox("1. Calibration")
        v_c = QtWidgets.QVBoxLayout()
        
        # 1. 내부 위젯들을 무조건 위로 밀착시킴 (버튼 사이 벌어짐 방지)
        v_c.setAlignment(QtCore.Qt.AlignTop) 
        v_c.setSpacing(5)

        btn_intri = QtWidgets.QPushButton("Run Intrinsic (Offline images)")
        btn_intri.clicked.connect(self.run_intrinsic_calibration)
        btn_calib = QtWidgets.QPushButton("Run Extrinsic (Manual)")
        btn_calib.clicked.connect(self.run_calibration)
        btn_reload = QtWidgets.QPushButton("Reload JSON")
        btn_reload.clicked.connect(self.load_extrinsic)
        
        self.pbar_intrinsic = QtWidgets.QProgressBar()
        self.pbar_intrinsic.setRange(0, 0)
        self.pbar_intrinsic.setVisible(False)

        self.txt_intrinsic_log = QtWidgets.QPlainTextEdit()
        self.txt_intrinsic_log.setReadOnly(True)
        self.txt_intrinsic_log.setMaximumHeight(160)

        v_c.addWidget(btn_intri)
        v_c.addWidget(btn_calib)
        v_c.addWidget(btn_reload)
        v_c.addWidget(self.pbar_intrinsic)
        v_c.addWidget(self.txt_intrinsic_log)

        gb_calib.setLayout(v_c)
        
        # 2. 그룹박스가 세로로 늘어나지 못하게 제한 (그룹박스 하단 빈공간 제거)
        gb_calib.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        
        vbox.addWidget(gb_calib)

        gb_lane = QtWidgets.QGroupBox("2. Lane Editor")
        v_l = QtWidgets.QVBoxLayout()
        btn_edit = QtWidgets.QPushButton("🖌️ Open Editor (Pop-up)")
        btn_edit.setStyleSheet("background-color: #FFD700; color: black; font-weight: bold; padding: 10px;")
        btn_edit.clicked.connect(self.open_lane_editor)
        v_l.addWidget(btn_edit)
        gb_lane.setLayout(v_l)
        vbox.addWidget(gb_lane)

        gb_vis = QtWidgets.QGroupBox("3. View Options")
        v_vis = QtWidgets.QVBoxLayout()
        self.chk_show_poly = QtWidgets.QCheckBox("Show Lane Polygons (ROI)")
        self.chk_show_poly.setChecked(False)
        v_vis.addWidget(self.chk_show_poly)
        v_vis.addWidget(QtWidgets.QLabel("--- Filter Lane (Box & Poly) ---"))

        self.chk_lanes = {}
        display_order = ["IN1", "OUT1", "IN2", "OUT2", "IN3", "OUT3"]
        g_vis = QtWidgets.QGridLayout()

        for i, name in enumerate(display_order):
            chk = QtWidgets.QCheckBox(name)
            is_active = name in START_ACTIVE_LANES
            chk.setChecked(is_active)

            self.chk_lanes[name] = chk
            g_vis.addWidget(chk, i // 2, i % 2)

        v_vis.addLayout(g_vis)

        btn_reset_id = QtWidgets.QPushButton("Reset Local IDs")
        btn_reset_id.clicked.connect(self.reset_ids)
        v_vis.addWidget(btn_reset_id)
        gb_vis.setLayout(v_vis)
        vbox.addWidget(gb_vis)

        vbox.addStretch()
        self.lbl_log = QtWidgets.QLabel("System Ready")
        vbox.addWidget(self.lbl_log)
        layout.addWidget(panel, stretch=1)

    def reset_ids(self):
        self.lane_counters = {name: 0 for name in self.lane_counters.keys()}
        self.global_to_local_ids = {}
        self.lbl_log.setText("Local IDs Reset.")

    # Pause/resume functionality has been removed; no toggle_pause method needed

    # ------------------ Callback (Synchronized) ------------------
    def cb_sync(self, img_msg, radar_msg):
        # 1) 이미지
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except:
            return

        # 2) 레이더
        try:
            field_names = [f.name for f in radar_msg.fields]
            if not {"x", "y", "z"}.issubset(field_names):
                return

            doppler_candidates = ("doppler", "doppler_mps", "velocity", "vel", "radial_velocity")
            power_candidates = ("power", "intensity", "rcs")

            doppler_field = next((f for f in doppler_candidates if f in field_names), None)
            power_field = next((f for f in power_candidates if f in field_names), None)

            target_fields = ["x", "y", "z"]
            if power_field:
                target_fields.append(power_field)
            if doppler_field:
                target_fields.append(doppler_field)

            g = pc2.read_points(radar_msg, field_names=tuple(target_fields), skip_nans=True)
            self.radar_points_all = np.array(list(g))

            if self.radar_points_all.shape[0] > 0:
                radar_points = self.radar_points_all[:, :3]
                col_idx = 3
                radar_power = None
                radar_doppler = None

                if power_field:
                    radar_power = self.radar_points_all[:, col_idx]
                    col_idx += 1
                if doppler_field:
                    radar_doppler = self.radar_points_all[:, col_idx]
            else:
                radar_points, radar_power, radar_doppler = None, None, None
        except:
            return

        frame = {
            "stamp": img_msg.header.stamp.to_sec() if img_msg.header.stamp else time.time(),
            "cv_image": cv_image,
            "radar_points": radar_points,
            "radar_power": radar_power,
            "radar_doppler": radar_doppler,
        }

        # Always store the latest synchronized frame (pause functionality removed)
        self.latest_frame = frame

    def cb_info(self, msg):
        if self.cam_K is None:
            self.cam_K = np.array(msg.K).reshape(3, 3)

    def cb_final_result(self, msg):
        objects = []
        for obj in msg.objects:
            speed_kph = obj.speed_kph
            if (not np.isfinite(speed_kph)) and (time.time() - self.assoc_last_update_time <= 0.3):
                cached_speed = self.assoc_speed_by_id.get(obj.id)
                if cached_speed is not None and np.isfinite(cached_speed):
                    speed_kph = cached_speed
            objects.append({
                'id': obj.id,
                'bbox': [int(obj.bbox.xmin), int(obj.bbox.ymin), int(obj.bbox.xmax), int(obj.bbox.ymax)],
                'vel': speed_kph
            })
        self.vis_objects = objects
        self.last_update_time = time.time()

    def cb_association_result(self, msg):
        assoc_speeds = {}
        for obj in msg.objects:
            if np.isfinite(obj.speed_kph):
                assoc_speeds[obj.id] = obj.speed_kph
        self.assoc_speed_by_id = assoc_speeds
        self.assoc_last_update_time = time.time()

    def cb_tracker_result(self, msg):
        if time.time() - self.last_update_time > 0.3:
            objects = []
            for det in msg.detections:
                objects.append({
                    'id': det.id,
                    'bbox': [int(det.bbox.xmin), int(det.bbox.ymin), int(det.bbox.xmax), int(det.bbox.ymax)],
                    'vel': float('nan')
                })
            self.vis_objects = objects

    def get_text_position(self, pts):
        sorted_indices = np.argsort(pts[:, 1])[::-1]
        bottom_two = pts[sorted_indices[:2]]
        left_idx = np.argmin(bottom_two[:, 0])
        return tuple(bottom_two[left_idx])

    # ------------------ Update Loops ------------------
    def update_loop(self):
        self._maybe_reload_extrinsic()
        # Always use the latest frame. Pause/resume functionality removed.
        frame = self.latest_frame

        if frame is None:
            return

        # 다른 기능(차선 편집 등)에서 현재 프레임 참조가 필요할 수 있어 저장
        self.cv_image = frame["cv_image"]
        self.radar_points = frame["radar_points"]

        disp = frame["cv_image"].copy()
        radar_points = frame["radar_points"]
        radar_power = frame["radar_power"]
        radar_doppler = frame["radar_doppler"]

        # 1. Objects
        active_lane_polys = {}
        for name, chk in self.chk_lanes.items():
            if not chk.isChecked():
                continue
            poly = self.lane_polys.get(name)
            if poly is not None and len(poly) > 2:
                active_lane_polys[name] = poly
        has_lane_filter = bool(active_lane_polys)

        front_lanes = {"IN1", "IN2", "IN3"}
        front_by_lane = {}
        if has_lane_filter:
            for obj in self.vis_objects:
                g_id = obj['id']
                x1, y1, x2, y2 = obj['bbox']
                cx, cy = (x1 + x2) // 2, y2

                target_lane = None
                for name, poly in active_lane_polys.items():
                    if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                        target_lane = name
                        break

                if target_lane and target_lane in front_lanes:
                    current = front_by_lane.get(target_lane)
                    if current is None or y2 > current["y2"]:
                        front_by_lane[target_lane] = {"id": g_id, "y2": y2}

        front_ids = {v["id"] for v in front_by_lane.values()}

        for obj in self.vis_objects:
            g_id = obj['id']
            x1, y1, x2, y2 = obj['bbox']
            cx, cy = (x1 + x2) // 2, y2

            target_lane = None
            if has_lane_filter:
                for name, poly in active_lane_polys.items():
                    if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                        target_lane = name
                        break

                if target_lane is None:
                    continue

            if target_lane:
                if g_id not in self.global_to_local_ids:
                    self.lane_counters[target_lane] += 1
                    self.global_to_local_ids[g_id] = self.lane_counters[target_lane]
                local_id = self.global_to_local_ids[g_id]

                label_prefix = f"No: {local_id} ({target_lane})"
            else:
                label_prefix = f"ID: {g_id}"

            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), BBOX_LINE_THICKNESS)
            vel = obj['vel']
            if target_lane in front_lanes and g_id not in front_ids:
                vel = float("nan")
            v_str = f"{int(vel)}km/h" if np.isfinite(vel) else "--km/h"
            line1 = label_prefix
            line2 = f"Vel: {v_str}"

            font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            (w1, h1), _ = cv2.getTextSize(line1, font, scale, thick)
            (w2, h2), _ = cv2.getTextSize(line2, font, scale, thick)
            max_w = max(w1, w2)
            total_h = h1 + h2 + 8
            bg_top = y1 - total_h - 12
            cv2.rectangle(disp, (x1, bg_top), (x1 + max_w + 10, y1), (0, 0, 0), -1)
            cv2.putText(disp, line1, (x1 + 5, y1 - h2 - 10), font, scale, (255, 255, 255), thick)
            cv2.putText(disp, line2, (x1 + 5, y1 - 5), font, scale, (255, 255, 255), thick)

        # 2. Radar Overlay
        projected_count = 0
        if radar_points is not None and self.cam_K is not None:
            pts_r = radar_points.T
            pts_c = self.Extr_R @ pts_r + self.Extr_t.reshape(3, 1)
            valid_idx = pts_c[2, :] > 0.5

            if np.any(valid_idx):
                pts_c = pts_c[:, valid_idx]
                has_doppler = radar_doppler is not None
                dopplers = radar_doppler[valid_idx] if has_doppler else np.zeros(pts_c.shape[1])
                projected_count = int(pts_c.shape[1])

                uvs = self.cam_K @ pts_c
                uvs /= uvs[2, :]
                h, w = disp.shape[:2]

                for i in range(uvs.shape[1]):
                    u, v = int(uvs[0, i]), int(uvs[1, i])
                    
                    # 점 148개 중 앞쪽 5개만 좌표를 출력해서 위치 확인
                    if i < 5: 
                        print(f"[DEBUG] Point {i} -> u: {u}, v: {v} (Screen Size: {w}x{h})")

                    if 0 <= u < w and 0 <= v < h:
                        if has_doppler:
                            speed_kmh = dopplers[i] * 3.6
                            if abs(speed_kmh) < NOISE_MIN_SPEED_KMH:
                                continue

                            if speed_kmh < SPEED_THRESHOLD_RED:
                                color = (0, 0, 255)
                            elif speed_kmh > SPEED_THRESHOLD_BLUE:
                                color = (255, 0, 0)
                            else:
                                color = (255, 255, 255)
                        else:
                            color = (255, 255, 255)

                        cv2.circle(disp, (u, v), OVERLAY_POINT_RADIUS, color, -1)


        speed_values = [obj["vel"] for obj in self.vis_objects if np.isfinite(obj["vel"])]
        speed_text = "--"
        if speed_values:
            speed_text = f"{np.median(speed_values):.1f}km/h"

        extrinsic_text = "Extrinsic: --"
        if self.extrinsic_last_loaded is not None:
            extrinsic_text = time.strftime("Extrinsic: %H:%M:%S", time.localtime(self.extrinsic_last_loaded))

        overlay_lines = [
            extrinsic_text,
            f"Radar in view: {projected_count}",
            f"Median speed: {speed_text}",
        ]
        for idx, line in enumerate(overlay_lines):
            y = 30 + idx * 20
            cv2.putText(disp, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


        # 3. Lane
        if self.chk_show_poly.isChecked():
            for name, pts in self.lane_polys.items():
                if name in self.chk_lanes and not self.chk_lanes[name].isChecked():
                    continue
                if pts is not None and len(pts) > 0:
                    cv2.polylines(disp, [pts], True, (0, 255, 255), 2)
                    text_pos = self.get_text_position(pts)
                    cv2.putText(disp, name, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        self.viewer.update_image(disp)

        # Graph display removed; no radar scatter update

    # ------------------ Misc ------------------
    def open_lane_editor(self):
        if self.cv_image is None:
            return
        dlg = LaneEditorDialog(self.cv_image, self.lane_polys, self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self.lane_polys = dlg.get_polys()
            self.save_lane_polys()
            self.lbl_log.setText("Lane Polygons Updated.")

    def save_lane_polys(self):
        lane_utils.save_lane_polys(self.lane_json_path, self.lane_polys)

    def load_lane_polys(self):
        try:
            self.lane_polys = lane_utils.load_lane_polys(self.lane_json_path)
        except:
            self.lane_polys = {}

    def load_extrinsic(self):
        if os.path.exists(self.extrinsic_path):
            try:
                with open(self.extrinsic_path, 'r') as f:
                    data = json.load(f)
                self.Extr_R = np.array(data['R'])
                self.Extr_t = np.array(data['t'])
                self.extrinsic_mtime = os.path.getmtime(self.extrinsic_path)
                self.extrinsic_last_loaded = time.time()
                if hasattr(self, "lbl_log"):
                    self.lbl_log.setText("Extrinsic Loaded.")
            except:
                pass

    def run_calibration(self):
        # Launch manual calibration window (replicating SensorsCalibration manual tool)
        dlg = ManualCalibWindow(self)
        dlg.exec()

    def run_intrinsic_calibration(self):
        """
        GUI에서 intrinsic 캘리브레이션(오프라인 이미지) 실행.
        perception_test/image 폴더의 calib_01.jpg ~ 를 사용한다고 가정.
        """
        try:
            rp = rospkg.RosPack()
            pkg_path = rp.get_path("perception_test")
            image_dir = os.path.join(pkg_path, "image")  # <- 너가 말한 경로
            out_yaml = os.path.join(pkg_path, "config", "camera_intrinsic.yaml")

            if not os.path.isdir(image_dir):
                self.lbl_log.setText(f"[Intrinsic] image dir not found: {image_dir}")
                return

            # jpg가 있는지 간단 체크
            import glob
            imgs = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
            if len(imgs) == 0:
                self.lbl_log.setText(f"[Intrinsic] No .jpg in {image_dir}")
                return

            # 이미 실행 중이면 중복 실행 방지
            if hasattr(self, "intrinsic_proc") and self.intrinsic_proc is not None:
                if self.intrinsic_proc.state() != QtCore.QProcess.NotRunning:
                    self.lbl_log.setText("[Intrinsic] already running...")
                    return

            self.txt_intrinsic_log.clear()
            self.pbar_intrinsic.setVisible(True)
            self.pbar_intrinsic.setRange(0, 0)  # 돌아가는 중 표시
            self.lbl_log.setText("[Intrinsic] Running... (GUI에서 로그 확인)")

            # QProcess로 rosrun 실행 (터미널 안 봐도 stdout을 GUI로 가져올 수 있음)
            self.intrinsic_proc = QtCore.QProcess(self)
            self.intrinsic_proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)

            # 중요: rosrun은 ROS 환경이 잡힌 상태에서 GUI가 실행된 경우 정상 동작
            ws = os.path.expanduser("~/motrex/catkin_ws")
            setup_bash = os.path.join(ws, "devel", "setup.bash")

            cmd = "bash"

            ros_cmd = (
                f"source {setup_bash} && "
                f"rosrun perception_test calibration_in_node.py "
                f"_image_dir:={image_dir} "
                f"_output_path:={out_yaml} "
                f"_board/width:=10 _board/height:=7 _board/square_size:=0.025 "
                f"_min_samples:=20 _max_selected:=45 _grid_size_mm:=50 "
                f"_save_selected:=true _save_undistort:=true"
            )

            args = ["-lc", ros_cmd]

            self.intrinsic_proc.readyReadStandardOutput.connect(self._on_intrinsic_stdout)
            self.intrinsic_proc.finished.connect(self._on_intrinsic_finished)

            self.intrinsic_proc.start(cmd, args)

        except Exception as e:
            self.lbl_log.setText(f"[Intrinsic] start error: {e}")
            traceback.print_exc()

    def _on_intrinsic_stdout(self):
        """
        calibration_in_node.py가 찍는 로그를 GUI에 표시.
        (Select image / Calibration done 같은 문구를 그대로 보여줌)
        """
        if not hasattr(self, "intrinsic_proc") or self.intrinsic_proc is None:
            return
        data = bytes(self.intrinsic_proc.readAllStandardOutput()).decode("utf-8", errors="ignore")
        if not data:
            return
        self.txt_intrinsic_log.appendPlainText(data.rstrip())

        # 진행률을 대략적으로라도 보이게: "Select image:" 카운트 기반
        # (완전 정확할 필요 없고, “진행 중” 체감용)
        if "Select image:" in data or "Select image" in data:
            # 로그 전체에서 선택 개수 대략 계산
            text = self.txt_intrinsic_log.toPlainText()
            selected = text.count("Select image")
            self.pbar_intrinsic.setRange(0, 45)
            self.pbar_intrinsic.setValue(min(selected, 45))

    def _on_intrinsic_finished(self, exit_code, exit_status):
        """
        종료 시 YAML 읽어서 K/D 요약을 lbl_log에 표시
        """
        try:
            self.pbar_intrinsic.setVisible(False)

            rp = rospkg.RosPack()
            pkg_path = rp.get_path("perception_test")
            out_yaml = os.path.join(pkg_path, "config", "camera_intrinsic.yaml")

            if exit_code == 0 and os.path.exists(out_yaml):
                # YAML에서 K/D 읽어 요약
                with open(out_yaml, "r", encoding="utf-8") as f:
                    y = yaml.safe_load(f) or {}
                ci = (y.get("camera_info") or {})
                K = ci.get("K", [])
                D = ci.get("D", [])
                # fx, fy, cx, cy 요약
                fx = K[0] if len(K) >= 9 else None
                fy = K[4] if len(K) >= 9 else None
                cx = K[2] if len(K) >= 9 else None
                cy = K[5] if len(K) >= 9 else None

                self.lbl_log.setText(f"[Intrinsic] DONE ✅ fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f} | D={D}")
                self.txt_intrinsic_log.appendPlainText("\n[Intrinsic] DONE. camera_intrinsic.yaml updated.\n")
            else:
                self.lbl_log.setText(f"[Intrinsic] FAILED (exit={exit_code}). See log box.")
                self.txt_intrinsic_log.appendPlainText(f"\n[Intrinsic] FAILED exit_code={exit_code}\n")

        except Exception as e:
            self.lbl_log.setText(f"[Intrinsic] finish error: {e}")
            traceback.print_exc()

    def _maybe_reload_extrinsic(self):
        if not os.path.exists(self.extrinsic_path):
            return
        try:
            mtime = os.path.getmtime(self.extrinsic_path)
        except OSError:
            return
        if self.extrinsic_mtime is None or mtime > self.extrinsic_mtime:
            self.load_extrinsic()


################################################################################
# ManualCalibWindow
################################################################################
class ManualCalibWindow(QtWidgets.QDialog):
    """
    A dialog that allows manual adjustment of the Radar->Camera extrinsic
    transformation. It replicates the interactive behavior of the
    SensorsCalibration radar2camera manual calibration tool. Users can
    incrementally rotate and translate the extrinsic parameters via
    keyboard controls or on-screen buttons, adjust the step sizes, and
    save or reset the calibration result.

    Keyboard bindings (same as SensorsCalibration):

      - Rotation (degrees):
          q: +x (roll)    a: -x (roll)
          w: +y (pitch)   s: -y (pitch)
          e: +z (yaw)     d: -z (yaw)

      - Translation (meters):
          r: +x           f: -x
          t: +y           g: -y
          y: +z           h: -z

    Step sizes for rotation and translation can be adjusted using the
    provided spin boxes. The current transformation is visualized by
    projecting the radar point cloud onto the camera image.
    """

    def __init__(self, gui: 'RealWorldGUI'):
        super().__init__(gui)
        self.gui = gui
        self.setWindowTitle("Manual Radar-Camera Calibration")
        self.setModal(True)
        self.resize(1000, 700)

        # Initial extrinsic values copied from the main GUI
        self.R_init = np.array(self.gui.Extr_R, dtype=float)
        self.t_init = np.array(self.gui.Extr_t, dtype=float).reshape(3, 1)
        self.R_current = self.R_init.copy()
        self.t_current = self.t_init.copy()

        # Default step sizes
        self.deg_step = 1.0  # degrees
        self.trans_step = 0.05  # meters
        self.point_radius = OVERLAY_POINT_RADIUS

        # Build UI
        main_layout = QtWidgets.QHBoxLayout(self)

        # Image display
        self.img_view = ImageCanvasViewer()
        main_layout.addWidget(self.img_view, stretch=4)

        # Control panel
        ctrl_panel = QtWidgets.QWidget()
        ctrl_layout = QtWidgets.QVBoxLayout(ctrl_panel)
        ctrl_panel.setFixedWidth(280)

        # Step size controls
        step_group = QtWidgets.QGroupBox("Step Sizes")
        step_layout = QtWidgets.QFormLayout()
        self.spin_deg = QtWidgets.QDoubleSpinBox()
        self.spin_deg.setRange(0.1, 10.0)
        self.spin_deg.setSingleStep(0.1)
        self.spin_deg.setValue(self.deg_step)
        self.spin_deg.setSuffix(" deg")
        self.spin_deg.valueChanged.connect(self._update_deg_step)
        step_layout.addRow("Rotation step:", self.spin_deg)

        self.spin_trans = QtWidgets.QDoubleSpinBox()
        self.spin_trans.setRange(0.01, 0.5)
        self.spin_trans.setSingleStep(0.01)
        self.spin_trans.setValue(self.trans_step)
        self.spin_trans.setSuffix(" m")
        self.spin_trans.valueChanged.connect(self._update_trans_step)
        step_layout.addRow("Translation step:", self.spin_trans)
        step_group.setLayout(step_layout)
        ctrl_layout.addWidget(step_group)

        # Point size slider
        size_group = QtWidgets.QGroupBox("Point Size")
        size_layout = QtWidgets.QHBoxLayout()
        self.slider_point = QtWidgets.QSlider(Qt.Horizontal)
        self.slider_point.setRange(1, 15)
        self.slider_point.setValue(self.point_radius)
        self.slider_point.valueChanged.connect(self._update_point_size)
        size_layout.addWidget(QtWidgets.QLabel("1"))
        size_layout.addWidget(self.slider_point)
        size_layout.addWidget(QtWidgets.QLabel("15"))
        size_group.setLayout(size_layout)
        ctrl_layout.addWidget(size_group)

        # Buttons: Reset, Save
        btn_reset = QtWidgets.QPushButton("Reset")
        btn_reset.clicked.connect(self._reset_extrinsic)
        btn_save = QtWidgets.QPushButton("Save Result")
        btn_save.clicked.connect(self._save_extrinsic)
        ctrl_layout.addWidget(btn_reset)
        ctrl_layout.addWidget(btn_save)

        # Instructions label
        instr = QtWidgets.QLabel(
            "Use keyboard to adjust extrinsic:\n"
            "Rotation: q/w/e (plus) and a/s/d (minus)\n"
            "Translation: r/t/y (plus) and f/g/h (minus)\n"
            "Adjust step sizes above. Press Save to write to extrinsic.json."
        )
        instr.setWordWrap(True)
        ctrl_layout.addWidget(instr)
        ctrl_layout.addStretch()

        main_layout.addWidget(ctrl_panel, stretch=1)

        # Timer to refresh display
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_view)
        self.timer.start(REFRESH_RATE_MS)

        # Ensure this dialog receives keyboard focus
        self.setFocusPolicy(Qt.StrongFocus)

    def _update_deg_step(self, val: float):
        self.deg_step = float(val)

    def _update_trans_step(self, val: float):
        self.trans_step = float(val)

    def _update_point_size(self, val: int):
        self.point_radius = int(val)

    def _reset_extrinsic(self):
        self.R_current = self.R_init.copy()
        self.t_current = self.t_init.copy()
        self.update_view()

    def _save_extrinsic(self):
        # Save current extrinsic to JSON file and reload in main GUI
        R = self.R_current.tolist()
        t = self.t_current.reshape(-1).tolist()
        data = {"R": R, "t": t}
        try:
            os.makedirs(os.path.dirname(self.gui.extrinsic_path), exist_ok=True)
            with open(self.gui.extrinsic_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            # Reload extrinsic in GUI
            self.gui.load_extrinsic()
            QtWidgets.QMessageBox.information(self, "Saved", "Extrinsic parameters saved and reloaded.")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Save Error", f"Failed to save extrinsic: {e}")

    def keyPressEvent(self, event):
        # Handle rotation and translation keys
        key = event.key()
        if key == Qt.Key_Q:
            self._apply_rotation('x', +self.deg_step)
        elif key == Qt.Key_A:
            self._apply_rotation('x', -self.deg_step)
        elif key == Qt.Key_W:
            self._apply_rotation('y', +self.deg_step)
        elif key == Qt.Key_S:
            self._apply_rotation('y', -self.deg_step)
        elif key == Qt.Key_E:
            self._apply_rotation('z', +self.deg_step)
        elif key == Qt.Key_D:
            self._apply_rotation('z', -self.deg_step)
        elif key == Qt.Key_R:
            self._apply_translation('x', +self.trans_step)
        elif key == Qt.Key_F:
            self._apply_translation('x', -self.trans_step)
        elif key == Qt.Key_T:
            self._apply_translation('y', +self.trans_step)
        elif key == Qt.Key_G:
            self._apply_translation('y', -self.trans_step)
        elif key == Qt.Key_Y:
            self._apply_translation('z', +self.trans_step)
        elif key == Qt.Key_H:
            self._apply_translation('z', -self.trans_step)
        else:
            super().keyPressEvent(event)
        # update view after any change
        self.update_view()

    def _apply_rotation(self, axis: str, delta_deg: float):
        # Compute incremental rotation matrix around axis (camera frame)
        angle_rad = np.deg2rad(delta_deg)
        if axis == 'x':
            R_delta = np.array([
                [1, 0, 0],
                [0, np.cos(angle_rad), -np.sin(angle_rad)],
                [0, np.sin(angle_rad), np.cos(angle_rad)]
            ])
        elif axis == 'y':
            R_delta = np.array([
                [np.cos(angle_rad), 0, np.sin(angle_rad)],
                [0, 1, 0],
                [-np.sin(angle_rad), 0, np.cos(angle_rad)]
            ])
        elif axis == 'z':
            R_delta = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad), 0],
                [np.sin(angle_rad), np.cos(angle_rad), 0],
                [0, 0, 1]
            ])
        else:
            return
        # Apply rotation: R_new = R_delta @ R_current, t_new = R_delta @ t_current
        self.R_current = R_delta @ self.R_current
        self.t_current = R_delta @ self.t_current

    def _apply_translation(self, axis: str, delta: float):
        delta_vec = np.zeros((3, 1))
        idx = {'x': 0, 'y': 1, 'z': 2}.get(axis)
        if idx is not None:
            delta_vec[idx, 0] = delta
            self.t_current = self.t_current + delta_vec

    def update_view(self):
        # Fetch latest image and radar data from main GUI
        cv_img = self.gui.cv_image
        radar_points = self.gui.radar_points
        cam_K = self.gui.cam_K
        if cv_img is None or radar_points is None or cam_K is None:
            return
        disp = cv_img.copy()
        pts_r = radar_points.T  # shape (3,N)
        pts_c = self.R_current @ pts_r + self.t_current
        valid = pts_c[2, :] > 0.5
        if np.any(valid):
            pts_c = pts_c[:, valid]
            uvs = cam_K @ pts_c
            uvs /= uvs[2, :]
            h, w = disp.shape[:2]
            for i in range(uvs.shape[1]):
                u, v = int(uvs[0, i]), int(uvs[1, i])
                if 0 <= u < w and 0 <= v < h:
                    cv2.circle(disp, (u, v), self.point_radius, (0, 255, 255), -1)
        # Display overlay image
        self.img_view.update_image(disp)


def main():
    try:
        app = QtWidgets.QApplication(sys.argv)
        gui = RealWorldGUI()
        gui.show()
        sys.exit(app.exec())
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()