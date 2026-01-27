#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
real_gui.py (Data-Driven Axis Fix)

[데이터 분석 결과 반영]
- message_filters로 영상-레이더 시간 동기화 적용.

[추가 수정(수동 캘리브레이션 UX 개선)]
- 레이더 포인트 오버레이에 줌(축소/확대) 기능 추가 및 줌 스텝 조절 UI 제공.
- 수동 캘리브레이션 창에서 회전/이동/줌 누적 변화량을 표시하여 조정량을 즉시 확인 가능.
- 레이더 오버레이 색상 규칙을 보완하여 접근/이탈/저속 포인트를 쉽게 구분.
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
        self._image_size = None
        self._scaled_rect = None
        self._click_callback = None
        self.zoom = 1.0
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
        self._image_size = (w, h)
        self.update()

    def set_click_callback(self, callback):
        self._click_callback = callback

    def set_zoom(self, zoom: float):
        self.zoom = float(np.clip(zoom, 0.00001, 5.0))
        self.update()

    def nudge_zoom(self, delta: float):
        self.set_zoom(self.zoom + delta)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        if self.pixmap:
            target_w = int(self.width() * self.zoom)
            target_h = int(self.height() * self.zoom)
            target_size = QtCore.QSize(max(1, target_w), max(1, target_h))
            scaled = self.pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
            self._scaled_rect = QtCore.QRect(x, y, scaled.width(), scaled.height())
        else:
            painter.setPen(QtCore.Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter, "Waiting for Camera Stream...")

    def mousePressEvent(self, event):
        if not self._click_callback or not self._scaled_rect or not self._image_size:
            return
        pos = event.position().toPoint()
        if not self._scaled_rect.contains(pos):
            return
        scale_x = self._image_size[0] / self._scaled_rect.width()
        scale_y = self._image_size[1] / self._scaled_rect.height()
        img_x = (pos.x() - self._scaled_rect.x()) * scale_x
        img_y = (pos.y() - self._scaled_rect.y()) * scale_y
        self._click_callback(int(img_x), int(img_y))

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
        self.radar_doppler = frame["radar_doppler"]

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

                            if abs(speed_kmh) <= 0.5:
                                color = (0, 255, 0)
                            elif speed_kmh < SPEED_THRESHOLD_RED:
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

# ==============================================================================
# ManualCalibWindow (Camera-Frame Control: Intuitive Mode)
# ==============================================================================
class ManualCalibWindow(QtWidgets.QDialog):
    """
    Manual Calibration with 'Screen-Space' Controls.
    - Controls apply rotation relative to the CAMERA VIEW (Screen), not the Radar body.
    - Solves the issue where 'Yaw' causes vertical movement due to axis misalignment.
    """

    def __init__(self, gui: 'RealWorldGUI'):
        super().__init__(gui)
        self.gui = gui
        self.setWindowTitle("Manual Calibration (Intuitive Screen-Space Control)")
        self.setModal(True)
        self.resize(1700, 1000)

        # -------------------------------------------------------------
        # 1. Initialize State
        # -------------------------------------------------------------
        self.T_init = np.eye(4, dtype=np.float64)
        self.T_init[:3, :3] = np.array(self.gui.Extr_R, dtype=np.float64)
        self.T_init[:3, 3] = np.array(self.gui.Extr_t, dtype=np.float64).flatten()
        self.T_current = self.T_init.copy()

        # Step Sizes
        self.deg_step = 0.5
        self.trans_step = 0.05 # 5cm 단위로 정밀하게
        self.zoom_step = 0.1
        self.point_radius = OVERLAY_POINT_RADIUS
        self.radar_zoom = 1.0
        
        # Accumulators (For display only)
        self.acc_rot = np.zeros(3, dtype=float)
        self.acc_trans = np.zeros(3, dtype=float)
        
        # BEV Settings
        self.lane_points = []
        self.lane_pick_active = False
        self.homography = None
        self.px_per_meter = 10.0
        self.bev_size = (500, 1000)
        
        # Config Path
        try:
            rp = rospkg.RosPack()
            self.config_dir = os.path.join(rp.get_path("perception_test"), "config")
        except:
            self.config_dir = os.path.dirname(self.gui.extrinsic_path)
        os.makedirs(self.config_dir, exist_ok=True)
        self.homography_path = os.path.join(self.config_dir, "center_camera-homography.json")

        # ================= UI Layout =================
        main_layout = QtWidgets.QHBoxLayout(self)
        
        # Left Viewers
        left_cont = QtWidgets.QWidget()
        left_layout = QtWidgets.QHBoxLayout(left_cont)
        self.bev_view = ImageCanvasViewer()
        self.img_view = ImageCanvasViewer()
        self.img_view.set_click_callback(self._on_image_click)
        left_layout.addWidget(self.bev_view, stretch=4)
        left_layout.addWidget(self.img_view, stretch=5)
        main_layout.addWidget(left_cont, stretch=5)

        # Right Controls
        ctrl_panel = QtWidgets.QWidget()
        ctrl_panel.setFixedWidth(420)
        vbox = QtWidgets.QVBoxLayout(ctrl_panel)

        # (A) Visuals
        gb_vis = QtWidgets.QGroupBox("1. Visual Aids")
        f_vis = QtWidgets.QFormLayout()
        self.chk_axis = QtWidgets.QCheckBox("Show Axes"); self.chk_axis.setChecked(True)
        self.chk_grid = QtWidgets.QCheckBox("Show Grid"); self.chk_grid.setChecked(True)
        self.chk_axis.toggled.connect(self.update_view)
        self.chk_grid.toggled.connect(self.update_view)
        f_vis.addRow(self.chk_axis, self.chk_grid)
        gb_vis.setLayout(f_vis)
        vbox.addWidget(gb_vis)

        # (B) Steps
        gb_step = QtWidgets.QGroupBox("2. Step Sizes")
        f_step = QtWidgets.QFormLayout()
        self.s_deg = QtWidgets.QDoubleSpinBox(); self.s_deg.setRange(0.1, 10.0); self.s_deg.setValue(self.deg_step)
        self.s_deg.valueChanged.connect(lambda v: setattr(self, 'deg_step', v))
        self.s_trans = QtWidgets.QDoubleSpinBox(); self.s_trans.setRange(0.01, 1.0); self.s_trans.setValue(self.trans_step)
        self.s_trans.valueChanged.connect(lambda v: setattr(self, 'trans_step', v))
        self.s_zoom = QtWidgets.QDoubleSpinBox(); self.s_zoom.setRange(0.1, 5.0); self.s_zoom.setValue(1.0)
        self.s_zoom.valueChanged.connect(lambda v: setattr(self, 'zoom_step', v))
        f_step.addRow("Rot (°):", self.s_deg)
        f_step.addRow("Move (m):", self.s_trans)
        f_step.addRow("Zoom Step:", self.s_zoom)
        gb_step.setLayout(f_step)
        vbox.addWidget(gb_step)

        # (C) BEV
        gb_bev = QtWidgets.QGroupBox("3. BEV Setup")
        h_bev = QtWidgets.QHBoxLayout()
        self.btn_pick = QtWidgets.QPushButton("Pick 4 Pts"); self.btn_pick.setCheckable(True)
        self.btn_pick.clicked.connect(self._toggle_lane_pick)
        btn_clr = QtWidgets.QPushButton("Clear"); btn_clr.clicked.connect(self._reset_lane_points)
        h_bev.addWidget(self.btn_pick); h_bev.addWidget(btn_clr)
        self.lbl_bev = QtWidgets.QLabel("Not Loaded")
        self.lbl_bev.setStyleSheet("color: #0000FF; font-weight: bold;")
        v_bev = QtWidgets.QVBoxLayout(); v_bev.addWidget(self.lbl_bev); v_bev.addLayout(h_bev)
        gb_bev.setLayout(v_bev)
        vbox.addWidget(gb_bev)

        # (D) Changes
        gb_delta = QtWidgets.QGroupBox("4. Changes (Screen Relative)")
        g_delta = QtWidgets.QGridLayout()
        self.l_rot = [QtWidgets.QLabel("0.0°") for _ in range(3)]
        self.l_trans = [QtWidgets.QLabel("0.00m") for _ in range(3)]
        self.l_zoom = QtWidgets.QLabel("1.00 x")
        
        red = "color: #FF0000; font-weight: bold;"
        rows = ["Tilt (Up/Dn)", "Pan (L/R)", "Roll (CW/CCW)"]
        for i in range(3):
            lb = QtWidgets.QLabel(rows[i]); lb.setStyleSheet(red)
            self.l_rot[i].setStyleSheet(red); self.l_trans[i].setStyleSheet(red)
            g_delta.addWidget(lb, i, 0)
            g_delta.addWidget(self.l_rot[i], i, 1)
            g_delta.addWidget(self.l_trans[i], i, 2)
        
        lz = QtWidgets.QLabel("Zoom:"); lz.setStyleSheet(red); self.l_zoom.setStyleSheet(red)
        g_delta.addWidget(lz, 3, 0); g_delta.addWidget(self.l_zoom, 3, 1)
        gb_delta.setLayout(g_delta)
        vbox.addWidget(gb_delta)

        # (E) Actions
        h_act = QtWidgets.QHBoxLayout()
        btn_rst = QtWidgets.QPushButton("Reset"); btn_rst.clicked.connect(self._reset_T)
        btn_sv = QtWidgets.QPushButton("Save"); btn_sv.setStyleSheet("background-color:#007700; color:white; font-weight:bold")
        btn_sv.clicked.connect(self._save_extrinsic)
        h_act.addWidget(btn_rst); h_act.addWidget(btn_sv)
        vbox.addLayout(h_act)

        # (F) Guide
        gb_key = QtWidgets.QGroupBox("Control Guide (Look at Screen)")
        l_k = QtWidgets.QLabel(
            "⬆⬇ Pitch (Tilt): [W] / [S]\n"
            "⬅➡ Yaw (Pan):  [A] / [D]\n"
            "🔄 Roll:       [Q] / [E]\n"
            "Move: [R]/[F](X), [T]/[G](Y), [Y]/[H](Z)"
        )
        l_k.setStyleSheet("color: #333333; font-weight: bold;")
        vk = QtWidgets.QVBoxLayout(); vk.addWidget(l_k)
        gb_key.setLayout(vk)
        vbox.addWidget(gb_key)
        vbox.addStretch()

        main_layout.addWidget(ctrl_panel)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_view)
        self.timer.start(33)
        self._load_homography_json(silent=True)
        self.setFocusPolicy(Qt.StrongFocus)

    # ------------------------------------------------
    # Logic: Camera-Frame Rotation (Pre-Multiplication)
    # ------------------------------------------------
    def keyPressEvent(self, event):
        k = event.key()
        
        # Rotation (Screen Based)
        # W/S: Rotate around Camera X (Right) -> Tilt Up/Down
        if k == Qt.Key_W: self._rotate_camera_frame(0, 1)  # Pitch Up
        elif k == Qt.Key_S: self._rotate_camera_frame(0, -1) # Pitch Down
        
        # A/D: Rotate around Camera Y (Down) -> Pan Left/Right
        elif k == Qt.Key_A: self._rotate_camera_frame(1, 1)  # Yaw Left
        elif k == Qt.Key_D: self._rotate_camera_frame(1, -1) # Yaw Right
        
        # Q/E: Rotate around Camera Z (Forward) -> Roll
        elif k == Qt.Key_Q: self._rotate_camera_frame(2, 1)  # Roll CCW
        elif k == Qt.Key_E: self._rotate_camera_frame(2, -1) # Roll CW
        
        # Translation (Still in Camera Frame directions)
        elif k == Qt.Key_R: self._translate_camera_frame(0, 1) # Right
        elif k == Qt.Key_F: self._translate_camera_frame(0, -1) # Left
        elif k == Qt.Key_T: self._translate_camera_frame(1, 1) # Down
        elif k == Qt.Key_G: self._translate_camera_frame(1, -1) # Up
        elif k == Qt.Key_Y: self._translate_camera_frame(2, 1) # Forward
        elif k == Qt.Key_H: self._translate_camera_frame(2, -1) # Backward
        
        # Zoom
        elif k == Qt.Key_Z: 
            self.radar_zoom = max(0.1, self.radar_zoom - self.zoom_step)
            self.l_zoom.setText(f"{self.radar_zoom:.2f} x")
        elif k == Qt.Key_X: 
            self.radar_zoom += self.zoom_step
            self.l_zoom.setText(f"{self.radar_zoom:.2f} x")
        
        self.update_view()

    def _rotate_camera_frame(self, axis, sign):
        """
        Apply rotation in CAMERA FRAME (Pre-multiplication).
        T_new = R_delta @ T_current
        """
        angle = sign * np.deg2rad(self.deg_step)
        c, s = np.cos(angle), np.sin(angle)
        
        R_delta = np.eye(4)
        if axis == 0: # X (Pitch)
            R_delta[:3, :3] = np.array([[1,0,0],[0,c,-s],[0,s,c]])
        elif axis == 1: # Y (Yaw)
            R_delta[:3, :3] = np.array([[c,0,s],[0,1,0],[-s,0,c]])
        elif axis == 2: # Z (Roll)
            R_delta[:3, :3] = np.array([[c,-s,0],[s,c,0],[0,0,1]])
            
        # Apply to LEFT side (Global/Camera frame)
        self.T_current = R_delta @ self.T_current
        
        self.acc_rot[axis] += (sign * self.deg_step)
        self._update_delta_ui()

    def _translate_camera_frame(self, axis, sign):
        """
        Apply translation in CAMERA FRAME.
        """
        T_delta = np.eye(4)
        T_delta[axis, 3] = sign * self.trans_step
        
        # Apply to LEFT side
        self.T_current = T_delta @ self.T_current
        
        self.acc_trans[axis] += (sign * self.trans_step)
        self._update_delta_ui()

    def _update_delta_ui(self):
        for i in range(3):
            self.l_rot[i].setText(f"{self.acc_rot[i]:+.1f}°")
            self.l_trans[i].setText(f"{self.acc_trans[i]:+.2f}m")

    def _reset_T(self):
        self.T_current = self.T_init.copy()
        self.acc_rot[:] = 0; self.acc_trans[:] = 0
        self.radar_zoom = 1.0
        self._update_delta_ui()
        self.update_view()

    def _save_extrinsic(self):
        R = self.T_current[:3, :3]; t = self.T_current[:3, 3].reshape(3, 1)
        data = {"R": R.tolist(), "t": t.flatten().tolist()}
        with open(self.gui.extrinsic_path, 'w') as f: json.dump(data, f, indent=4)
        self.gui.load_extrinsic()
        QtWidgets.QMessageBox.information(self, "Saved", "Extrinsic Updated!")

    # ------------------------------------------------
    # View & Projection
    # ------------------------------------------------
    def update_view(self):
        cv_img = self.gui.cv_image
        if cv_img is None: return
        disp = cv_img.copy()
        h, w = disp.shape[:2]
        K = self.gui.cam_K
        cx, cy = (K[0,2], K[1,2]) if K is not None else (w/2, h/2)

        # 1. BEV
        if self.homography is not None:
            bev = cv2.warpPerspective(disp, self.homography, self.bev_size)
        else:
            bev = np.zeros((self.bev_size[1], self.bev_size[0], 3), dtype=np.uint8)
        self._draw_bev_overlays(bev)

        # 2. Grid on Camera
        if self.chk_grid.isChecked() and K is not None:
            self._draw_ground_grid(disp, K, cx, cy)

        # 3. Points
        if self.gui.radar_points is not None and K is not None:
            pts_r = self.gui.radar_points.T
            # T_current maps Radar -> Camera directly now
            pts_c = (self.T_current @ np.vstack((pts_r, np.ones((1, pts_r.shape[1])))))[:3, :]
            
            valid = pts_c[2,:] > 0.5
            if np.any(valid):
                uvs = K @ pts_c[:, valid]
                uvs /= uvs[2, :]
                u = (uvs[0,:] - cx) * self.radar_zoom + cx
                v = (uvs[1,:] - cy) * self.radar_zoom + cy
                
                for i in range(len(u)):
                    ux, vx = int(u[i]), int(v[i])
                    if 0<=ux<w and 0<=vx<h: cv2.circle(disp, (ux, vx), self.point_radius, (0, 255, 0), -1)
                
                # Draw on BEV (Un-zoomed)
                if self.homography is not None:
                    uvs_src = np.array([uvs[0,:], uvs[1,:]]).T.reshape(-1, 1, 2)
                    uvs_dst = cv2.perspectiveTransform(uvs_src, self.homography)
                    for pt in uvs_dst:
                        bx, by = int(pt[0][0]), int(pt[0][1])
                        if 0<=bx<self.bev_size[0] and 0<=by<self.bev_size[1]:
                            cv2.circle(bev, (bx, by), 4, (0, 255, 0), -1)

        # 4. Axis
        if self.chk_axis.isChecked() and K is not None:
            self._draw_radar_axes(disp, K, cx, cy)

        # 5. Lane Points
        for idx, (x, y) in enumerate(self.lane_points):
            cv2.circle(disp, (int(x), int(y)), 6, (0, 165, 255), -1)
            cv2.putText(disp, str(idx+1), (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)

        self.img_view.update_image(disp)
        self.bev_view.update_image(bev)

    def _draw_bev_overlays(self, bev):
        h, w = bev.shape[:2]; cx = w//2; px = int(self.px_per_meter)
        # Grid
        for i in range(0, h, px*10):
            cv2.line(bev, (0, h-i), (w, h-i), (50,50,50), 1)
            if i>0: cv2.putText(bev, f"{i//px}m", (5, h-i-5), 0, 0.5, (100,100,100), 1)
        # Lane (Blue)
        lx = int(cx - 1.75*px); rx = int(cx + 1.75*px)
        cv2.line(bev, (lx, 0), (lx, h), (255,0,0), 2)
        cv2.line(bev, (rx, 0), (rx, h), (255,0,0), 2)
        cv2.line(bev, (cx, 0), (cx, h), (80,80,80), 1)

    def _draw_ground_grid(self, img, K, cx, cy):
        xs = np.linspace(-10, 10, 5); ys = np.linspace(0, 40, 5); rh = -1.0
        lines = []
        for x in xs: lines.append([[x, 0, rh], [x, 40, rh]])
        for y in ys: lines.append([[-10, y, rh], [10, y, rh]])
        for ln in lines:
            pts = (self.T_current @ np.vstack((np.array(ln).T, np.ones((1,2)))))[:3,:]
            if pts[2,0]<0.5 or pts[2,1]<0.5: continue
            uvs = K @ pts; uvs /= uvs[2,:]
            u1=(uvs[0,0]-cx)*self.radar_zoom+cx; v1=(uvs[1,0]-cy)*self.radar_zoom+cy
            u2=(uvs[0,1]-cx)*self.radar_zoom+cx; v2=(uvs[1,1]-cy)*self.radar_zoom+cy
            if -2000<u1<4000: cv2.line(img, (int(u1),int(v1)), (int(u2),int(v2)), (200,200,0), 1)

    def _draw_radar_axes(self, img, K, cx, cy):
        pts = np.array([[0,0,0],[2,0,0],[0,2,0],[0,0,2]], dtype=float).T
        pts_c = (self.T_current @ np.vstack((pts, np.ones((1,4)))))[:3,:]
        uvs = K @ pts_c; uvs /= uvs[2,:]
        def px(i): return (int((uvs[0,i]-cx)*self.radar_zoom+cx), int((uvs[1,i]-cy)*self.radar_zoom+cy))
        o=px(0); x=px(1); y=px(2); z=px(3)
        h,w=img.shape[:2]
        if not (0<=o[0]<w and 0<=o[1]<h): return
        cv2.arrowedLine(img, o, x, (0,0,255), 3, 0.1); cv2.putText(img, "X", x, 0, 0.6, (0,0,255), 2)
        cv2.arrowedLine(img, o, y, (0,255,0), 3, 0.1); cv2.putText(img, "Y", y, 0, 0.6, (0,255,0), 2)
        cv2.arrowedLine(img, o, z, (255,0,0), 3, 0.1); cv2.putText(img, "Z", z, 0, 0.6, (255,0,0), 2)

    # BEV Setup (Same as before)
    def _toggle_lane_pick(self):
        self.lane_pick_active = self.btn_pick.isChecked()
        if self.lane_pick_active: self.lane_points = []; self.lbl_bev.setText("Pick 4 points (Bot->Top)")
        else: self.lbl_bev.setText("Canceled")
    def _reset_lane_points(self): self.lane_points = []; self.homography = None; self.lbl_bev.setText("Cleared")
    def _on_image_click(self, x, y):
        if not self.lane_pick_active: return
        self.lane_points.append((x, y))
        if len(self.lane_points) == 4: self._compute_homography(); self.lane_pick_active = False; self.btn_pick.setChecked(False)
    def _compute_homography(self):
        pts = np.array(self.lane_points, dtype=np.float32)
        idx = np.argsort(pts[:, 1])[::-1]; b = pts[idx[:2]]; t = pts[idx[2:]]
        b = b[np.argsort(b[:, 0])]; t = t[np.argsort(t[:, 0])]
        src = np.array([b[0], b[1], t[1], t[0]], dtype=np.float32)
        bw, bh = self.bev_size; sc = self.px_per_meter; cx = bw/2.0; by = bh-50.0
        dst = np.array([[cx-1.75*sc, by], [cx+1.75*sc, by], [cx+1.75*sc, by-15.0*sc], [cx-1.75*sc, by-15.0*sc]], dtype=np.float32)
        self.homography = cv2.getPerspectiveTransform(src, dst)
        self.lbl_bev.setText("Status: Clean BEV Set")
        self._save_homography_json(src, dst)
    def _save_homography_json(self, s, d): 
        d = {"center_camera-homography": {"param": {"src_quad": [{"x":float(p[0]),"y":float(p[1])} for p in s], "dst_quad_pixels": [{"x":float(p[0]),"y":float(p[1])} for p in d]}}}
        with open(self.homography_path, 'w') as f: json.dump(d, f, indent=4)
    def _load_homography_json(self, silent=False):
        if not os.path.exists(self.homography_path): return
        try:
            with open(self.homography_path, 'r') as f: d = json.load(f)
            p = d[list(d.keys())[0]]["param"]
            s = np.array([[x["x"], x["y"]] for x in p["src_quad"]], dtype=np.float32)
            if "dst_quad_pixels" in p:
                ds = np.array([[x["x"], x["y"]] for x in p["dst_quad_pixels"]], dtype=np.float32)
                self.homography = cv2.getPerspectiveTransform(s, ds)
                self.lbl_bev.setText("Status: Config Loaded")
        except: pass














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