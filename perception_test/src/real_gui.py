#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
real_gui.py (Updated)

[변경 사항]
1. UI 레이아웃 재구성 (우측 패널):
   - 2번: Calibration Accuracy (신설 - 게이지, 이모지, 자석선 토글)
   - 3번: Lane Editor
   - 4번: View Options (텍스트 토글 추가)
   - 5번: Radar Speed Filters (대표 점 토글 추가 - 기본값 True)
   - (Speed Estimation 그룹 제거 -> 내부 기본값으로 동작 유지)

2. 시각화 로직 개선:
   - Magnet Line: BBox 상단 중앙(노란 점) <-> 레이더 대표 점 연결 (초록~빨강)
   - Representative Point: 상단 30% 영역의 중앙값 사용 (Single Point 모드)
   - Accuracy Gauge: 오차율 기반 실시간 점수 표시

3. 데이터 저장 (Data Logging) 수정:
   - update_loop 내 record_buffer 채우기 로직 추가
   - 차선/ID별 고유 키 생성 (IN1->IN2 등 경로 반영)
   - 거리별/ID별 통계 CSV 저장 기능 구현
"""

import sys
import os
import json
import time
import traceback
from collections import deque

import cv2
import numpy as np

from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Qt, Signal, Slot
import message_filters
import rospy
import rospkg
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from cv_bridge import CvBridge
from std_msgs.msg import Bool, String

# ==============================================================================
# Setup & Imports
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
NODES_DIR = os.path.join(CURRENT_DIR, "nodes")
if NODES_DIR in sys.path:
    sys.path.remove(NODES_DIR)
sys.path.insert(0, NODES_DIR)
from perception_lib.lane_editor import LaneEditorDialog
from perception_lib.logging_utils import DataLogger
from perception_lib.speed_utils import (
    build_record_row,
    find_target_lane,
    parse_radar_pointcloud,
    project_points,
    select_representative_point,
    update_lane_tracking,
    update_speed_estimate,
)

from perception_test.msg import AssociationArray, DetectionArray
from perception_lib import calibration_utils
from perception_lib import lane_utils

os.environ["QT_LOGGING_RULES"] = "qt.gui.painting=false"

# ==============================================================================
# [PARAMETER SETTINGS] 사용자 설정
# ==============================================================================

REFRESH_RATE_MS = 33           # 30 FPS
MAX_REASONABLE_KMH = 100.0   # 육교 고정형 기준 상한

# Radar Coloring & Filtering
SPEED_THRESHOLD_RED = -0.5     # m/s (이보다 작으면 접근/빨강)
SPEED_THRESHOLD_BLUE = 0.5     # m/s (이보다 크면 이탈/파랑)
NOISE_MIN_SPEED_KMH = 5.0      # 정지/저속(노이즈) 필터 기준 (km/h)

# Clustering Parameters (Rviz 스타일)
CLUSTER_MAX_DIST = 2.5         # 점 사이 거리 (m)
CLUSTER_MIN_PTS = 3            # 최소 점 개수

CLUSTER_VEL_GATE_MPS = 2.0     # 클러스터 내 속도 일관성 게이트 (m/s)
BBOX2D_MIN_AREA_PX2 = 250      # 2D bbox 최소 면적 (너무 작은 박스 제거)
BBOX2D_MARGIN_PX = 6           # 2D bbox 여유 픽셀

# Visualization
OVERLAY_POINT_RADIUS = 4
BBOX_LINE_THICKNESS = 2
AXIS_LENGTH = 3.0              # 좌표축 화살표 길이 (m)

# Topics
TOPIC_IMAGE = "/camera/image_raw"
TOPIC_RADAR = "/point_cloud"
TOPIC_CAMERA_INFO = "/camera/camera_info"
TOPIC_FINAL_OUT = "/perception_test/output"
TOPIC_TRACKS = "/perception_test/tracks"
TOPIC_ASSOCIATED = "/perception_test/associated"

BRIGHTNESS_PARAM_ENABLE = "/object_detector/runtime/brightness_enable"
BRIGHTNESS_PARAM_GAIN = "/object_detector/runtime/brightness_gain"

START_ACTIVE_LANES = ["IN1", "IN2", "IN3", "OUT1", "OUT2", "OUT3"]

# ==============================================================================
# UI Classes
# ==============================================================================
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
# ManualCalibWindow
# ==============================================================================
class ManualCalibWindow(QtWidgets.QDialog):
    # ROS 메시지를 GUI 스레드로 안전하게 넘기기 위한 시그널
    diag_signal = Signal(str)

    def __init__(self, gui: 'RealWorldGUI'):
        super().__init__(gui, QtCore.Qt.Window | QtCore.Qt.WindowMinMaxButtonsHint | QtCore.Qt.WindowCloseButtonHint)
        self.gui = gui
        self.setWindowTitle("Manual Calibration - Independent Tool")
        
        self.setModal(False) 
        main_geo = self.gui.geometry()
        self.resize(1600, 900) 
        self.move(main_geo.x()+100, main_geo.y()+100) 

        # 물리 파라미터 초기화
        self.T_init = np.eye(4, dtype=np.float64)
        if self.gui.Extr_R is not None: 
            self.T_init[:3, :3] = np.array(self.gui.Extr_R, dtype=np.float64)
        if self.gui.Extr_t is not None: 
            self.T_init[:3, 3]  = np.array(self.gui.Extr_t, dtype=np.float64).flatten()
        
        self.T_current = self.T_init.copy()
        
        self.FIXED_DEG = 0.5   
        self.FIXED_MOV = 0.1   

        # ROS 통신 설정
        self.pub_diag_start = rospy.Publisher("/perception_test/diagnosis/start", Bool, queue_size=1)
        self.sub_diag_result = rospy.Subscriber("/perception_test/diagnosis/result", String, self._on_diag_msg)
        self.diag_signal.connect(self._update_log_ui)

        self.init_ui()
        self._timer = QtCore.QTimer(self); self._timer.timeout.connect(self.update_view); self._timer.start(33)

    def init_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)
        
        self.img_view = ImageCanvasViewer(self)
        main_layout.addWidget(self.img_view, stretch=1)

        ctrl_panel = QtWidgets.QWidget()
        ctrl_panel.setFixedWidth(400)
        vbox = QtWidgets.QVBoxLayout(ctrl_panel)
        vbox.setAlignment(QtCore.Qt.AlignTop)

        ctrl_panel.setStyleSheet("""
            QWidget { font-family: 'Segoe UI', Arial; font-size: 16px; }
            QGroupBox { 
                font-weight: 800; border: 2px solid #b0b0b0; border-radius:8px; 
                margin-top: 20px; padding-top: 15px; background-color: #f9f9f9;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; top: 0px; padding: 0 5px; }
            QPushButton { min-height:45px; border-radius:8px; font-weight:bold; border: 1px solid #999; background:#eee; }
            QPushButton:pressed { background:#ccc; }
        """)

        # (1) Visual Options
        gb_vis = QtWidgets.QGroupBox("1. Visual Options")
        g_vis = QtWidgets.QGridLayout(gb_vis)
        self.chk_grid = QtWidgets.QCheckBox("Purple Grid"); self.chk_grid.setChecked(True)
        self.chk_bbox = QtWidgets.QCheckBox("Vehicle Box"); self.chk_bbox.setChecked(True)
        self.chk_raw = QtWidgets.QCheckBox("Raw Radar Pts"); self.chk_raw.setChecked(True)
        self.chk_noise_filter = QtWidgets.QCheckBox("Noise Filter (<5km/h)"); self.chk_noise_filter.setChecked(True)
        
        g_vis.addWidget(self.chk_grid, 0, 0); g_vis.addWidget(self.chk_bbox, 0, 1)
        g_vis.addWidget(self.chk_raw, 1, 0); g_vis.addWidget(self.chk_noise_filter, 1, 1)
        vbox.addWidget(gb_vis)

        # (2) Diagnostic Tool
        gb_diag = QtWidgets.QGroupBox("2. Diagnostic Tool")
        v_diag = QtWidgets.QVBoxLayout(gb_diag)
        self.btn_diag = QtWidgets.QPushButton("🔍 START DIAGNOSIS (Check Error)")
        self.btn_diag.setStyleSheet("background-color: #3498db; color: white;")
        self.btn_diag.clicked.connect(self._start_diagnostic_ros)
        
        self.txt_log = QtWidgets.QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setPlaceholderText("Ready. Press button to analyze traffic.")
        self.txt_log.setStyleSheet("background:#222; color:#0f0; font-family:Consolas; font-size:14px;")
        self.txt_log.setMinimumHeight(180)
        v_diag.addWidget(self.btn_diag); v_diag.addWidget(self.txt_log)
        vbox.addWidget(gb_diag)

        L_GRAY, D_GRAY = "#e0e0e0", "#757575"

        # (3) Translation Pad
        gb_trans = QtWidgets.QGroupBox("3. Translation (0.1m)")
        t_grid = QtWidgets.QGridLayout(gb_trans)
        t_grid.addWidget(self._create_btn("FWD ▲ (T)", L_GRAY, lambda: self._move(1, 1)), 0, 1)
        t_grid.addWidget(self._create_btn("LEFT ◀ (R)", L_GRAY, lambda: self._move(0, -1)), 1, 0)
        t_grid.addWidget(self._create_btn("RIGHT ▶ (F)", L_GRAY, lambda: self._move(0, 1)), 1, 2)
        t_grid.addWidget(self._create_btn("BWD ▼ (G)", L_GRAY, lambda: self._move(1, -1)), 2, 1)
        t_grid.addWidget(self._create_btn("UP ⤒ (Y)", D_GRAY, lambda: self._move(2, 1), "white"), 0, 2)
        t_grid.addWidget(self._create_btn("DOWN ⤓ (H)", D_GRAY, lambda: self._move(2, -1), "white"), 2, 2)
        vbox.addWidget(gb_trans)

        # (4) Rotation Pad
        gb_rot = QtWidgets.QGroupBox("4. Rotation (0.5°)")
        r_grid = QtWidgets.QGridLayout(gb_rot)
        r_grid.addWidget(self._create_btn("Pitch+ ⇈ (Q)", L_GRAY, lambda: self._rotate(0, 1)), 0, 1)
        r_grid.addWidget(self._create_btn("Yaw+ ↶ (W)", L_GRAY, lambda: self._rotate(1, -1)), 1, 0)
        r_grid.addWidget(self._create_btn("Yaw- ↷ (S)", L_GRAY, lambda: self._rotate(1, 1)), 1, 2)
        r_grid.addWidget(self._create_btn("Pitch- ⇊ (A)", L_GRAY, lambda: self._rotate(0, -1)), 2, 1)
        r_grid.addWidget(self._create_btn("Roll+ ↺ (E)", D_GRAY, lambda: self._rotate(2, 1), "white"), 0, 0)
        r_grid.addWidget(self._create_btn("Roll- ↻ (D)", D_GRAY, lambda: self._rotate(2, -1), "white"), 0, 2)
        vbox.addWidget(gb_rot)

        # (5) Bottom Buttons
        self.btn_save = QtWidgets.QPushButton("💾 APPLY & SAVE EXTRINSIC")
        self.btn_save.setStyleSheet("background-color: #4CAF50; color: white; min-height: 50px;")
        self.btn_save.clicked.connect(self._save_extrinsic)
        self.btn_reset = QtWidgets.QPushButton("↺ RESET TO INITIAL")
        self.btn_reset.setStyleSheet("background-color: #f44336; color: white; min-height: 40px;")
        self.btn_reset.clicked.connect(self._reset_T)

        vbox.addStretch(); vbox.addWidget(self.btn_save); vbox.addWidget(self.btn_reset)
        main_layout.addWidget(ctrl_panel)

    def _create_btn(self, text, bg_color, func, text_color="black"):
        btn = QtWidgets.QPushButton(text)
        btn.setStyleSheet(f"background-color: {bg_color}; color: {text_color}; font-size: 12px;")
        btn.clicked.connect(func); return btn

    def _move(self, axis, sign):
        move_vec = np.zeros(3); move_vec[axis] = sign * self.FIXED_MOV
        self.T_current[:3, 3] += self.T_current[:3, :3] @ move_vec
        self.update_view()

    def _rotate(self, axis, sign):
        rad = np.deg2rad(sign * self.FIXED_DEG)
        c, s = np.cos(rad), np.sin(rad)
        if axis == 0: R_inc = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif axis == 1: R_inc = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        else: R_inc = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        self.T_current[:3, :3] = self.T_current[:3, :3] @ R_inc
        self.update_view()

    def keyPressEvent(self, event):
        k = event.key()
        if k == QtCore.Qt.Key_Q: self._rotate(0, 1)
        elif k == QtCore.Qt.Key_A: self._rotate(0, -1)
        elif k == QtCore.Qt.Key_W: self._rotate(1, -1)
        elif k == QtCore.Qt.Key_S: self._rotate(1, 1)
        elif k == QtCore.Qt.Key_E: self._rotate(2, 1)
        elif k == QtCore.Qt.Key_D: self._rotate(2, -1)
        elif k == QtCore.Qt.Key_R: self._move(0, -1)
        elif k == QtCore.Qt.Key_F: self._move(0, 1)
        elif k == QtCore.Qt.Key_T: self._move(1, 1)
        elif k == QtCore.Qt.Key_G: self._move(1, -1)
        elif k == QtCore.Qt.Key_Y: self._move(2, 1)
        elif k == QtCore.Qt.Key_H: self._move(2, -1)
        event.accept()

    def update_view(self):
        try:
            if self.gui.latest_frame is None or self.gui.cam_K is None: return
            raw_img = self.gui.latest_frame.get("cv_image")
            pts_r = self.gui.latest_frame.get("radar_points")
            dops = self.gui.latest_frame.get("radar_doppler")
            if raw_img is None: return

            disp = raw_img.copy()
            K = self.gui.cam_K
            uvs_man, valid_man = project_points(K, self.T_current[:3, :3], self.T_current[:3, 3], pts_r)

            if self.chk_grid.isChecked():
                self._draw_grid_and_axis(disp, K, K[0, 2], K[1, 2])
            if self.chk_bbox.isChecked() and self.gui.vis_objects:
                for obj in self.gui.vis_objects:
                    x1, y1, x2, y2 = obj['bbox']
                    cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if self.chk_raw.isChecked() and uvs_man is not None:
                for i in range(len(uvs_man)):
                    if not valid_man[i]: continue
                    vel = dops[i] if (dops is not None and i < len(dops)) else 0.0
                    if self.chk_noise_filter.isChecked() and abs(vel) < 1.4: continue
                    color = (0, 0, 255) if vel < -0.5 else (255, 0, 0) if vel > 0.5 else (0, 255, 0)
                    cv2.circle(disp, (int(uvs_man[i, 0]), int(uvs_man[i, 1])), 3, color, -1)
            self.img_view.update_image(disp)
        except Exception: pass

    def _draw_grid_and_axis(self, img, K, cx, cy):
        grid_z = -1.5; purple = (255, 0, 255)
        for x in np.arange(-15, 16, 3): self._draw_safe_line(img, [x, 0, grid_z], [x, 100, grid_z], K, cx, cy, purple)
        for y in range(0, 101, 10): self._draw_safe_line(img, [-15, y, grid_z], [15, y, grid_z], K, cx, cy, purple)

    def _draw_safe_line(self, img, p1, p2, K, cx, cy, color):
        pts = np.vstack([p1, p2]); uv, v = project_points(K, self.T_current[:3,:3], self.T_current[:3,3], pts)
        if v[0] and v[1]: cv2.line(img, (int(uv[0,0]), int(uv[0,1])), (int(uv[1,0]), int(uv[1,1])), color, 1)

    def _save_extrinsic(self):
        data = { "R": self.T_current[:3, :3].tolist(), "t": self.T_current[:3, 3].flatten().tolist() }
        with open(self.gui.extrinsic_path, 'w') as f: json.dump(data, f, indent=4)
        self.gui.load_extrinsic(); self.close()

    def _reset_T(self): 
        self.T_current = self.T_init.copy(); self.update_view()
        self.txt_log.appendPlainText(">>> Reset to initial values.")

    def _start_diagnostic_ros(self):
        """ROS 노드에게 진단 시작 요청 (고정 시점용 멘트)"""
        self.btn_diag.setEnabled(False)
        # [수정] 육교 고정 시점이므로 '운전' 지시 제거
        self.txt_log.setPlainText(">> [REQUEST] Starting 30s Diagnosis...\n>> Collecting traffic data (50m+)...")
        self.pub_diag_start.publish(True)

    def _on_diag_msg(self, msg):
        self.diag_signal.emit(msg.data)

    @Slot(str)
    def _update_log_ui(self, text):
        self.btn_diag.setEnabled(True)
        self.txt_log.appendPlainText("\n" + "="*30)
        self.txt_log.appendPlainText(text)
        self.txt_log.appendPlainText("="*30)
        self.txt_log.verticalScrollBar().setValue(self.txt_log.verticalScrollBar().maximum())

    def closeEvent(self, e): 
        self._timer.stop() 
        super().closeEvent(e)

# ==============================================================================
# Main GUI - [수정됨: 우측 패널 상단 세로 로고 배치, 전체화면]
# ==============================================================================
class RealWorldGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Motrex-SKKU Sensor Fusion GUI")
        
        # 전체 화면으로 시작
        self.showMaximized()

        rospy.init_node('real_gui_node', anonymous=True)
        self.bridge = CvBridge()
        pkg_path = rospkg.RosPack().get_path("perception_test")
        self.nodes_dir = os.path.join(CURRENT_DIR, "nodes")
        # extrinsic and lane polygon files are stored in the config folder
        self.extrinsic_path = os.path.join(pkg_path, "config", "extrinsic.json")
        self.lane_json_path = os.path.join(pkg_path, "config", "lane_polys.json")

        self.cv_image = None
        self.radar_points = None
        self.radar_doppler = None
        self.last_radar_pts = None
        self.last_radar_dop = None
        self.last_radar_ts = 0.0
        self.radar_hold_threshold = 0.06
        self.cam_K = None
        self.lane_polys = {}
        self.Extr_R = np.eye(3)
        self.Extr_t = np.zeros((3, 1))
        self.vis_objects = []
        self.assoc_speed_by_id = {}
        self.assoc_last_update_time = 0.0
        self.lane_counters = {name: 0 for name in ["IN1", "IN2", "IN3", "OUT1", "OUT2", "OUT3"]}
        self.global_to_local_ids = {}
        self.vehicle_lane_paths = {}
        self.extrinsic_mtime = None
        self.extrinsic_last_loaded = None
        self.rep_pt_mem = {} 
        self.track_confirmed_cnt = {}
        self.pt_alpha = 0.15
        self.vel_gate_mps = 1.5 
        self.spatial_weight_sigma = 30.0 
        self.min_pts_for_rep = 2
        self.kf_vel = {}  # 차량 ID별 속도 칼만 필터 딕셔너리
        self.kf_score = {} # 차량 ID별 점수 칼만 필터 (점수 급락 방지용)

        # ---------------- Speed estimation memory / smoothing ----------------
        self.vel_memory = {}
        self.vel_hold_sec_default = 10.0
        self.radar_frame_count = 0
        self.radar_warmup_frames = 8
        self.max_jump_kmh = 25.0
        self.use_nearest_k_points = True
        self.nearest_k = 3
        self.speed_update_period = 2
        self._speed_frame_counter = 0
        self.speed_hard_max_kmh = 280.0
        self.speed_soft_max_kmh = 100.0

        # Data Recording Buffer
        self.data_logger = DataLogger()

        # Buffer for the latest synchronized frame
        self.latest_frame = None

        self.load_extrinsic()
        self.load_lane_polys()
        self.init_ui()

        # -------------------------------------------------------------
        # Time Synchronizer
        # -------------------------------------------------------------
        image_sub = message_filters.Subscriber(TOPIC_IMAGE, Image)
        radar_sub = message_filters.Subscriber(TOPIC_RADAR, PointCloud2)

        self.ts = message_filters.ApproximateTimeSynchronizer([image_sub, radar_sub], 10, 0.1)
        self.ts.registerCallback(self.cb_sync)

        rospy.Subscriber(TOPIC_CAMERA_INFO, CameraInfo, self.cb_info, queue_size=1)
        rospy.Subscriber(TOPIC_FINAL_OUT, AssociationArray, self.cb_final_result, queue_size=1)
        rospy.Subscriber(TOPIC_ASSOCIATED, AssociationArray, self.cb_association_result, queue_size=1)
        rospy.Subscriber(TOPIC_TRACKS, DetectionArray, self.cb_tracker_result, queue_size=1)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(REFRESH_RATE_MS)

        self.last_update_time = time.time()

    def init_ui(self):
        # 메인 위젯
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        
        # [수정] 전체 레이아웃: 가로 배치 (좌: 뷰어 | 우: 패널)
        # 상단 헤더 없이 꽉 차게 사용
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # 1. 왼쪽: 이미지 뷰어
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.viewer = ImageCanvasViewer()
        left_layout.addWidget(self.viewer, stretch=1)
        layout.addWidget(left_widget, stretch=1) # 뷰어가 남는 공간 모두 차지

        # 2. 오른쪽: 컨트롤 패널
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(400) # 너비 고정
        
        # 스타일시트 적용
        panel.setStyleSheet("""
        QWidget { font-family: Segoe UI, Arial; font-size: 20px; }
        QGroupBox {
            font-weight: 800; color:#222;
            border: 2px solid #d9d9d9; border-radius:10px;
            margin-top: 35px; /* 제목 겹침 방지 */
            background:#f7f7f7;
            padding-top: 15px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 15px; top: 0px; padding: 0 5px;
        }
        QLabel { color:#222; }
        QCheckBox { spacing:6px; }
        QDoubleSpinBox, QSpinBox, QComboBox {
            min-height:32px; padding:2px 6px;
            border:1px solid #d0d0d0; border-radius:6px;
        }
        QPushButton {
            min-height:36px; border-radius:8px;
            font-weight:700; background:#efefef;
        }
        """)

        vbox = QtWidgets.QVBoxLayout(panel)
        vbox.setAlignment(Qt.AlignTop)
        vbox.setSpacing(15)

        # ---------------------------------------------------------
        # [수정] 로고 영역 (우측 패널 최상단, 세로 배치)
        # ---------------------------------------------------------
        try:
            rp = rospkg.RosPack()
            pkg_path = rp.get_path("perception_test")
            img_dir = os.path.join(pkg_path, "image", "etc")
            
            lbl_skku = QtWidgets.QLabel()
            lbl_motrex = QtWidgets.QLabel()
            
            pix_skku = QtGui.QPixmap(os.path.join(img_dir, "SKKU.png"))
            pix_motrex = QtGui.QPixmap(os.path.join(img_dir, "Motrex.jpeg"))
            
            # [수정] 세로 배치이므로 너비 기준으로 스케일링
            # 패널 너비(400) 안쪽으로 들어오게 설정
            if not pix_skku.isNull():
                # 성대 로고 크게
                scaled_skku = pix_skku.scaledToWidth(400, Qt.SmoothTransformation)
                lbl_skku.setPixmap(scaled_skku)
                lbl_skku.setAlignment(Qt.AlignCenter)
            
            if not pix_motrex.isNull():
                # 모트렉스 로고는 상대적으로 작게
                scaled_motrex = pix_motrex.scaledToWidth(300, Qt.SmoothTransformation)
                lbl_motrex.setPixmap(scaled_motrex)
                lbl_motrex.setAlignment(Qt.AlignCenter)
            
            # 세로(Vertical)로 추가
            vbox.addWidget(lbl_skku)
            vbox.addSpacing(5) # 로고 사이 간격
            vbox.addWidget(lbl_motrex)
            vbox.addSpacing(20) # 로고와 컨트롤 사이 간격
            
        except Exception as e:
            print(f"Logo load error: {e}")


        # ---------------------------------------------------------
        # 컨트롤 위젯들
        # ---------------------------------------------------------

        # 1. Calibration
        gb_calib = QtWidgets.QGroupBox("1. Calibration")
        v_c = QtWidgets.QVBoxLayout()
        v_c.setAlignment(QtCore.Qt.AlignTop) 
        v_c.setSpacing(8)

        btn_intri = QtWidgets.QPushButton("Run Intrinsic")
        btn_intri.clicked.connect(self.run_intrinsic_calibration)
        btn_autocal = QtWidgets.QPushButton("Run Extrinsic (Auto)")
        btn_autocal.clicked.connect(self.run_autocalibration)
        btn_calib = QtWidgets.QPushButton("Run Extrinsic (Manual)")
        btn_calib.clicked.connect(self.run_calibration)
        btn_reload = QtWidgets.QPushButton("Reload JSON")
        btn_reload.clicked.connect(self.load_extrinsic)
        
        self.pbar_intrinsic = QtWidgets.QProgressBar()
        self.pbar_intrinsic.setRange(0, 0)
        self.pbar_intrinsic.setVisible(False)

        self.txt_intrinsic_log = QtWidgets.QPlainTextEdit()
        self.txt_intrinsic_log.setReadOnly(True)
        self.txt_intrinsic_log.setMaximumHeight(100)

        self.pbar_extrinsic = QtWidgets.QProgressBar()
        self.pbar_extrinsic.setRange(0, 0)
        self.pbar_extrinsic.setVisible(False)

        self.txt_extrinsic_log = QtWidgets.QPlainTextEdit()
        self.txt_extrinsic_log.setReadOnly(True)
        self.txt_extrinsic_log.setMaximumHeight(100)

        v_c.addWidget(btn_intri)
        v_c.addWidget(btn_autocal)
        v_c.addWidget(btn_calib)
        v_c.addWidget(btn_reload)
        v_c.addWidget(self.pbar_intrinsic)
        v_c.addWidget(self.txt_intrinsic_log)
        v_c.addWidget(self.pbar_extrinsic)
        v_c.addWidget(self.txt_extrinsic_log)

        gb_calib.setLayout(v_c)
        vbox.addWidget(gb_calib)

        # 2. Calibration Accuracy (New)
        gb2 = QtWidgets.QGroupBox("2. Calibration Accuracy")
        v2 = QtWidgets.QVBoxLayout()
        
        h_acc = QtWidgets.QHBoxLayout()
        self.bar_acc = QtWidgets.QProgressBar()
        self.bar_acc.setRange(0, 100)
        self.bar_acc.setValue(0)
        self.bar_acc.setStyleSheet("QProgressBar::chunk { background-color: #4CAF50; }")
        self.lbl_emoji = QtWidgets.QLabel("😐")
        self.lbl_emoji.setStyleSheet("font-size: 30px;")
        h_acc.addWidget(self.bar_acc); h_acc.addWidget(self.lbl_emoji)
        
        self.chk_magnet = QtWidgets.QCheckBox("Show Magnet Line (Validation)")
        self.chk_magnet.setChecked(True)
        
        v2.addLayout(h_acc)
        v2.addWidget(self.chk_magnet)
        gb2.setLayout(v2); vbox.addWidget(gb2)

        # 3. Lane Editor
        gb_lane = QtWidgets.QGroupBox("3. Lane Editor")
        v_l = QtWidgets.QVBoxLayout()
        btn_edit = QtWidgets.QPushButton("🖌️ Open Editor")
        btn_edit.setStyleSheet("background-color: #FFD700; color: black; font-weight: bold; padding: 10px;")
        btn_edit.clicked.connect(self.open_lane_editor)
        v_l.addWidget(btn_edit)
        gb_lane.setLayout(v_l)
        vbox.addWidget(gb_lane)

        # 4. Data Logging
        gb_logging = QtWidgets.QGroupBox("4. Data Logging")
        v_log = QtWidgets.QVBoxLayout()
        v_log.setSpacing(10)

        self.btn_main_record = QtWidgets.QPushButton("🔴 START MAIN RECORDING")
        self.btn_main_record.setCheckable(True)
        # 연한 초록색(#CCFFCC) 기본 스타일 설정
        self.btn_main_record.setStyleSheet("""
            QPushButton { 
                min-height: 50px; font-size: 20px; 
                background-color: #CCFFCC; border: 1px solid #999; 
            }
            QPushButton:checked { 
                background-color: #FFCCCC; color: red; border: 2px solid red; 
            }
        """)
        self.btn_main_record.clicked.connect(self._toggle_recording)

        self.lbl_record_status = QtWidgets.QLabel("Status: Ready")
        self.lbl_record_status.setAlignment(Qt.AlignCenter)
        self.lbl_record_status.setStyleSheet("font-size: 16px; color: #555; font-weight: bold;")

        v_log.addWidget(self.btn_main_record)
        v_log.addWidget(self.lbl_record_status)
        gb_logging.setLayout(v_log)
        
        vbox.addWidget(gb_logging) # 4번 섹션으로 추가

        # 5. View Options
        gb_vis = QtWidgets.QGroupBox("5. View Options")
        v_vis = QtWidgets.QVBoxLayout()
        
        self.chk_show_poly = QtWidgets.QCheckBox("Show Lane Polygons")
        self.chk_show_poly.setChecked(False)
        
        # [수정] 아이디와 속도 체크박스
        self.chk_show_id = QtWidgets.QCheckBox("Show ID Text")
        self.chk_show_id.setChecked(True)
        self.chk_show_speed = QtWidgets.QCheckBox("Show Speed Text")
        self.chk_show_speed.setChecked(True)

        # [추가] 6번에서 가져온 레이더 포인트 토글 (검정색, 진하게)
        self.chk_show_radar = QtWidgets.QCheckBox("Show Radar Points")
        self.chk_show_radar.setChecked(True)
        self.chk_show_radar.setStyleSheet("color: black; font-weight: bold;")
        
        v_vis.addWidget(self.chk_show_poly)
        v_vis.addWidget(self.chk_show_id)
        v_vis.addWidget(self.chk_show_speed)
        v_vis.addWidget(self.chk_show_radar)

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

        # [삭제] 6. Radar Speed Filters 섹션 전체 삭제됨

        # 6. Image Brightness (번호 7 -> 6으로 변경)
        gb_brightness = QtWidgets.QGroupBox("6. Image Brightness")
        v_b = QtWidgets.QVBoxLayout()
        self.chk_brightness = QtWidgets.QCheckBox("Enable Brightness Gain")
        self.chk_brightness.setChecked(False)
        row_b = QtWidgets.QHBoxLayout()
        row_b.addWidget(QtWidgets.QLabel("Gain (x):"))
        self.spin_brightness_gain = QtWidgets.QDoubleSpinBox()
        self.spin_brightness_gain.setDecimals(2)
        self.spin_brightness_gain.setRange(0.3, 2.5)
        self.spin_brightness_gain.setSingleStep(0.05)
        self.spin_brightness_gain.setValue(1.0)
        row_b.addWidget(self.spin_brightness_gain, stretch=1)
        v_b.addWidget(self.chk_brightness)
        v_b.addLayout(row_b)
        gb_brightness.setLayout(v_b)
        vbox.addWidget(gb_brightness)
        
        self.chk_brightness.toggled.connect(self._update_brightness_params)
        self.spin_brightness_gain.valueChanged.connect(self._update_brightness_params)

        # 하단 여백 추가
        vbox.addStretch()
        layout.addWidget(panel, stretch=0)
        
        self._update_brightness_params()
        
        # 내부적으로 사용하는 speed parameter 들 (UI에는 없지만 로직 유지를 위해 필요)
        self.spin_hold_sec = type('obj', (object,), {'value': lambda: 10.0})
        self.cmb_smoothing = type('obj', (object,), {'currentText': lambda: "EMA"})
        self.spin_ema_alpha = type('obj', (object,), {'value': lambda: 0.35})

    def reset_ids(self):
        self.lane_counters = {name: 0 for name in self.lane_counters.keys()}
        self.global_to_local_ids = {}
        self.vehicle_lane_paths = {}

    def _update_brightness_params(self):
        if not hasattr(self, "chk_brightness") or not hasattr(self, "spin_brightness_gain"):
            return
        enabled = bool(self.chk_brightness.isChecked())
        gain = float(self.spin_brightness_gain.value())
        try:
            rospy.set_param(BRIGHTNESS_PARAM_ENABLE, enabled)
            rospy.set_param(BRIGHTNESS_PARAM_GAIN, gain)
        except Exception as e:
            print(f"Brightness Param Error: {e}")

    def _toggle_recording(self):
        """기록 시작/중단 토글 및 스타일 제어"""
        if self.btn_main_record.isChecked():
            # 저장 중 상태 (연한 빨간색은 QSS에서 처리됨)
            self.data_logger.start()
            self.btn_main_record.setText("⏹ STOP & SAVE DATA")
            self.lbl_record_status.setText("Status: Recording...")
        else:
            # 저장 대기 상태 (연한 초록색은 QSS에서 처리됨)
            self.btn_main_record.setText("🔴 START MAIN RECORDING")
            self._save_sequential_data()

    def _save_sequential_data(self):
        """순차적 폴더(1, 2, 3...) 생성 및 데이터 저장 (수정됨)"""
        result = self.data_logger.stop_and_save()
        if result["status"] == "no_data":
            self.lbl_record_status.setText("Status: No Data Collected")
            return

        if result["status"] == "saved":
            self.lbl_record_status.setText(f"Status: Saved in Folder {result['folder']}")
            print(f"[Data Log] Saved to {result['target_dir']}")
            return
        self.lbl_record_status.setText("Status: Error")

    # ------------------ Callback (Synchronized) ------------------
    def cb_sync(self, img_msg, radar_msg):
        # 1) 이미지
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except:
            return

        # 2) 레이더
        radar_points, radar_doppler = parse_radar_pointcloud(radar_msg)
        if radar_points is None and radar_doppler is None:
            return

        frame = {
            "stamp": img_msg.header.stamp.to_sec() if img_msg.header.stamp else time.time(),
            "cv_image": cv_image,
            "radar_points": radar_points,
            "radar_doppler": radar_doppler,
        }

        if radar_points is not None and radar_doppler is not None:
            self.radar_frame_count += 1 
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
        try:
            self._maybe_reload_extrinsic()
            self._speed_frame_counter += 1
            
            # 0. 데이터 유효성 검사
            if self.latest_frame is None: return

            # 1. 데이터 추출 (None 체크 강화)
            self.cv_image = self.latest_frame.get("cv_image")
            if self.cv_image is None: self.cv_image = self.latest_frame.get("img")

            pts_raw = self.latest_frame.get("radar_points")
            if pts_raw is None: pts_raw = self.latest_frame.get("pts")

            dop_raw = self.latest_frame.get("radar_doppler")
            if dop_raw is None: dop_raw = self.latest_frame.get("dop")

            now = time.time()
            if pts_raw is not None and len(pts_raw) > 0:
                # 1. 새로운 유효 데이터가 들어온 경우 -> 메모리 업데이트
                self.last_radar_pts = pts_raw
                self.last_radar_dop = dop_raw
                self.last_radar_ts = now
            else:
                # 2. 데이터가 없는 경우 -> 60ms 이내의 최신 데이터가 있다면 복구
                if self.last_radar_pts is not None:
                    time_elapsed = now - self.last_radar_ts
                    if time_elapsed < self.radar_hold_threshold:
                        pts_raw = self.last_radar_pts
                        dop_raw = self.last_radar_dop
                        # print(f"[Hold] Using cached radar data ({time_elapsed*1000:.1f}ms old)")

            if self.cv_image is None: return

            disp = self.cv_image.copy()
            
            # 밝기 조절
            if hasattr(self, "chk_brightness") and self.chk_brightness.isChecked():
                g = self.spin_brightness_gain.value()
                if abs(g - 1.0) > 0.001:
                    disp = cv2.convertScaleAbs(disp, alpha=g, beta=0)

            # 2. 레이더 투영
            proj_uvs = None
            proj_valid = None
            projected_count = 0 

            if pts_raw is not None and self.cam_K is not None:
                uvs, valid = project_points(self.cam_K, self.Extr_R, self.Extr_t, pts_raw)
                proj_uvs = uvs
                proj_valid = valid
                projected_count = int(np.sum(valid))

            # 3. 차선 필터 준비
            active_lane_polys = {}
            for name, chk in self.chk_lanes.items():
                if chk.isChecked():
                    poly = self.lane_polys.get(name)
                    if poly is not None and len(poly) > 2:
                        active_lane_polys[name] = poly
            has_lane_filter = bool(active_lane_polys)

            draw_items = []
            acc_scores = []
            now_ts = time.time()

            # 4. 객체별 데이터 처리 루프
            confirmed_objects = [] # 실제로 2초 이상 검증된 객체들만 담을 리스트
            
            for obj in self.vis_objects:
                g_id = obj['id']
                
                # 연속 감지 카운트 업데이트
                self.track_confirmed_cnt[g_id] = self.track_confirmed_cnt.get(g_id, 0) + 1
                
                # 2초(30 FPS * 2 = 60프레임) 미만으로 감지된 객체는 무시 (반사광 박스 제거)
                if self.track_confirmed_cnt[g_id] < 20:
                    continue
                
                confirmed_objects.append(obj)
                
            # 검증된 객체들로 다시 루프 수행
            for obj in confirmed_objects:
                g_id = obj['id']
                x1, y1, x2, y2 = obj['bbox']
                
                # 1. 차선 필터링 (기존 로직 유지)
                cx, cy = (x1 + x2) // 2, y2
                target_lane = None
                if has_lane_filter:
                    target_lane = find_target_lane((cx, cy), active_lane_polys)
                    if target_lane is None:
                        continue

                curr_lane_str, lane_path, local_no, label = update_lane_tracking(
                    g_id,
                    target_lane,
                    self.vehicle_lane_paths,
                    self.lane_counters,
                    self.global_to_local_ids,
                )

                obj['lane_path'] = lane_path
                obj['lane'] = curr_lane_str

                target_pt = (int((x1+x2)/2), int((y1+y2)/2)) 
                
                meas_kmh = None
                rep_pt = None
                rep_vel_raw = None
                score = 0
                radar_dist = -1.0 # 거리 저장용 변수 초기화

                if proj_uvs is not None and dop_raw is not None:
                    meas_kmh, rep_pt, rep_vel_raw, score, radar_dist = select_representative_point(
                        proj_uvs,
                        proj_valid,
                        dop_raw,
                        (x1, y1, x2, y2),
                        pts_raw,
                        self.vel_gate_mps,
                        self.rep_pt_mem,
                        self.pt_alpha,
                        now_ts,
                        g_id,
                        NOISE_MIN_SPEED_KMH,  # [수정] UI 변수 대신 고정값(5.0) 사용
                    )
                    if rep_pt is not None:
                        acc_scores.append(score)

                vel_out, score_out = update_speed_estimate(
                    g_id,
                    meas_kmh,
                    score,
                    now_ts,
                    self.vel_memory,
                    self.kf_vel,
                    self.kf_score,
                )

                obj['vel'] = round(vel_out, 1) if vel_out is not None else float('nan')
                score = score_out

                # --- [DATA LOGGING] ---
                if self.data_logger.is_recording:
                    data_row = build_record_row(
                        now_ts,
                        g_id,
                        local_no,
                        lane_path,
                        curr_lane_str,
                        radar_dist,
                        meas_kmh,
                        vel_out,
                        score,
                    )
                    self.data_logger.add_record(data_row)

                draw_items.append({
                    "obj": obj, "bbox": (x1, y1, x2, y2), "label": label,
                    "rep_pt": rep_pt, "rep_vel": rep_vel_raw, "target_pt": target_pt, "score": score
                })

            # 5. 전역 UI 업데이트
            if acc_scores:
                avg = sum(acc_scores) / len(acc_scores)
                self.bar_acc.setValue(int(avg))
                self.lbl_emoji.setText("😊" if avg >= 80 else "😐" if avg >= 50 else "😟")
            
            if self.data_logger.is_recording:
                 self.lbl_record_status.setText(f"Recording... [{self.data_logger.record_count} pts]")

            # 6. 그리기 (Draw)
            # [수정] "Show Radar Points"가 켜져 있을 때만 노이즈 필터링된 점들을 그림
            if self.chk_show_radar.isChecked() and proj_uvs is not None:
                for i in range(len(proj_uvs)):
                    if not proj_valid[i]: continue
                    u, v = int(proj_uvs[i, 0]), int(proj_uvs[i, 1])
                    spd = dop_raw[i] * 3.6
                    
                    # [수정] UI 변수 대신 고정값 사용
                    if abs(spd) < NOISE_MIN_SPEED_KMH: continue
                    
                    col = (0, 0, 255) if spd < -0.5 else (255, 0, 0) if spd > 0.5 else (0, 255, 0)
                    cv2.circle(disp, (u, v), 2, col, -1)

            for it in draw_items:
                x1, y1, x2, y2 = it["bbox"]
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(disp, it["target_pt"], 4, (0, 255, 255), -1) 

                if it["rep_pt"] is not None:
                    if self.chk_magnet.isChecked():
                        lc = (0, 255, 0) if it["score"] > 50 else (0, 0, 255)
                        cv2.line(disp, it["target_pt"], it["rep_pt"], lc, 2)
                    
                    # [수정] 대표점은 체크박스 없이 항상 그림 (단, 속도가 노이즈 이상일 때)
                    rv = it["rep_vel"]
                    if rv is not None and abs(rv) >= NOISE_MIN_SPEED_KMH:
                        col = (0, 0, 255) if rv < -0.5 else (255, 0, 0) if rv > 0.5 else (0, 255, 0)
                        cv2.circle(disp, it["rep_pt"], 5, col, -1)
                        cv2.circle(disp, it["rep_pt"], 6, (255, 255, 255), 1)

                lines = []
                if self.chk_show_id.isChecked(): lines.append(it['label'])
                if self.chk_show_speed.isChecked():
                    v = it["obj"].get("vel", float("nan"))
                    lines.append(f"Vel: {v:.1f}km/h" if np.isfinite(v) else "Vel: --km/h")

                if lines:
                    font, sc, th = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                    max_w, total_h, l_sizes = 0, 0, []
                    for ln in lines:
                        (w, h), _ = cv2.getTextSize(ln, font, sc, th)
                        max_w = max(max_w, w); total_h += h + 12; l_sizes.append((w, h))

                    bg_top = y1 - total_h - 15
                    cv2.rectangle(disp, (x1, bg_top), (x1 + max_w + 10, y1), (0, 0, 0), -1)
                    curr_y = y1 - 10
                    for i in range(len(lines) - 1, -1, -1):
                        cv2.putText(disp, lines[i], (x1 + 5, curr_y), font, sc, (255, 255, 255), th)
                        curr_y -= l_sizes[i][1] + 12

            speed_vals = [obj["vel"] for obj in self.vis_objects if np.isfinite(obj["vel"])]
            med_speed = f"{np.median(speed_vals):.2f}km/h" if speed_vals else "--"
            overlay_text = f"Radar: {projected_count} pts | Median Speed: {med_speed}"
            cv2.putText(disp, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if self.chk_show_poly.isChecked():
                for pts in self.lane_polys.values():
                    if pts is not None and len(pts) > 0:
                        cv2.polylines(disp, [pts], True, (0, 255, 255), 2)

            current_ids = {o['id'] for o in self.vis_objects}
            self.track_confirmed_cnt = {k: v for k, v in self.track_confirmed_cnt.items() if k in current_ids}

            self.viewer.update_image(disp)

        except Exception as e:
            print(f"[GUI Error] update_loop failed: {e}")
            traceback.print_exc()
            
    def open_lane_editor(self):
        if self.cv_image is None:
            return
        dlg = LaneEditorDialog(self.cv_image, self.lane_polys, self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self.lane_polys = dlg.get_polys()
            self.save_lane_polys()
            # self.lbl_log.setText("Lane Polygons Updated.") (로그 라벨 제거됨)

    def save_lane_polys(self):
        lane_utils.save_lane_polys(self.lane_json_path, self.lane_polys)

    def load_lane_polys(self):
        try:
            self.lane_polys = lane_utils.load_lane_polys(self.lane_json_path)
        except Exception:
            self.lane_polys = {}

    def load_extrinsic(self):
        calibration_utils.load_extrinsic(self)

    def run_calibration(self):
        calibration_utils.run_calibration(self, ManualCalibWindow)

    def run_autocalibration(self):
        calibration_utils.run_autocalibration(self)

    def _on_extrinsic_stdout(self):
        if not hasattr(self, "extrinsic_proc") or self.extrinsic_proc is None:
            return
        data = bytes(self.extrinsic_proc.readAllStandardOutput()).decode("utf-8", errors="ignore")
        if not data:
            return
        self.txt_extrinsic_log.appendPlainText(data.rstrip())

    def _on_extrinsic_stderr(self):
        if not hasattr(self, "extrinsic_proc") or self.extrinsic_proc is None:
            return
        data = bytes(self.extrinsic_proc.readAllStandardError()).decode("utf-8", errors="ignore")
        if not data:
            return
        self.txt_extrinsic_log.appendPlainText(data.rstrip())

    def _on_extrinsic_finished(self):
        if hasattr(self, "pbar_extrinsic") and self.pbar_extrinsic is not None:
            self.pbar_extrinsic.setVisible(False)
        # 오류 수정: 문자열 내 실제 줄바꿈 제거
        self.txt_extrinsic_log.appendPlainText("[Extrinsic] DONE.")

    def run_intrinsic_calibration(self):
        calibration_utils.run_intrinsic_calibration(self)

    def _on_intrinsic_stdout(self):
        if not hasattr(self, "intrinsic_proc") or self.intrinsic_proc is None:
            return
        data = bytes(self.intrinsic_proc.readAllStandardOutput()).decode("utf-8", errors="ignore")
        if not data:
            return
        self.txt_intrinsic_log.appendPlainText(data.rstrip())

        if "Select image:" in data or "Select image" in data:
            text = self.txt_intrinsic_log.toPlainText()
            selected = text.count("Select image")
            self.pbar_intrinsic.setRange(0, 45)
            self.pbar_intrinsic.setValue(min(selected, 45))

    def _on_intrinsic_stderr(self):
        if not hasattr(self, "intrinsic_proc") or self.intrinsic_proc is None:
            return
        data = bytes(self.intrinsic_proc.readAllStandardError()).decode("utf-8", errors="ignore")
        if not data:
            return
        self.txt_intrinsic_log.appendPlainText(data.rstrip())

    def _on_intrinsic_finished(self, exit_code, exit_status):
        try:
            self.pbar_intrinsic.setVisible(False)
            rp = rospkg.RosPack()
            pkg_path = rp.get_path("perception_test")
            out_yaml = os.path.join(pkg_path, "config", "camera_intrinsic.yaml")

            if exit_code == 0 and os.path.exists(out_yaml):
                import yaml
                with open(out_yaml, "r", encoding="utf-8") as f:
                    y = yaml.safe_load(f) or {}
                ci = (y.get("camera_info") or {})
                # 오류 수정: 줄바꿈 문자 처리 방식 변경
                self.txt_intrinsic_log.appendPlainText("[Intrinsic] DONE. camera_intrinsic.yaml updated.")
            else:
                self.txt_intrinsic_log.appendPlainText(f"[Intrinsic] FAILED exit_code={exit_code}")

        except Exception as e:
            print(f"[Intrinsic] finish error: {e}")
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

def main():
    try:
        app = QtWidgets.QApplication(sys.argv)
        def _qt_msg_handler(mode, context, message):
            if message.startswith("QPainter::end: Painter ended with"):
                return
            sys.stderr.write(message + " ")

        QtCore.qInstallMessageHandler(_qt_msg_handler)
        gui = RealWorldGUI()
        gui.show()
        sys.exit(app.exec())
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()