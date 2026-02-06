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
"""

import sys
import os
import json
import time
import traceback
import numpy as np
import cv2
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Qt
from collections import deque

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
# Setup & Imports
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

try:
    from PySide6 import QtWidgets, QtGui, QtCore
    import rospy
    import rospkg
    import message_filters
    from sensor_msgs.msg import Image, PointCloud2, CameraInfo
    from cv_bridge import CvBridge
    import sensor_msgs.point_cloud2 as pc2

    from perception_test.msg import AssociationArray, DetectionArray
    from perception_lib import perception_utils
    from perception_lib import lane_utils
except ImportError as e:
    print(f"\n[GUI Error] 필수 라이브러리 로드 실패: {e}")
    sys.exit(1)


# ==============================================================================
# Helper Functions (Clustering & Math)
# ==============================================================================

def _robust_mean_velocity(vs: np.ndarray, trim_ratio: float = 0.2) -> float:
    """클러스터 속도 요약: 상/하위 trim_ratio 제거 후 평균(노이즈 완화)."""
    if vs is None or len(vs) == 0:
        return 0.0
    vs = np.asarray(vs, dtype=np.float64)
    if vs.size < 5:
        return float(np.mean(vs))
    vs_sorted = np.sort(vs)
    k = int(np.floor(vs_sorted.size * trim_ratio))
    if k*2 >= vs_sorted.size:
        return float(np.mean(vs_sorted))
    vs_trim = vs_sorted[k: vs_sorted.size - k]
    return float(np.mean(vs_trim))

def cluster_radar_points(points_xyz, points_vel, max_dist=2.0, min_pts=3, vel_gate_mps=None):
    """
    간단한 유클리드 거리 기반 클러스터링 (BFS)
    points_xyz: (N, 3) array
    points_vel: (N, ) array (Doppler)
    return: list of dict {'center':(x,y,z), 'min':..., 'max':..., 'vel':..., 'pts':...}
    """
    n = points_xyz.shape[0]
    if n == 0:
        return []

    # XY 평면 거리만 사용하여 클러스터링 (Z는 무시)
    pts_xy = points_xyz[:, :2]

    if points_vel is None:
        points_vel = np.zeros(n, dtype=np.float64)
    if vel_gate_mps is None:
        vel_gate_mps = CLUSTER_VEL_GATE_MPS

    visited = np.zeros(n, dtype=bool)
    clusters = []

    for i in range(n):
        if visited[i]:
            continue

        # BFS
        queue = [i]
        visited[i] = True
        cluster_indices = []

        while queue:
            idx = queue.pop()
            cluster_indices.append(idx)

            # 현재 점과 다른 모든 점 사이의 거리 계산 (Vectorized)
            dists = np.linalg.norm(pts_xy - pts_xy[idx], axis=1)
            vel_d = np.abs(points_vel - points_vel[idx])
            neighbors = np.where((dists <= max_dist) & (vel_d <= vel_gate_mps) & (~visited))[0]

            for nb in neighbors:
                visited[nb] = True
                queue.append(nb)

        if len(cluster_indices) >= min_pts:
            indices = np.array(cluster_indices)
            c_pts = points_xyz[indices]
            c_vels = points_vel[indices]
            
            # BBox Info extraction
            min_xyz = np.min(c_pts, axis=0)
            max_xyz = np.max(c_pts, axis=0)
            center = (min_xyz + max_xyz) / 2.0
            mean_vel = _robust_mean_velocity(c_vels)

            clusters.append({
                'center': center,
                'min': min_xyz,
                'max': max_xyz,
                'vel': mean_vel,
                'indices': indices
            })

    return clusters

def project_points(K, R, t, points_3d):
    """
    3D Points (N,3) -> 2D Pixels (N,2) projection
    """
    if points_3d.shape[0] == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=bool)

    # T: Radar -> Camera
    # P_cam = R * P_rad + t
    pts_h = np.hstack((points_3d, np.ones((points_3d.shape[0], 1)))) # (N, 4)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    
    pts_cam = (T @ pts_h.T).T # (N, 4)
    
    # Valid check (Z > 0.5m)
    valid = pts_cam[:, 2] > 0.5
    
    uvs = np.zeros((points_3d.shape[0], 2))
    
    if np.any(valid):
        p_valid = pts_cam[valid, :3].T # (3, N_valid)
        uv_homo = K @ p_valid
        uv_homo /= uv_homo[2, :] # Normalize by Z
        uvs[valid] = uv_homo[:2, :].T

    return uvs, valid

# ==============================================================================
# Speed Estimation Helpers
# ==============================================================================

def cluster_1d_by_gap(values: np.ndarray, max_gap: float):
    '''
    1D 클러스터링: 정렬 후 인접 값 차이가 max_gap 이하인 구간을 같은 클러스터로 묶음.
    values: (N,) float
    return: list of index arrays (original indices)
    '''
    if values is None:
        return []
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return []
    order = np.argsort(values)
    clusters = []
    start = 0
    for i in range(1, order.size):
        if (values[order[i]] - values[order[i-1]]) > max_gap:
            clusters.append(order[start:i])
            start = i
    clusters.append(order[start:order.size])
    return clusters

def select_best_forward_cluster(forward_vals: np.ndarray, dopplers: np.ndarray, max_gap_m: float = 2.0):
    '''
    bbox 내부 포인트 중 forward축(기본 y)으로 1D clustering 후, '가장 그럴듯한 차량' 클러스터 선택.
    - 우선순위: 포인트 개수 많은 클러스터
    - 동률이면: forward 중앙값이 더 작은(더 가까운) 클러스터
    return: selected doppler array (m/s)
    '''
    if forward_vals is None or dopplers is None:
        return np.asarray([], dtype=np.float64)
    forward_vals = np.asarray(forward_vals, dtype=np.float64)
    dopplers = np.asarray(dopplers, dtype=np.float64)
    if forward_vals.size == 0:
        return np.asarray([], dtype=np.float64)

    clusters = cluster_1d_by_gap(forward_vals, max_gap=max_gap_m)
    if not clusters:
        return np.asarray([], dtype=np.float64)

    best = None
    best_score = None
    for c in clusters:
        c = np.asarray(c, dtype=int)
        if c.size == 0:
            continue
        cnt = int(c.size)
        med_f = float(np.median(forward_vals[c]))
        score = (cnt, -med_f)
        if best is None or score > best_score:
            best = c
            best_score = score

    if best is None:
        return np.asarray([], dtype=np.float64)
    return dopplers[best]

class ScalarKalman:
    '''속도(스칼라)만 필터링하는 간단한 칼만 필터.'''
    def __init__(self, x0=0.0, P0=25.0, Q=2.0, R=9.0):
        self.x = float(x0)
        self.P = float(P0)
        self.Q = float(Q)
        self.R = float(R)

    def predict(self):
        self.P = self.P + self.Q
        return self.x

    def update(self, z: float):
        z = float(z)
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1.0 - K) * self.P
        return self.x


# ==============================================================================
# UI Classes
# ==============================================================================

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
# ManualCalibWindow (Rviz-Style Visuals)
# ==============================================================================
class ManualCalibWindow(QtWidgets.QDialog):
    def __init__(self, gui: 'RealWorldGUI'):
        super().__init__(gui)
        self.gui = gui
        self.setWindowTitle("Manual Calibration (Center Matching)")
        self.setModal(True)
        self.resize(1600, 900)

        # ---------------------------------------------------------
        # 1. 초기값 및 설정
        # ---------------------------------------------------------
        self.T_init = np.eye(4, dtype=np.float64)
        if self.gui.Extr_R is not None:
            self.T_init[:3, :3] = np.array(self.gui.Extr_R, dtype=np.float64)
        if self.gui.Extr_t is not None:
            self.T_init[:3, 3]  = np.array(self.gui.Extr_t, dtype=np.float64).flatten()
        
        self.T_current = self.T_init.copy()

        # 조작 감도
        self.deg_step     = 0.5
        self.trans_step   = 0.1
        self.radar_zoom   = 1.0
        self.is_fine_mode = False

        # 화면 표시용 누적값
        self.acc_rot   = np.zeros(3, dtype=np.float64)
        self.acc_trans = np.zeros(3, dtype=np.float64)

        # ---------------------------------------------------------
        # 2. UI 패널 구성
        # ---------------------------------------------------------
        ctrl_panel = QtWidgets.QWidget()
        ctrl_panel.setFixedWidth(420)
        ctrl_panel.setStyleSheet("""
        QWidget { font-family: Segoe UI, Arial; font-size: 20px; }
        QGroupBox { font-weight: 800; color:#222; border: 2px solid #d9d9d9; border-radius:10px; margin-top: 30px; background:#f7f7f7; padding-top: 15px; }
        QGroupBox::title { subcontrol-origin: margin; left: 15px; top: 0px; padding: 0 5px; }
        QPushButton { min-height:40px; border-radius:8px; font-weight:700; background:#e0e0e0; }
        """)

        vbox = QtWidgets.QVBoxLayout(ctrl_panel)

        # (1) Visual Options
        gb_vis = QtWidgets.QGroupBox("1. Visual Options")
        g_vis = QtWidgets.QGridLayout()
        
        self.chk_grid = QtWidgets.QCheckBox("Mint Grid (1m)")
        self.chk_grid.setChecked(True)
        
        self.chk_bbox = QtWidgets.QCheckBox("Vehicle Box")
        self.chk_bbox.setChecked(True)
        
        self.chk_magnet = QtWidgets.QCheckBox("Magnet Line")
        self.chk_magnet.setChecked(True)
        self.chk_magnet.setStyleSheet("color: blue; font-weight: bold;")
        
        self.chk_raw = QtWidgets.QCheckBox("Raw Radar Points")
        self.chk_raw.setChecked(True)

        g_vis.addWidget(self.chk_grid, 0, 0)
        g_vis.addWidget(self.chk_bbox, 0, 1)
        g_vis.addWidget(self.chk_magnet, 1, 0)
        g_vis.addWidget(self.chk_raw, 1, 1)
        gb_vis.setLayout(g_vis)
        vbox.addWidget(gb_vis)

        # (2) Step Size
        gb_step = QtWidgets.QGroupBox("2. Step Size")
        g_step = QtWidgets.QGridLayout()
        
        self.s_deg = QtWidgets.QDoubleSpinBox(); self.s_deg.setValue(0.5)
        self.s_trans = QtWidgets.QDoubleSpinBox(); self.s_trans.setValue(0.1)
        self.s_zoom = QtWidgets.QDoubleSpinBox(); self.s_zoom.setValue(1.0)
        
        self.s_deg.valueChanged.connect(lambda v: setattr(self, 'deg_step', v))
        self.s_trans.valueChanged.connect(lambda v: setattr(self, 'trans_step', v))
        self.s_zoom.valueChanged.connect(lambda v: self._on_zoom_change(v))

        g_step.addWidget(QtWidgets.QLabel("Rot(°)"), 0, 0); g_step.addWidget(self.s_deg, 0, 1)
        g_step.addWidget(QtWidgets.QLabel("Move(m)"), 1, 0); g_step.addWidget(self.s_trans, 1, 1)
        g_step.addWidget(QtWidgets.QLabel("Zoom"), 2, 0); g_step.addWidget(self.s_zoom, 2, 1)
        gb_step.setLayout(g_step)
        vbox.addWidget(gb_step)

        # (3) Guide
        gb_guide = QtWidgets.QGroupBox("Key Guide")
        l_guide = QtWidgets.QLabel(
            "ROTATION:\n  Q/A: X-axis (Pitch)\n  W/S: Y-axis (Yaw)\n  E/D: Z-axis (Roll)\n\n"
            "TRANSLATION:\n  R/F: X-axis (Left/Right)\n  T/G: Y-axis (Fwd/Back)\n  Y/H: Z-axis (Up/Down)\n\n"
            "Shift + Key: Fine Tuning"
        )
        l_guide.setStyleSheet("font-size: 16px; font-family: Consolas;")
        v_guide = QtWidgets.QVBoxLayout()
        v_guide.addWidget(l_guide)
        gb_guide.setLayout(v_guide)
        vbox.addWidget(gb_guide)

        # Buttons
        btn_save = QtWidgets.QPushButton("💾 SAVE Extrinsic")
        btn_save.setStyleSheet("background-color: #4CAF50; color: white;")
        btn_save.clicked.connect(self._save_extrinsic)
        
        btn_reset = QtWidgets.QPushButton("↺ RESET")
        btn_reset.setStyleSheet("background-color: #f44336; color: white;")
        btn_reset.clicked.connect(self._reset_T)

        vbox.addStretch()
        vbox.addWidget(btn_save)
        vbox.addWidget(btn_reset)

        # ---------------------------------------------------------
        # 3. Layout & Timer
        # ---------------------------------------------------------
        self.img_view = ImageCanvasViewer(self)
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.addWidget(self.img_view, stretch=1)
        main_layout.addWidget(ctrl_panel, stretch=0)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.update_view)
        self._timer.start(33)

        self.update_view()

    # -------------------------------------------------------------
    # [핵심 로직] 박스 중앙(Center) 기준 매칭
    # -------------------------------------------------------------
    def update_view(self):
        try:
            # 1. 데이터 가져오기 (latest_frame 사용)
            if self.gui.latest_frame is None: return
            raw_img = self.gui.latest_frame.get("cv_image")
            if raw_img is None: return

            # 레이더 데이터
            pts_r = self.gui.latest_frame.get("radar_points")
            dops  = self.gui.latest_frame.get("radar_doppler")

            disp = raw_img.copy()
            h, w = disp.shape[:2]
            
            K = self.gui.cam_K
            if K is None: return
            cx, cy = K[0, 2], K[1, 2]

            # 2. 재투영 (T_current 반영)
            uvs_man, valid_man = None, None
            num_points = 0

            if pts_r is not None and len(pts_r) > 0:
                num_points = len(pts_r)
                uvs_man, valid_man = project_points(K, self.T_current[:3, :3], self.T_current[:3, 3], pts_r)

            # 3. 그리드 (민트색, 1m)
            if self.chk_grid.isChecked():
                self._draw_grid_and_axis(disp, K, cx, cy)

            # 4. 차량 박스 & 대표점 매칭 (중앙 기준)
            if self.chk_bbox.isChecked() and self.gui.vis_objects:
                for obj in self.gui.vis_objects:
                    x1, y1, x2, y2 = obj['bbox']
                    # 차량 박스 (녹색)
                    cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # [수정됨] 타겟 포인트: 박스 정중앙 (Center)
                    # 기존 상단(y1)에서 중앙((y1+y2)/2)으로 변경
                    target_pt = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
                    
                    rep_pt = None
                    
                    if uvs_man is not None:
                        # 1) 박스 내부 점 필터링
                        in_box = (valid_man & 
                                  (uvs_man[:, 0] >= x1) & (uvs_man[:, 0] <= x2) & 
                                  (uvs_man[:, 1] >= y1) & (uvs_man[:, 1] <= y2))
                        idxs = np.where(in_box)[0]

                        if idxs.size > 0:
                            curr_pts = uvs_man[idxs]
                            
                            # 2) [핵심] 중앙점(target_pt)과 레이더 점들 사이의 거리 계산
                            dists = np.hypot(curr_pts[:, 0] - target_pt[0], curr_pts[:, 1] - target_pt[1])
                            
                            # 3) 중앙에 가장 가까운 상위 K개 점만 선택 (K=3)
                            # 이렇게 하면 "중앙"에 뭉쳐있는 레이더 점을 대표점으로 사용하게 됨
                            k = min(len(dists), 3)
                            sorted_indices = np.argsort(dists)
                            closest_indices = sorted_indices[:k]
                            
                            # 4) 선택된 점들의 평균 좌표 -> 대표점(노란점)
                            best_pts = curr_pts[closest_indices]
                            u_mean = np.mean(best_pts[:, 0])
                            v_mean = np.mean(best_pts[:, 1])
                            
                            rep_pt = (int(u_mean), int(v_mean))

                    # 시각화
                    if rep_pt is not None:
                        tgt_pt_int = (int(target_pt[0]), int(target_pt[1]))
                        
                        # 노란점(레이더), 하늘색점(영상 타겟)
                        cv2.circle(disp, rep_pt, 5, (0, 255, 255), -1) 
                        cv2.circle(disp, tgt_pt_int, 4, (255, 255, 0), -1)
                        
                        # 마그넷 라인 (중앙 <-> 레이더 중앙)
                        if self.chk_magnet.isChecked():
                            err = np.hypot(rep_pt[0]-tgt_pt_int[0], rep_pt[1]-tgt_pt_int[1])
                            # 가까우면 초록, 멀면 빨강
                            col = (0, 0, 255) if err > 20 else (0, 255, 0)
                            cv2.line(disp, tgt_pt_int, rep_pt, col, 2)

            # 5. Raw Points 표시
            if self.chk_raw.isChecked() and uvs_man is not None:
                for i in range(len(uvs_man)):
                    if not valid_man[i]: continue
                    u = (uvs_man[i, 0] - cx) * self.radar_zoom + cx
                    v = (uvs_man[i, 1] - cy) * self.radar_zoom + cy
                    
                    if not (0 <= u < w and 0 <= v < h): continue
                    
                    vel = dops[i] if (dops is not None and i < len(dops)) else 0.0
                    # 접근(빨강), 이탈(파랑), 정지(초록)
                    if vel < -0.5: c = (0, 0, 255)
                    elif vel > 0.5: c = (255, 0, 0)
                    else: c = (0, 255, 0)

                    cv2.circle(disp, (int(u), int(v)), 4, c, -1)

            # 디버그 텍스트
            cv2.putText(disp, f"Radar Pts: {num_points}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            self.img_view.update_image(disp)

        except Exception as e:
            print(f"[ManualCalib] Error: {e}")

    # -------------------------------------------------------------
    # Grid & Axis
    # -------------------------------------------------------------
    def _draw_grid_and_axis(self, img, K, cx, cy):
        # Axis
        pts_axis = np.array([[0,0,0], [3,0,0], [0,3,0], [0,0,3]])
        uvs, valid = project_points(K, self.T_current[:3,:3], self.T_current[:3,3], pts_axis)
        
        if valid[0]:
            o = (int((uvs[0,0]-cx)*self.radar_zoom+cx), int((uvs[0,1]-cy)*self.radar_zoom+cy))
            cols = [(0,0,255), (0,255,0), (255,0,0)]
            for i in range(1, 4):
                if valid[i]:
                    p = (int((uvs[i,0]-cx)*self.radar_zoom+cx), int((uvs[i,1]-cy)*self.radar_zoom+cy))
                    cv2.arrowedLine(img, o, p, cols[i-1], 3)

        # Mint Grid (1m)
        grid_z = -1.5
        lines = []
        for x in range(-20, 21, 1):
            lines.append([[x, 0, grid_z], [x, 80, grid_z]])
        for y in range(0, 81, 1):
            lines.append([[-20, y, grid_z], [20, y, grid_z]])
            
        mint_color = (180, 255, 180)
        
        for p_start, p_end in lines:
            pts = np.array([p_start, p_end])
            uvs, valid = project_points(K, self.T_current[:3,:3], self.T_current[:3,3], pts)
            if valid[0] and valid[1]:
                p1 = (int((uvs[0,0]-cx)*self.radar_zoom+cx), int((uvs[0,1]-cy)*self.radar_zoom+cy))
                p2 = (int((uvs[1,0]-cx)*self.radar_zoom+cx), int((uvs[1,1]-cy)*self.radar_zoom+cy))
                cv2.line(img, p1, p2, mint_color, 1)

    # -------------------------------------------------------------
    # Keyboard Controls
    # -------------------------------------------------------------
    def keyPressEvent(self, event):
        k = event.key()
        mod = event.modifiers()
        self.is_fine_mode = (mod & Qt.ShiftModifier)
        d_deg = 0.1 if self.is_fine_mode else self.deg_step
        d_mov = 0.01 if self.is_fine_mode else self.trans_step

        if k == Qt.Key_Q: self._adjust_rot(0, 1, d_deg)
        elif k == Qt.Key_A: self._adjust_rot(0, -1, d_deg)
        elif k == Qt.Key_W: self._adjust_rot(1, 1, d_deg)
        elif k == Qt.Key_S: self._adjust_rot(1, -1, d_deg)
        elif k == Qt.Key_E: self._adjust_rot(2, 1, d_deg)
        elif k == Qt.Key_D: self._adjust_rot(2, -1, d_deg)
        elif k == Qt.Key_R: self._adjust_trans(0, 1, d_mov)
        elif k == Qt.Key_F: self._adjust_trans(0, -1, d_mov)
        elif k == Qt.Key_T: self._adjust_trans(1, 1, d_mov)
        elif k == Qt.Key_G: self._adjust_trans(1, -1, d_mov)
        elif k == Qt.Key_Y: self._adjust_trans(2, 1, d_mov)
        elif k == Qt.Key_H: self._adjust_trans(2, -1, d_mov)
        elif k == Qt.Key_Z: self._on_zoom_change(self.radar_zoom - 0.1)
        elif k == Qt.Key_X: self._on_zoom_change(self.radar_zoom + 0.1)
        
        self.update_view()

    def _adjust_rot(self, axis, sign, step):
        rad = np.deg2rad(sign * step)
        c, s = np.cos(rad), np.sin(rad)
        if axis == 0: R_inc = np.array([[1,0,0],[0,c,-s],[0,s,c]])
        elif axis == 1: R_inc = np.array([[c,0,s],[0,1,0],[-s,0,c]])
        else: R_inc = np.array([[c,-s,0],[s,c,0],[0,0,1]])
        self.T_current[:3, :3] = self.T_current[:3, :3] @ R_inc
        self.acc_rot[axis] += sign * step

    def _adjust_trans(self, axis, sign, step):
        move_vec = np.zeros(3); move_vec[axis] = sign * step
        d_cam = self.T_current[:3, :3] @ move_vec
        self.T_current[:3, 3] += d_cam
        self.acc_trans[axis] += sign * step

    def _on_zoom_change(self, val):
        self.radar_zoom = max(0.1, min(5.0, val))
        self.s_zoom.blockSignals(True)
        self.s_zoom.setValue(self.radar_zoom)
        self.s_zoom.blockSignals(False)

    def _reset_T(self):
        self.T_current = self.T_init.copy()
        self.acc_rot[:] = 0; self.acc_trans[:] = 0
        self.update_view()

    def _save_extrinsic(self):
        data = { "R": self.T_current[:3, :3].tolist(), "t": self.T_current[:3, 3].flatten().tolist() }
        try:
            with open(self.gui.extrinsic_path, 'w') as f:
                json.dump(data, f, indent=4)
            QtWidgets.QMessageBox.information(self, "Success", "Extrinsic parameters saved!")
            self.gui.load_extrinsic()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

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
        self.rep_pt_mem = {} 
        self.pt_alpha = 0.15
        self.vel_gate_mps = 1.5 
        self.spatial_weight_sigma = 30.0 
        self.min_pts_for_rep = 2

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

        # 4. View Options
        gb_vis = QtWidgets.QGroupBox("4. View Options")
        v_vis = QtWidgets.QVBoxLayout()
        self.chk_show_poly = QtWidgets.QCheckBox("Show Lane Polygons"); self.chk_show_poly.setChecked(False)
        
        # [수정] 아이디와 속도 체크박스 분리
        self.chk_show_id = QtWidgets.QCheckBox("Show ID Text"); self.chk_show_id.setChecked(True)
        self.chk_show_speed = QtWidgets.QCheckBox("Show Speed Text"); self.chk_show_speed.setChecked(True)
        
        v_vis.addWidget(self.chk_show_poly)
        v_vis.addWidget(self.chk_show_id)
        v_vis.addWidget(self.chk_show_speed)

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

        # 5. Radar Speed Filters
        gb_radar_f = QtWidgets.QGroupBox("5. Radar Speed Filters")
        v_rf = QtWidgets.QVBoxLayout()
        self.chk_show_approach = QtWidgets.QCheckBox("Approaching (+)")
        self.chk_show_approach.setChecked(True)
        self.chk_show_recede = QtWidgets.QCheckBox("Receding (-)")
        self.chk_show_recede.setChecked(True)
        
        # New Toggle for Single Point
        self.chk_single_point = QtWidgets.QCheckBox("Show Representative Point Only")
        self.chk_single_point.setChecked(True)
        self.chk_single_point.setStyleSheet("color: blue; font-weight: bold;")

        self.chk_filter_low_speed = QtWidgets.QCheckBox("Low-Speed Noise Filter")
        self.chk_filter_low_speed.setChecked(True)

        row_thr = QtWidgets.QHBoxLayout()
        row_thr.addWidget(QtWidgets.QLabel("Min Speed (km/h):"))
        self.spin_min_speed_kmh = QtWidgets.QDoubleSpinBox()
        self.spin_min_speed_kmh.setDecimals(1)
        self.spin_min_speed_kmh.setRange(0.0, 80.0)
        self.spin_min_speed_kmh.setSingleStep(0.5)
        self.spin_min_speed_kmh.setValue(NOISE_MIN_SPEED_KMH)
        row_thr.addWidget(self.spin_min_speed_kmh, stretch=1)

        v_rf.addWidget(self.chk_show_approach)
        v_rf.addWidget(self.chk_show_recede)
        v_rf.addWidget(self.chk_filter_low_speed)
        v_rf.addWidget(self.chk_single_point)
        v_rf.addLayout(row_thr)

        gb_radar_f.setLayout(v_rf)
        vbox.addWidget(gb_radar_f)

        # 6. Image Brightness
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

        # 하단 여백 추가 (패널 아래쪽)
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
            radar_points_all = np.array(list(g))

            if radar_points_all.shape[0] > 0:
                radar_points = radar_points_all[:, :3]
                col_idx = 3
                radar_doppler = None

                if power_field:
                    col_idx += 1
                if doppler_field:
                    radar_doppler = radar_points_all[:, col_idx]
            else:
                radar_points, radar_doppler = None, None
        except:
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
            for obj in self.vis_objects:
                g_id = obj['id']
                x1, y1, x2, y2 = obj['bbox']
                
                # 차선 필터링
                cx, cy = (x1 + x2) // 2, y2
                target_lane = None
                if has_lane_filter:
                    for name, poly in active_lane_polys.items():
                        if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                            target_lane = name
                            break
                    if target_lane is None:
                        continue

                # ID 관리
                if target_lane and g_id not in self.global_to_local_ids:
                    self.lane_counters[target_lane] += 1
                    self.global_to_local_ids[g_id] = self.lane_counters[target_lane]
                label = f"No: {self.global_to_local_ids.get(g_id, g_id)} ({target_lane})" if target_lane else f"ID: {g_id}"
                
                # =============================================================
                # [수정 포인트 1] 타겟 포인트를 '박스 정중앙'으로 변경
                # (가려짐 발생 시 BBox가 줄어들므로, 줄어든 박스의 중앙 = 지붕 중앙)
                # =============================================================
                target_pt = (int((x1+x2)/2), int((y1+y2)/2)) 
                
                meas_kmh = None
                rep_pt = None
                rep_vel_raw = None
                score = 0

                # --- 속도 추정 ---
                if proj_uvs is not None and dop_raw is not None:
                    # 1) BBox 내부에 있는 레이더 점 인덱스 추출
                    in_box = (proj_valid & (proj_uvs[:, 0] >= x1) & (proj_uvs[:, 0] <= x2) & (proj_uvs[:, 1] >= y1) & (proj_uvs[:, 1] <= y2))
                    idxs = np.where(in_box)[0]
                    
                    if idxs.size > 0:
                        # =============================================================
                        # [수정 포인트 2] 도플러 속도 필터 (뒷차량 가려짐 해결의 핵심)
                        # - 앞뒤 차량이 겹쳐 보여도 속도가 다르면 분리해냅니다.
                        # - 박스 내 주된 속도(Median)와 다른 점들은 노이즈로 간주하고 버립니다.
                        # =============================================================
                        med_dop = np.median(dop_raw[idxs])
                        vel_mask = np.abs(dop_raw[idxs] - med_dop) < self.vel_gate_mps
                        valid_idxs = idxs[vel_mask] if np.any(vel_mask) else idxs
                        
                        # [수정] 상단 필터 제거 -> 전체 영역 사용
                        target_idxs = valid_idxs 
                        
                        # 3) 중앙 추종 로직 (K-Nearest to Center)
                        # 타겟 지점: BBox 정중앙
                        t_u, t_v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        
                        if len(target_idxs) > 0:
                            current_pts = proj_uvs[target_idxs]
                            # 타겟 지점(중앙)과 레이더 점들 간의 거리 계산
                            dists = np.hypot(current_pts[:, 0] - t_u, current_pts[:, 1] - t_v)
                            
                            # 가장 가까운 3개 점 선택 (밀도 무시하고 중앙 점 선택)
                            k = min(len(dists), 3)
                            sorted_indices = np.argsort(dists)
                            closest_indices = sorted_indices[:k]
                            
                            best_pts = current_pts[closest_indices]
                            meas_u = np.mean(best_pts[:, 0])
                            meas_v = np.mean(best_pts[:, 1])
                            
                            # 속도는 해당 점들의 중앙값 사용
                            rep_vel_raw = np.median(dop_raw[target_idxs[closest_indices]]) * 3.6

                            # 4) 시계열 스무딩 (EMA)
                            if g_id in self.rep_pt_mem:
                                prev_u, prev_v, last_ts = self.rep_pt_mem[g_id]
                                if now_ts - last_ts < 1.0:
                                    smooth_u = self.pt_alpha * meas_u + (1.0 - self.pt_alpha) * prev_u
                                    smooth_v = self.pt_alpha * meas_v + (1.0 - self.pt_alpha) * prev_v
                                else:
                                    smooth_u, smooth_v = meas_u, meas_v
                            else:
                                smooth_u, smooth_v = meas_u, meas_v

                            self.rep_pt_mem[g_id] = [smooth_u, smooth_v, now_ts]
                            rep_pt = (int(smooth_u), int(smooth_v))
                            
                            if abs(rep_vel_raw) >= self.spin_min_speed_kmh.value():
                                meas_kmh = abs(rep_vel_raw)

                            # 정확도 점수 (중앙 기준)
                            dist_px = np.hypot(target_pt[0]-rep_pt[0], target_pt[1]-rep_pt[1])
                            diag_len = np.hypot(x2-x1, y2-y1)
                            err_ratio = dist_px / max(1, diag_len)
                            score = max(0, min(100, 100 - (err_ratio * 200)))
                            acc_scores.append(score)

                # --- [강화된 속도 필터링 v2] ---
                # 1) Hard Limit Filter: 상한선을 120km/h로 낮춤 (130, 127 차단)
                if meas_kmh is not None:
                    if abs(meas_kmh) > 120.0:  
                        meas_kmh = None

                mem = self.vel_memory.get(g_id, {})
                last_vel = mem.get("vel_kmh")
                last_ts = mem.get("ts", -1.0)
                fail_cnt = mem.get("fail_count", 0) # 연속 실패 횟수 가져오기
                
                # 2) Jump Filter with Recovery (락다운 방지 로직)
                JUMP_THRESHOLD = 10.0  # 사용자 설정 (10km/h 차이)
                
                if meas_kmh is not None and last_vel is not None:
                    diff = abs(meas_kmh - last_vel)
                    
                    if diff > JUMP_THRESHOLD:
                        # 차이가 크면 일단 카운트 증가
                        fail_cnt += 1
                        
                        # [핵심] 연속으로 5번 이상(약 0.15초) 차이가 난다면?
                        # -> "내 메모리가 틀렸거나 차가 급가감속 중이다" -> 새로운 값 수용
                        if fail_cnt > 5:
                            # print(f"Recovering ID:{g_id} {last_vel}->{meas_kmh}")
                            fail_cnt = 0 # 리셋하고 현재 값(meas_kmh) 허용
                        else:
                            # 아직은 노이즈로 간주하고 무시
                            meas_kmh = None 
                    else:
                        # 정상 범위 내라면 카운트 초기화
                        fail_cnt = 0
                else:
                    # 비교 대상이 없으면 카운트 0
                    fail_cnt = 0

                # --- 스무딩 (EMA) ---
                vel_out = None
                
                if meas_kmh is not None:
                    # 정상 업데이트
                    alpha = self.spin_ema_alpha.value()
                    prev = mem.get("ema", meas_kmh)
                    vel_out = (alpha * meas_kmh) + ((1.0 - alpha) * prev)
                    
                    mem["ema"] = vel_out
                    mem["vel_kmh"] = vel_out
                    mem["ts"] = now_ts
                    mem["fail_count"] = 0 # 성공했으므로 실패 카운트 0 저장
                    self.vel_memory[g_id] = mem
                    
                elif last_vel is not None and (now_ts - last_ts) <= 2.0:
                    # 데이터가 튀어서(None) 들어왔을 때, 이전 값 유지
                    # 단, 실패 카운트는 저장해둬야 다음 루프에서 누적됨
                    mem["fail_count"] = fail_cnt 
                    self.vel_memory[g_id] = mem
                    vel_out = last_vel

                if meas_kmh is not None:
                    alpha = self.spin_ema_alpha.value()
                    prev = mem.get("ema", meas_kmh)
                    vel_out = (alpha * meas_kmh) + ((1.0 - alpha) * prev)
                    mem["ema"] = vel_out
                    mem["vel_kmh"] = vel_out
                    mem["ts"] = now_ts
                    self.vel_memory[g_id] = mem
                elif last_vel is not None and (now_ts - last_ts) <= 5.0:
                    vel_out = last_vel

                obj['vel'] = round(vel_out, 1) if vel_out is not None else float('nan')

                draw_items.append({
                    "obj": obj, "bbox": (x1, y1, x2, y2), "label": label,
                    "rep_pt": rep_pt, "rep_vel": rep_vel_raw, "target_pt": target_pt, "score": score
                })

            # 5. 전역 UI 업데이트
            if acc_scores:
                avg = sum(acc_scores) / len(acc_scores)
                self.bar_acc.setValue(int(avg))
                self.lbl_emoji.setText("😊" if avg >= 80 else "😐" if avg >= 50 else "😟")

            # 6. 그리기 (Draw)
            if not self.chk_single_point.isChecked() and proj_uvs is not None:
                for i in range(len(proj_uvs)):
                    if not proj_valid[i]: continue
                    u, v = int(proj_uvs[i, 0]), int(proj_uvs[i, 1])
                    spd = dop_raw[i] * 3.6
                    if abs(spd) < self.spin_min_speed_kmh.value(): continue
                    col = (0, 0, 255) if spd < -0.5 else (255, 0, 0) if spd > 0.5 else (0, 255, 0)
                    cv2.circle(disp, (u, v), 2, col, -1)

            for it in draw_items:
                x1, y1, x2, y2 = it["bbox"]
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # [수정] 노란 타겟 점 (중앙)
                cv2.circle(disp, it["target_pt"], 4, (0, 255, 255), -1) 

                if it["rep_pt"] is not None:
                    if self.chk_magnet.isChecked():
                        lc = (0, 255, 0) if it["score"] > 50 else (0, 0, 255)
                        cv2.line(disp, it["target_pt"], it["rep_pt"], lc, 2)
                    
                    if self.chk_single_point.isChecked():
                        rv = it["rep_vel"]
                        if rv is not None and abs(rv) >= self.spin_min_speed_kmh.value():
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
            except:
                pass

    def run_calibration(self):
        dlg = ManualCalibWindow(self)
        dlg.exec()

    def run_autocalibration(self):
        try:
            if hasattr(self, "extrinsic_proc") and self.extrinsic_proc is not None:
                if self.extrinsic_proc.state() != QtCore.QProcess.NotRunning:
                    print("[Extrinsic] already running...")
                    return

            self.extrinsic_proc = QtCore.QProcess(self)
            self.extrinsic_proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)

            self.txt_extrinsic_log.clear()
            self.txt_extrinsic_log.appendPlainText("[Extrinsic] START")
            self.pbar_extrinsic.setVisible(True)
            self.pbar_extrinsic.setRange(0, 0)

            ws = os.path.expanduser("~/motrex/catkin_ws")
            setup_bash = os.path.join(ws, "devel", "setup.bash")

            cmd = "bash"
            ros_cmd = (
                f"source {setup_bash} && "
                f"rosrun perception_test calibration_ex_node.py "
                f"_extrinsic_path:={self.extrinsic_path}"
            )
            args = ["-lc", ros_cmd]
            self.extrinsic_proc.readyReadStandardOutput.connect(self._on_extrinsic_stdout)
            self.extrinsic_proc.readyReadStandardError.connect(self._on_extrinsic_stderr)
            self.extrinsic_proc.finished.connect(self._on_extrinsic_finished)
            self.extrinsic_proc.start(cmd, args)
        except Exception as e:
            print(f"[Extrinsic] start error: {e}")
            traceback.print_exc()

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
        try:
            rp = rospkg.RosPack()
            pkg_path = rp.get_path("perception_test")
            image_dir = os.path.join(pkg_path, "image")
            out_yaml = os.path.join(pkg_path, "config", "camera_intrinsic.yaml")

            if not os.path.isdir(image_dir):
                print(f"[Intrinsic] image dir not found: {image_dir}")
                return

            import glob
            imgs = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
            if len(imgs) == 0:
                print(f"[Intrinsic] No .jpg in {image_dir}")
                return

            if hasattr(self, "intrinsic_proc") and self.intrinsic_proc is not None:
                if self.intrinsic_proc.state() != QtCore.QProcess.NotRunning:
                    print("[Intrinsic] already running...")
                    return

            self.txt_intrinsic_log.clear()
            self.txt_intrinsic_log.appendPlainText("[Intrinsic] START")
            self.pbar_intrinsic.setVisible(True)
            self.pbar_intrinsic.setRange(0, 0)

            self.intrinsic_proc = QtCore.QProcess(self)
            self.intrinsic_proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)

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
            self.intrinsic_proc.readyReadStandardError.connect(self._on_intrinsic_stderr)
            self.intrinsic_proc.finished.connect(self._on_intrinsic_finished)
            self.intrinsic_proc.start(cmd, args)

        except Exception as e:
            print(f"[Intrinsic] start error: {e}")
            traceback.print_exc()

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