#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
real_gui.py (Rviz-Style Calibration Tool Integrated)

[기능 요약]
1. Main GUI: 카메라/레이더 퓨전 결과 확인, 차선 편집, 캘리브레이션 툴 실행
2. Manual Calibration: 
   - 점(Point) 대신 클러스터링된 박스(BBox) 표시
   - 속도별 색상 구분 (빨강:접근, 파랑:이탈)
   - 좌표축(Axis) 및 지면 격자(Grid) 표시
   - 직관적인 화면 기준(Screen-Space) 조작
"""

import sys
import os
import json
import time
import traceback
import numpy as np
import cv2
from collections import deque

# ==============================================================================
# [PARAMETER SETTINGS] 사용자 설정
# ==============================================================================

REFRESH_RATE_MS = 33           # 30 FPS
MAX_REASONABLE_KMH = 100.0   # 육교 고정형 기준 상한

# Radar Coloring & Filtering
SPEED_THRESHOLD_RED = -0.5     # m/s (이보다 작으면 접근/빨강)
SPEED_THRESHOLD_BLUE = 0.5     # m/s (이보다 크면 이탈/파랑)
NOISE_MIN_SPEED_KMH = 5.0      # 정지/저속(노이즈) 필터 기준 (km/h) — 권장 5km/h

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

START_ACTIVE_LANES = ["IN1", "IN2", "IN3"]

# ==============================================================================
# Setup & Imports
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

try:
    from PySide6 import QtWidgets, QtGui, QtCore
    from PySide6.QtCore import Qt
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
            # 실제로는 KD-Tree가 빠르지만, 점 개수가 수백 개 수준이면 브루트포스도 충분함
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
# ManualCalibWindow (Rviz-Style Visuals) - [핵심 수정 부분]
# ==============================================================================
class ManualCalibWindow(QtWidgets.QDialog):
    """
    Rviz 스타일의 시각화 도구를 포함한 수동 캘리브레이션 창.
    - Axis Gizmo: 레이더 원점과 축 방향 표시
    - BBox: 차량 클러스터링 및 속도별 색상(접근:빨강, 이탈:파랑) 표시
    - Grid: 지면(Z=0) 격자 표시로 Pitch/Roll 확인 용이
    - Screen-Space Controls: 직관적인 조작 (W/A/S/D)
    """
    def __init__(self, gui: 'RealWorldGUI'):
        super().__init__(gui)
        self.gui = gui
        self.setWindowTitle("Manual Calibration (Rviz Style Visuals)")
        self.setModal(True)
        self.resize(1700, 1000)

        # 1. State Init
        self.T_init = np.eye(4, dtype=np.float64)
        self.T_init[:3, :3] = np.array(self.gui.Extr_R, dtype=np.float64)
        self.T_init[:3, 3] = np.array(self.gui.Extr_t, dtype=np.float64).flatten()
        self.T_current = self.T_init.copy()

        # Settings
        self.deg_step = 0.5
        self.trans_step = 0.1
        self.radar_zoom = 1.0
        self.is_fine_mode = False

        # Accumulators (Display purpose)
        self.acc_rot = np.zeros(3)   # Pitch, Yaw, Roll
        self.acc_trans = np.zeros(3) # X, Y, Z

        # BEV
        self.lane_points = []; self.homography = None
        self.px_per_meter = 10.0; self.bev_size = (500, 1000)
        
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
        
        # Left Container (Views)
        left_cont = QtWidgets.QWidget()
        left_layout = QtWidgets.QHBoxLayout(left_cont); left_layout.setContentsMargins(0,0,0,0)
        self.bev_view = ImageCanvasViewer()
        self.img_view = ImageCanvasViewer()
        self.img_view.set_click_callback(self._on_image_click)
        left_layout.addWidget(self.bev_view, stretch=4)
        left_layout.addWidget(self.img_view, stretch=5)
        main_layout.addWidget(left_cont, stretch=5)

        # Right Container (Controls)
        ctrl_panel = QtWidgets.QWidget(); ctrl_panel.setFixedWidth(420)
        vbox = QtWidgets.QVBoxLayout(ctrl_panel)

        # (A) Visualization Options
        gb_vis = QtWidgets.QGroupBox("1. Visual Aids (Like Rviz)")
        f_vis = QtWidgets.QFormLayout()
        self.chk_axis = QtWidgets.QCheckBox("Show Radar Axis (RGB)"); self.chk_axis.setChecked(True)
        self.chk_grid = QtWidgets.QCheckBox("Show Ground Grid (10m)"); self.chk_grid.setChecked(True)
        self.chk_bbox = QtWidgets.QCheckBox("Show Moving Vehicles (Box)"); self.chk_bbox.setChecked(True)
        self.chk_raw = QtWidgets.QCheckBox("Show Raw Points"); self.chk_raw.setChecked(True)
        
        self.chk_hide_static = QtWidgets.QCheckBox(f"Hide Static (|v| < {NOISE_MIN_SPEED_KMH:.1f} km/h)")
        self.chk_hide_static.setChecked(False)

        # speed threshold (km/h) for hiding static/slow points/clusters
        self.min_speed_kmh = float(NOISE_MIN_SPEED_KMH)
        self.s_min_kmh = QtWidgets.QDoubleSpinBox(); self.s_min_kmh.setRange(0.0, 200.0)
        self.s_min_kmh.setValue(self.min_speed_kmh); self.s_min_kmh.setSingleStep(0.5)
        self.s_min_kmh.valueChanged.connect(self._on_min_speed_changed)

        self.chk_axis.toggled.connect(self.update_view)
        self.chk_grid.toggled.connect(self.update_view)
        self.chk_bbox.toggled.connect(self.update_view)
        self.chk_raw.toggled.connect(self.update_view)
        
        self.chk_hide_static.toggled.connect(self.update_view)
        f_vis.addRow(self.chk_axis, self.chk_grid)
        f_vis.addRow(self.chk_bbox, self.chk_raw)
        f_vis.addRow(self.chk_hide_static, self.s_min_kmh)
        gb_vis.setLayout(f_vis); vbox.addWidget(gb_vis)

        # (B) Control Values
        gb_step = QtWidgets.QGroupBox("2. Adjustments")
        f_step = QtWidgets.QFormLayout()
        self.s_deg = QtWidgets.QDoubleSpinBox(); self.s_deg.setRange(0.01, 10.0); self.s_deg.setValue(self.deg_step)
        self.s_deg.valueChanged.connect(lambda v: setattr(self, 'deg_step', v))
        self.s_trans = QtWidgets.QDoubleSpinBox(); self.s_trans.setRange(0.001, 1.0); self.s_trans.setValue(self.trans_step)
        self.s_trans.valueChanged.connect(lambda v: setattr(self, 'trans_step', v))
        self.s_zoom = QtWidgets.QDoubleSpinBox(); self.s_zoom.setRange(0.1, 5.0); self.s_zoom.setValue(1.0)
        self.s_zoom.valueChanged.connect(lambda v: setattr(self, 'radar_zoom', v)) # Directly bind
        
        f_step.addRow("Rot Step(°):", self.s_deg)
        f_step.addRow("Move Step(m):", self.s_trans)
        f_step.addRow("Zoom:", self.s_zoom)
        gb_step.setLayout(f_step); vbox.addWidget(gb_step)

        # (B2) Radar Clustering Params (for stable boxes)
        gb_clu = QtWidgets.QGroupBox("2.5 Radar Clustering")
        f_clu = QtWidgets.QFormLayout()
        self.s_clu_dist = QtWidgets.QDoubleSpinBox(); self.s_clu_dist.setRange(0.3, 10.0); self.s_clu_dist.setValue(CLUSTER_MAX_DIST); self.s_clu_dist.setSingleStep(0.1)
        self.s_clu_minpts = QtWidgets.QSpinBox(); self.s_clu_minpts.setRange(1, 50); self.s_clu_minpts.setValue(CLUSTER_MIN_PTS)
        self.s_clu_velgate = QtWidgets.QDoubleSpinBox(); self.s_clu_velgate.setRange(0.0, 50.0); self.s_clu_velgate.setValue(CLUSTER_VEL_GATE_MPS); self.s_clu_velgate.setSingleStep(0.2)
        self.s_bbox_area = QtWidgets.QSpinBox(); self.s_bbox_area.setRange(0, 20000); self.s_bbox_area.setValue(BBOX2D_MIN_AREA_PX2); self.s_bbox_area.setSingleStep(50)
        self.s_bbox_margin = QtWidgets.QSpinBox(); self.s_bbox_margin.setRange(0, 50); self.s_bbox_margin.setValue(BBOX2D_MARGIN_PX); self.s_bbox_margin.setSingleStep(1)

        for w in [self.s_clu_dist, self.s_clu_minpts, self.s_clu_velgate, self.s_bbox_area, self.s_bbox_margin]:
            try:
                w.valueChanged.connect(self.update_view)
            except Exception:
                pass

        f_clu.addRow("max_dist (m):", self.s_clu_dist)
        f_clu.addRow("min_pts:", self.s_clu_minpts)
        f_clu.addRow("vel_gate (m/s):", self.s_clu_velgate)
        f_clu.addRow("bbox min area (px^2):", self.s_bbox_area)
        f_clu.addRow("bbox margin (px):", self.s_bbox_margin)
        gb_clu.setLayout(f_clu); vbox.addWidget(gb_clu)


        # (C) Status Display
        gb_stat = QtWidgets.QGroupBox("3. State")
        g_stat = QtWidgets.QGridLayout()
        
        self.lbl_mode = QtWidgets.QLabel("Mode: Normal")
        self.lbl_mode.setStyleSheet("color: blue; font-weight: bold;")
        g_stat.addWidget(self.lbl_mode, 0, 0, 1, 2)
        
        # Values (Red)
        red_style = "color: #FF0000; font-weight: bold;"
        rows = ["Rot X (deg)", "Rot Y (deg)", "Rot Z (deg)", "Trans X (m)", "Trans Y (m)", "Trans Z (m)"]
        self.v_labels = [QtWidgets.QLabel("0.00") for _ in range(6)]
        
        for i, name in enumerate(rows):
            l = QtWidgets.QLabel(name + ":"); l.setStyleSheet(red_style)
            self.v_labels[i].setStyleSheet(red_style)
            g_stat.addWidget(l, i+1, 0)
            g_stat.addWidget(self.v_labels[i], i+1, 1)
            
        gb_stat.setLayout(g_stat); vbox.addWidget(gb_stat)

        # (D) Actions
        h_act = QtWidgets.QHBoxLayout()
        btn_rst = QtWidgets.QPushButton("Reset"); btn_rst.clicked.connect(self._reset_T)
        btn_sv = QtWidgets.QPushButton("Save"); btn_sv.setStyleSheet("background-color:#007700; color:white; font-weight:bold")
        btn_sv.clicked.connect(self._save_extrinsic)
        h_act.addWidget(btn_rst); h_act.addWidget(btn_sv)
        vbox.addLayout(h_act)
        # (E) Guide
        l_guide = QtWidgets.QLabel(
            "⌨️  Extrinsic Control (User Mapping)\n"
            "  Rot(+):  q(+x)  w(+y)  e(+z)\n"
            "  Rot(-):  a(-x)  s(-y)  d(-z)\n"
            "  Trans(+): r(+x)  t(+y)  y(+z)\n"
            "  Trans(-): f(-x)  g(-y)  h(-z)\n\n"
            "🔍 Zoom: [Z] Out / [X] In\n"
            "⚡ Fine: Hold [Shift]"
        )
        l_guide.setStyleSheet("color: #333333;")
        vbox.addWidget(l_guide)
        
        vbox.addStretch()
        main_layout.addWidget(ctrl_panel)

        self.timer = QtCore.QTimer(); self.timer.timeout.connect(self.update_view); self.timer.start(50)
        self._load_homography_json(silent=True)
        self.setFocusPolicy(Qt.StrongFocus)

    # ------------------------------------------------
    # Controls (Screen Relative)
    # ------------------------------------------------
    def keyPressEvent(self, event):
        k = event.key()
        mod = event.modifiers()
        self.is_fine_mode = (mod & Qt.ShiftModifier)
        deg = 0.05 if self.is_fine_mode else self.deg_step
        trans = 0.01 if self.is_fine_mode else self.trans_step
        self.lbl_mode.setText("Mode: FINE (Precise)" if self.is_fine_mode else "Mode: Normal")

        # ------------------------------------------------
        # Keyboard mapping (User table)
        # Rotation (degree):
        #   +X: Q, -X: A
        #   +Y: W, -Y: S
        #   +Z: E, -Z: D
        # Translation (meter):
        #   +X: R, -X: F
        #   +Y: T, -Y: G
        #   +Z: Y, -Z: H
        # ------------------------------------------------

        # Rotation
        if k == Qt.Key_Q: self._rot(0, +1, deg)
        elif k == Qt.Key_A: self._rot(0, -1, deg)
        elif k == Qt.Key_W: self._rot(1, +1, deg)
        elif k == Qt.Key_S: self._rot(1, -1, deg)
        elif k == Qt.Key_E: self._rot(2, +1, deg)
        elif k == Qt.Key_D: self._rot(2, -1, deg)

        # Translation
        elif k == Qt.Key_R: self._mov(0, +1, trans)
        elif k == Qt.Key_F: self._mov(0, -1, trans)
        elif k == Qt.Key_T: self._mov(1, +1, trans)
        elif k == Qt.Key_G: self._mov(1, -1, trans)
        elif k == Qt.Key_Y: self._mov(2, +1, trans)
        elif k == Qt.Key_H: self._mov(2, -1, trans)

        # Zoom
        elif k == Qt.Key_Z: self.radar_zoom = max(0.1, self.radar_zoom - 0.1)
        elif k == Qt.Key_X: self.radar_zoom += 0.1

        self.s_zoom.setValue(self.radar_zoom)
        self._update_stat_ui()
        self.update_view()
        
    def keyReleaseEvent(self, event):
        mod = event.modifiers()
        self.is_fine_mode = (mod & Qt.ShiftModifier)
        self.lbl_mode.setText("Mode: FINE (Precise)" if self.is_fine_mode else "Mode: Normal")
    def _rot(self, axis, sign, step):
        """Rotate ONLY the rotation matrix R in T_current (Radar->Camera)."""
        angle = sign * np.deg2rad(step)
        c, s = np.cos(angle), np.sin(angle)

        if axis == 0:
            dR = np.array([[1, 0, 0],
                           [0, c, -s],
                           [0, s,  c]], dtype=np.float64)
        elif axis == 1:
            dR = np.array([[ c, 0, s],
                           [ 0, 1, 0],
                           [-s, 0, c]], dtype=np.float64)
        else:
            dR = np.array([[c, -s, 0],
                           [s,  c, 0],
                           [0,  0, 1]], dtype=np.float64)

        # Keep translation t fixed; update rotation only.
        self.T_current[:3, :3] = self.T_current[:3, :3] @ dR
        self.acc_rot[axis] += (sign * step)
    def _mov(self, axis, sign, step):
        """Translate in radar-axis units (x,y,z) and convert to camera translation t."""
        d = np.zeros(3, dtype=np.float64)
        d[axis] = sign * step

        # If T maps radar->camera: p_cam = R p_radar + t,
        # then a delta in radar frame corresponds to dt = R * d in camera frame.
        dt = self.T_current[:3, :3] @ d
        self.T_current[:3, 3] = self.T_current[:3, 3] + dt
        self.acc_trans[axis] += (sign * step)

    def _on_min_speed_changed(self, v):
        self.min_speed_kmh = float(v)
        # update checkbox label to reflect threshold
        try:
            self.chk_hide_static.setText(f"Hide Static (|v| < {self.min_speed_kmh:.1f} km/h)")
        except Exception:
            pass
        self.update_view()

    def _update_stat_ui(self):
        vals = [
            f"{self.acc_rot[0]:+.2f}°", f"{self.acc_rot[1]:+.2f}°", f"{self.acc_rot[2]:+.2f}°",
            f"{self.acc_trans[0]:+.2f}m", f"{self.acc_trans[1]:+.2f}m", f"{self.acc_trans[2]:+.2f}m"
        ]
        for i, v in enumerate(vals): self.v_labels[i].setText(v)

    def _reset_T(self):
        self.T_current = self.T_init.copy(); self.acc_rot[:]=0; self.acc_trans[:]=0; self.radar_zoom=1.0
        self.s_zoom.setValue(1.0)
        self._update_stat_ui(); self.update_view()

    def _save_extrinsic(self):
        R = self.T_current[:3, :3]; t = self.T_current[:3, 3].reshape(3, 1)
        data = {"R": R.tolist(), "t": t.flatten().tolist()}
        with open(self.gui.extrinsic_path, 'w') as f: json.dump(data, f, indent=4)
        self.gui.load_extrinsic()
        QtWidgets.QMessageBox.information(self, "Saved", "Extrinsic Updated!")

    # ------------------------------------------------
    # Visualization (Gizmo, Box, Grid)
    # ------------------------------------------------
    def update_view(self):
        cv_img = self.gui.cv_image
        if cv_img is None: return
        disp = cv_img.copy()
        h, w = disp.shape[:2]
        K = self.gui.cam_K
        cx, cy = (K[0,2], K[1,2]) if K is not None else (w/2, h/2)
        
        # Center for Zoom (Optical Center)
        cx_opt, cy_opt = (K[0,2], K[1,2]) if K is not None else (w/2, h/2)

        # 1. BEV
        if self.homography is not None:
            bev = cv2.warpPerspective(disp, self.homography, self.bev_size)
        else:
            bev = np.zeros((self.bev_size[1], self.bev_size[0], 3), dtype=np.uint8)
        self._draw_bev_overlays(bev)
        
        # 2. Draw Helpers
        if K is not None:
            if self.chk_grid.isChecked(): self._draw_grid(disp, K, cx_opt, cy_opt)
            if self.chk_axis.isChecked(): self._draw_axis(disp, K, cx_opt, cy_opt)

        # 3. Radar Processing
        if self.gui.radar_points is not None and K is not None:
            pts_r = self.gui.radar_points # N x 3
            dopplers = self.gui.radar_doppler
            
            # --- CLUSTERING (Raw Radar Frame) ---
            # Clustering in raw frame is safer as it's physically grounded
            clusters = cluster_radar_points(
                pts_r,
                dopplers,
                max_dist=float(getattr(self, 's_clu_dist', None).value() if hasattr(self,'s_clu_dist') else CLUSTER_MAX_DIST),
                min_pts=int(getattr(self, 's_clu_minpts', None).value() if hasattr(self,'s_clu_minpts') else CLUSTER_MIN_PTS),
                vel_gate_mps=float(getattr(self, 's_clu_velgate', None).value() if hasattr(self,'s_clu_velgate') else CLUSTER_VEL_GATE_MPS),
            )
            
            # Draw BBox
            if self.chk_bbox.isChecked():
                for c in clusters:
                    self._draw_bbox_2d(disp, c, pts_r, dopplers, K, cx_opt, cy_opt)

            # Draw Raw Points
            if self.chk_raw.isChecked():
                self._draw_raw_points(disp, pts_r, dopplers, K, cx_opt, cy_opt, w, h)
                
            # BEV Points (Transform only valid ones)
            # (Simplification: Just draw raw points on BEV for reference)
            # Use current T to project to camera, then homography
            pass 

        # 4. Lane Picker
        for idx, (x, y) in enumerate(self.lane_points):
            cv2.circle(disp, (int(x), int(y)), 6, (0, 165, 255), -1)
            cv2.putText(disp, str(idx+1), (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)

        self.img_view.update_image(disp)
        self.bev_view.update_image(bev)

    def _draw_raw_points(self, img, pts_r, dopplers, K, cx, cy, w, h):
        # Transform all points: T * P_r
        pts_r_h = np.hstack((pts_r, np.ones((pts_r.shape[0], 1))))
        pts_c = (self.T_current @ pts_r_h.T).T # (N, 4)
        
        valid = pts_c[:, 2] > 0.5
        if not np.any(valid): return
        
        pts_c = pts_c[valid, :3]
        dops = dopplers[valid] if dopplers is not None else np.zeros(np.sum(valid))
        
        uvs = K @ pts_c.T
        uvs /= uvs[2, :]
        
        for i in range(uvs.shape[1]):
            u = (uvs[0, i] - cx) * self.radar_zoom + cx
            v = (uvs[1, i] - cy) * self.radar_zoom + cy
            
            # Color by Doppler
            vel = dops[i] # m/s
            if self.chk_hide_static.isChecked() and abs(vel) * 3.6 < float(getattr(self,'min_speed_kmh', NOISE_MIN_SPEED_KMH)):
                continue
            if abs(vel) < 0.2: color = (180, 180, 180) # Static/Noise
            elif vel < SPEED_THRESHOLD_RED: color = (0, 0, 255) # Approach
            elif vel > SPEED_THRESHOLD_BLUE: color = (255, 0, 0) # Recede
            else: color = (0, 255, 0) # Slow moving
            
            if 0<=u<w and 0<=v<h:
                cv2.circle(img, (int(u), int(v)), 2, color, -1)


    def _draw_bbox_2d(self, img, cluster, pts_r_all, dopplers_all, K, cx, cy):
        """클러스터 포인트를 이미지로 투영해서 2D bbox를 만든 뒤 그린다 (매칭 친화)."""
        if pts_r_all is None or K is None:
            return

        idx = cluster.get('indices', None)
        if idx is None or len(idx) == 0:
            return

        # Speed filter (노이즈 제거): |v| < threshold_kmh 은 표시하지 않음
        vel = float(cluster.get('vel', 0.0))
        if self.chk_hide_static.isChecked():
            if abs(vel) * 3.6 < float(getattr(self, 'min_speed_kmh', NOISE_MIN_SPEED_KMH)):
                return

        # Slow-green box 제거 목적: 접근/이탈만 보여주기
        if (vel >= SPEED_THRESHOLD_RED) and (vel <= SPEED_THRESHOLD_BLUE):
            # (이 구간은 5km/h 필터를 쓰면 대부분 걸러지지만, 안전하게 한 번 더)
            return

        pts_r = pts_r_all[np.array(idx, dtype=np.int64)]
        # Radar->Camera
        pts_h = np.hstack((pts_r, np.ones((pts_r.shape[0], 1), dtype=np.float64)))
        pts_c = (self.T_current @ pts_h.T).T  # (M,4)

        valid = pts_c[:, 2] > 0.5
        if not np.any(valid):
            return
        pts_c = pts_c[valid, :3]

        uv = (K @ pts_c.T)
        uv /= uv[2, :]
        u = (uv[0, :] - cx) * self.radar_zoom + cx
        v = (uv[1, :] - cy) * self.radar_zoom + cy

        h, w = img.shape[:2]
        in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if np.count_nonzero(in_img) < 2:
            return
        u = u[in_img]
        v = v[in_img]

        umin, umax = float(np.min(u)), float(np.max(u))
        vmin, vmax = float(np.min(v)), float(np.max(v))

        margin = int(getattr(self, 's_bbox_margin', None).value() if hasattr(self, 's_bbox_margin') else BBOX2D_MARGIN_PX)
        umin -= margin; umax += margin
        vmin -= margin; vmax += margin

        # Clamp
        umin = max(0, min(w-1, int(round(umin))))
        umax = max(0, min(w-1, int(round(umax))))
        vmin = max(0, min(h-1, int(round(vmin))))
        vmax = max(0, min(h-1, int(round(vmax))))

        if umax <= umin or vmax <= vmin:
            return

        area = (umax - umin) * (vmax - vmin)
        min_area = int(getattr(self, 's_bbox_area', None).value() if hasattr(self, 's_bbox_area') else BBOX2D_MIN_AREA_PX2)
        if area < min_area:
            return

        # Color: 접근(vel<0) 빨강, 이탈(vel>0) 파랑
        color = (0, 0, 255) if vel < 0 else (255, 0, 0)

        cv2.rectangle(img, (umin, vmin), (umax, vmax), color, BBOX_LINE_THICKNESS)
        cv2.putText(img, f"{abs(vel*3.6):.0f} km/h", (umin, max(0, vmin-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _draw_bbox_3d(self, img, cluster, K, cx, cy):
        # Cluster info is in Radar Frame. Need to transform min/max to Camera Frame
        # Construct 8 corners in Radar Frame
        min_p = cluster['min']
        max_p = cluster['max']
        
        # Simple Axis Aligned BBox in Radar Frame
        corners = np.array([
            [min_p[0], min_p[1], min_p[2]],
            [max_p[0], min_p[1], min_p[2]],
            [max_p[0], max_p[1], min_p[2]],
            [min_p[0], max_p[1], min_p[2]],
            [min_p[0], min_p[1], max_p[2]],
            [max_p[0], min_p[1], max_p[2]],
            [max_p[0], max_p[1], max_p[2]],
            [min_p[0], max_p[1], max_p[2]]
        ])
        
        # Transform to Camera
        corners_h = np.hstack((corners, np.ones((8, 1))))
        corners_c = (self.T_current @ corners_h.T).T # (8, 4)
        
        if np.any(corners_c[:, 2] < 0.5): return # Behind camera
        
        # Project
        uvs = K @ corners_c[:, :3].T
        uvs /= uvs[2, :]
        
        px = []
        for i in range(8):
            u = (uvs[0, i] - cx) * self.radar_zoom + cx
            v = (uvs[1, i] - cy) * self.radar_zoom + cy
            px.append((int(u), int(v)))
            
        # Determine Color
        vel = cluster['vel']
        thr_kmh = float(getattr(self, 'min_speed_kmh', NOISE_MIN_SPEED_KMH))
        if self.chk_hide_static.isChecked() and abs(vel * 3.6) < thr_kmh: return  # Hide static clusters (optional)
        
        color = (0, 0, 255) if vel < SPEED_THRESHOLD_RED else (255, 0, 0) # Red/Blue
        if SPEED_THRESHOLD_RED <= vel <= SPEED_THRESHOLD_BLUE: color = (0, 255, 0)

        # Draw Lines (Wireframe)
        edges = [
            (0,1), (1,2), (2,3), (3,0), # Bottom
            (4,5), (5,6), (6,7), (7,4), # Top
            (0,4), (1,5), (2,6), (3,7)  # Sides
        ]
        
        # Check FOV
        h, w = img.shape[:2]
        in_fov = False
        for u, v in px:
            if 0<=u<w and 0<=v<h: in_fov = True; break
        if not in_fov: return

        for s, e in edges:
            cv2.line(img, px[s], px[e], color, 2)
            
        # Label
        top_center = px[4]
        cv2.putText(img, f"{vel*3.6:.0f}km/h", top_center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _draw_axis(self, img, K, cx, cy):
        # Draw Axis at Radar Origin (0,0,0)
        origin = np.array([0, 0, 0, 1])
        # Radar Frame (Project): X=Right, Y=Forward, Z=Up
        x_axis = np.array([AXIS_LENGTH, 0, 0, 1])
        y_axis = np.array([0, AXIS_LENGTH, 0, 1])
        z_axis = np.array([0, 0, AXIS_LENGTH, 1])
        
        pts = np.vstack([origin, x_axis, y_axis, z_axis])
        pts_c = (self.T_current @ pts.T).T
        
        if pts_c[0, 2] < 0.5: return
        
        uvs = K @ pts_c[:, :3].T
        uvs /= uvs[2, :]
        
        px = []
        for i in range(4):
            u = (uvs[0, i] - cx)*self.radar_zoom + cx
            v = (uvs[1, i] - cy)*self.radar_zoom + cy
            px.append((int(u), int(v)))
            
        o = px[0]
        cv2.arrowedLine(img, o, px[1], (0, 0, 255), 3) # Red X
        cv2.arrowedLine(img, o, px[2], (0, 255, 0), 3) # Green Y
        cv2.arrowedLine(img, o, px[3], (255, 0, 0), 3) # Blue Z
        cv2.putText(img, "RADAR", o, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    def _draw_grid(self, img, K, cx, cy):
        # Draw Ground Grid (Z = -1.5m roughly in Radar Frame)
        # Project Radar Frame: X=Right, Y=Forward, Z=Up
        z_h = -1.5

        xs = np.linspace(-20, 20, 9)   # left(-) to right(+)
        ys = np.linspace(0, 100, 11)   # near(0) to far(100m forward)

        # Lines along Forward (Y)
        for x in xs:
            p1 = np.array([x, 0, z_h, 1])
            p2 = np.array([x, 100, z_h, 1])
            self._draw_line_3d(img, p1, p2, K, cx, cy, (100, 100, 0))

        # Lines along Right (X)
        for y in ys:
            p1 = np.array([-20, y, z_h, 1])
            p2 = np.array([20, y, z_h, 1])
            self._draw_line_3d(img, p1, p2, K, cx, cy, (100, 100, 0))

    def _draw_line_3d(self, img, p1, p2, K, cx, cy, color):
        pts = np.vstack([p1, p2])
        pts_c = (self.T_current @ pts.T).T
        if pts_c[0, 2] < 0.5 or pts_c[1, 2] < 0.5: return
        
        uvs = K @ pts_c[:, :3].T
        uvs /= uvs[2, :]
        u1 = (uvs[0, 0]-cx)*self.radar_zoom + cx; v1 = (uvs[1, 0]-cy)*self.radar_zoom + cy
        u2 = (uvs[0, 1]-cx)*self.radar_zoom + cx; v2 = (uvs[1, 1]-cy)*self.radar_zoom + cy
        try:
            cv2.line(img, (int(u1), int(v1)), (int(u2), int(v2)), color, 1)
        except: pass

    def _draw_bev_overlays(self, bev):
        h, w = bev.shape[:2]; cx = w//2; px = int(self.px_per_meter)
        for i in range(0, h, px*10):
            cv2.line(bev, (0, h-i), (w, h-i), (50,50,50), 1)
            if i>0: cv2.putText(bev, f"{i//px}m", (5, h-i-5), 0, 0.5, (100,100,100), 1)
        lx = int(cx - 1.75*px); rx = int(cx + 1.75*px)
        cv2.line(bev, (lx, 0), (lx, h), (255,0,0), 2)
        cv2.line(bev, (rx, 0), (rx, h), (255,0,0), 2)
        cv2.line(bev, (cx, 0), (cx, h), (80,80,80), 1)

    # BEV Setup
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


# ==============================================================================
# Main GUI
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

        # ---------------- Speed estimation memory / smoothing ----------------
        # track_id -> {'vel_kmh': float, 'ts': float, 'ema': float, 'kf': ScalarKalman}
        self.vel_memory = {}
        self.vel_hold_sec_default = 10.0  # 포인트 잠깐 누락 시 속도 유지 시간(초)
        self.radar_frame_count = 0
        self.radar_warmup_frames = 8        # rosbag 시작 튐 방지용 (권장 5~10)
        self.max_jump_kmh = 25.0            # 1프레임 점프 제한 (권장 20~30)
        self.use_nearest_k_points = True    # bbox 중심 근접 K점 방식
        self.nearest_k = 3                  # 기본 3점
        self.speed_update_period = 2   # N 프레임마다 한 번만 계산
        self._speed_frame_counter = 0
        self.speed_hard_max_kmh = 280.0   # 레이더 스펙 상한 (절대 상한)
        self.speed_soft_max_kmh = 100.0   # 육교 환경 차량 상한

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

        # ---------------- Speed Filters / Smoothing (Main GUI) ----------------
        gb_speed = QtWidgets.QGroupBox("4. Speed Estimation")
        v_s = QtWidgets.QVBoxLayout()

        row1 = QtWidgets.QHBoxLayout()
        row1.addWidget(QtWidgets.QLabel("Smoothing:"))
        self.cmb_smoothing = QtWidgets.QComboBox()
        self.cmb_smoothing.addItems(["None", "EMA", "Kalman"])
        self.cmb_smoothing.setCurrentText("EMA")
        row1.addWidget(self.cmb_smoothing, stretch=1)
        v_s.addLayout(row1)

        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel("EMA α:"))
        self.spin_ema_alpha = QtWidgets.QDoubleSpinBox()
        self.spin_ema_alpha.setDecimals(2)
        self.spin_ema_alpha.setRange(0.05, 0.95)
        self.spin_ema_alpha.setSingleStep(0.05)
        self.spin_ema_alpha.setValue(0.35)
        row2.addWidget(self.spin_ema_alpha, stretch=1)
        v_s.addLayout(row2)

        row3 = QtWidgets.QHBoxLayout()
        row3.addWidget(QtWidgets.QLabel("Hold (s):"))
        self.spin_hold_sec = QtWidgets.QDoubleSpinBox()
        self.spin_hold_sec.setDecimals(1)
        self.spin_hold_sec.setRange(0.0, 30.0)
        self.spin_hold_sec.setSingleStep(0.5)
        self.spin_hold_sec.setValue(10.0)
        row3.addWidget(self.spin_hold_sec, stretch=1)
        v_s.addLayout(row3)

        gb_speed.setLayout(v_s)
        vbox.addWidget(gb_speed)

        gb_radar_f = QtWidgets.QGroupBox("5. Radar Speed Filters")
        v_rf = QtWidgets.QVBoxLayout()
        self.chk_show_approach = QtWidgets.QCheckBox("Approaching (+)")
        self.chk_show_approach.setChecked(True)
        self.chk_show_recede = QtWidgets.QCheckBox("Receding (-)")
        self.chk_show_recede.setChecked(True)

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
        v_rf.addLayout(row_thr)

        gb_radar_f.setLayout(v_rf)
        vbox.addWidget(gb_radar_f)

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

        vbox.addStretch()
        self.lbl_log = QtWidgets.QLabel("System Ready")
        vbox.addWidget(self.lbl_log)
        layout.addWidget(panel, stretch=1)

    def reset_ids(self):
        self.lane_counters = {name: 0 for name in self.lane_counters.keys()}
        self.global_to_local_ids = {}
        self.lbl_log.setText("Local IDs Reset.")

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

        if radar_points is not None and radar_doppler is not None:
            self.radar_frame_count += 1 
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
        self._speed_frame_counter += 1
        do_speed_update = (self._speed_frame_counter % self.speed_update_period == 0)
        # Always use the latest frame. Pause/resume functionality removed.
        frame = self.latest_frame

        if frame is None:
            return

        # 다른 기능(차선 편집 등)에서 현재 프레임 참조가 필요할 수 있어 저장
        self.cv_image = frame["cv_image"]
        self.radar_points = frame["radar_points"]
        self.radar_doppler = frame["radar_doppler"]

        disp = frame["cv_image"].copy()
        if hasattr(self, "chk_brightness") and self.chk_brightness.isChecked():
            gain = float(self.spin_brightness_gain.value()) if hasattr(self, "spin_brightness_gain") else 1.0
            if abs(gain - 1.0) > 1e-3:
                disp = cv2.convertScaleAbs(disp, alpha=gain, beta=0)
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

        # ---------------- Velocity estimation from radar points inside bbox ----------------
        # 1) Project radar points once (Radar->Cam->Image)
        proj_uvs = None
        proj_valid = None
        proj_dopplers = None
        if radar_points is not None and self.cam_K is not None and radar_doppler is not None:
            uvs, valid = project_points(self.cam_K, self.Extr_R, self.Extr_t, radar_points)
            proj_uvs = uvs
            proj_valid = valid
            proj_dopplers = radar_doppler

        # Helper: apply GUI radar speed filters (sign + low-speed)
        def _pass_speed_filters(dop_mps: float) -> bool:
            # sign filter
            if dop_mps >= 0 and hasattr(self, "chk_show_approach") and (not self.chk_show_approach.isChecked()):
                return False
            if dop_mps < 0 and hasattr(self, "chk_show_recede") and (not self.chk_show_recede.isChecked()):
                return False
            # low-speed filter
            if hasattr(self, "chk_filter_low_speed") and self.chk_filter_low_speed.isChecked():
                thr = float(self.spin_min_speed_kmh.value()) if hasattr(self, "spin_min_speed_kmh") else NOISE_MIN_SPEED_KMH
                if abs(dop_mps * 3.6) < thr:
                    return False
            return True
        
        def _pass_estimation_filters(dop_mps: float) -> bool:
            # 속도 계산용은 토글(Approach/Recede)과 무관하게,
            # low-speed 노이즈만 최소로 제거하는 게 안전함
            if hasattr(self, "chk_filter_low_speed") and self.chk_filter_low_speed.isChecked():
                thr = float(self.spin_min_speed_kmh.value()) if hasattr(self, "spin_min_speed_kmh") else NOISE_MIN_SPEED_KMH
                if abs(dop_mps * 3.6) < thr:
                    return False
            return True

        # 2) Prepare objects to draw (lane filter applied), and compute local id labels
        draw_items = []
        active_lane_polys = {}
        for name, chk in self.chk_lanes.items():
            if not chk.isChecked():
                continue
            poly = self.lane_polys.get(name)
            if poly is not None and len(poly) > 2:
                active_lane_polys[name] = poly
        has_lane_filter = bool(active_lane_polys)

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

            draw_items.append({
                "obj": obj,
                "id": g_id,
                "bbox": (x1, y1, x2, y2),
                "label": label_prefix,
                "y2": y2
            })

        # 3) Handle overlapping bboxes: assign each radar point to the nearest(front-most) bbox only
        # Front-most in image ≈ larger y2 (lower in image)
        used_point_mask = None
        if proj_uvs is not None:
            used_point_mask = np.zeros(proj_uvs.shape[0], dtype=bool)

        # Sort bboxes front-to-back
        draw_items.sort(key=lambda d: d["y2"], reverse=True)

        now_ts = time.time()
        hold_sec = float(self.spin_hold_sec.value()) if hasattr(self, "spin_hold_sec") else self.vel_hold_sec_default
        smoothing = self.cmb_smoothing.currentText() if hasattr(self, "cmb_smoothing") else "None"
        ema_alpha = float(self.spin_ema_alpha.value()) if hasattr(self, "spin_ema_alpha") else 0.35

        for item in draw_items:
            g_id = item["id"]
            x1, y1, x2, y2 = item["bbox"]

            meas_kmh = None

            # ✅ (A) 먼저 mem/last 읽기 (순서 버그 수정)
            mem = self.vel_memory.get(g_id, {})
            last_ts = float(mem.get("ts", -1.0))
            last_vel = mem.get("vel_kmh", None)

            # ✅ (B) 무거운 레이더 기반 속도 계산은 주기적으로만 실행
            if do_speed_update:
                # If tracker doesn't provide speed yet, estimate from radar points within bbox
                if proj_uvs is not None and proj_dopplers is not None:
                    inside = (
                        proj_valid &
                        (proj_uvs[:, 0] >= x1) & (proj_uvs[:, 0] <= x2) &
                        (proj_uvs[:, 1] >= y1) & (proj_uvs[:, 1] <= y2)
                    )
                    if used_point_mask is not None:
                        inside = inside & (~used_point_mask)

                    idxs = np.where(inside)[0]
                    if idxs.size > 0:
                        dops = proj_dopplers[idxs].astype(np.float64, copy=False)

                        keep = np.array([_pass_estimation_filters(d) for d in dops], dtype=bool)
                        idxs = idxs[keep]
                        dops = dops[keep]

                        if idxs.size > 0:
                            if not hasattr(self, "radar_start_ts"):
                                self.radar_start_ts = now_ts

                            if (now_ts - self.radar_start_ts) < 0.5:
                                meas_kmh = None
                            else:
                                if self.use_nearest_k_points:
                                    cx_box = 0.5 * (x1 + x2)
                                    cy_box = 0.5 * (y1 + y2)

                                    du = proj_uvs[idxs, 0] - cx_box
                                    dv = proj_uvs[idxs, 1] - cy_box
                                    dist2 = du*du + dv*dv

                                    k = min(int(self.nearest_k), idxs.size)
                                    if dist2.size > k:
                                        pick = np.argpartition(dist2, k)[:k]
                                    else:
                                        pick = np.arange(dist2.size)

                                    dops_sel = dops[pick]
                                    dop_med = float(np.median(dops_sel))
                                    meas_kmh = abs(dop_med * 3.6)
                                else:
                                    fwd = radar_points[idxs, 1]
                                    sel_dops = select_best_forward_cluster(fwd, dops, max_gap_m=2.0)
                                    if sel_dops.size > 0:
                                        dop_med = float(np.median(sel_dops))
                                        meas_kmh = abs(dop_med * 3.6)
                                    else:
                                        meas_kmh = None

                        # Mark used points for overlap handling
                        if used_point_mask is not None and idxs.size > 0:
                            used_point_mask[idxs] = True
            else:
                # ✅ 이번 프레임은 레이더 기반 재계산 안함 → hold로 흘려보냄
                meas_kmh = None

            # ✅ (C) perception pipeline 값이 있으면 언제나 우선 (가벼움 → 매프레임 OK)
            obj_ref = item["obj"]
            if np.isfinite(obj_ref.get("vel", float("nan"))):
                meas_kmh = abs(float(obj_ref["vel"]))

            # ✅ (D) 점프 게이트 (last_vel 정의된 이후에 해야 함)
            if meas_kmh is not None and last_vel is not None and np.isfinite(last_vel):
                if abs(meas_kmh - float(last_vel)) > float(self.max_jump_kmh):
                    meas_kmh = None

            if meas_kmh is not None and np.isfinite(meas_kmh):
                if meas_kmh > MAX_REASONABLE_KMH:
                    meas_kmh = None   # ❗ 저장 자체를 막음

            # (D-2) last_vel 자체가 비정상(고정 문제)일 때는 메모리 리셋해서 hold가 못 살리게 함
            if last_vel is not None and np.isfinite(last_vel):
                if float(last_vel) > float(self.speed_soft_max_kmh):
                    # 127 같은 값이 한 번 들어가면 이후 측정이 막힐 때 hold로 고정되므로,
                    # soft max 이상이면 메모리 폐기
                    mem.pop("vel_kmh", None)
                    mem.pop("ema", None)
                    mem.pop("kf", None)
                    mem["ts"] = -1.0
                    last_vel = None
                    last_ts = -1.0
                    self.vel_memory[g_id] = mem

            # (D-3) 이번 측정값 자체도 절대 상한/현실 상한으로 컷
            if meas_kmh is not None and np.isfinite(meas_kmh):
                if float(meas_kmh) > float(self.speed_hard_max_kmh):
                    meas_kmh = None
                elif float(meas_kmh) > float(self.speed_soft_max_kmh):
                    # soft max는 "버림" 또는 "클램프" 중 선택 가능
                    meas_kmh = None

            # ✅ (E) Hold + smoothing (매프레임 OK)
            vel_out = None
            if meas_kmh is not None and np.isfinite(meas_kmh):
                if smoothing == "EMA":
                    prev = float(mem.get("ema", meas_kmh))
                    vel_f = (ema_alpha * meas_kmh) + ((1.0 - ema_alpha) * prev)
                    mem["ema"] = vel_f
                    vel_out = vel_f
                elif smoothing == "Kalman":
                    kf = mem.get("kf", None)
                    if kf is None:
                        kf = ScalarKalman(x0=meas_kmh, P0=25.0, Q=2.0, R=9.0)
                    kf.predict()
                    vel_out = kf.update(meas_kmh)
                    mem["kf"] = kf
                else:
                    vel_out = float(meas_kmh)

                mem["vel_kmh"] = float(vel_out)
                mem["ts"] = now_ts
                self.vel_memory[g_id] = mem
            else:
                if last_vel is not None and np.isfinite(last_vel):
                    if hold_sec <= 0.0 or (now_ts - last_ts) <= hold_sec:
                        vel_out = float(last_vel)

            if vel_out is not None and np.isfinite(vel_out):
                vel_out = float(np.round(abs(vel_out), 2))
                obj_ref["vel"] = vel_out
            else:
                obj_ref["vel"] = float("nan")


        # 5) Draw bboxes + labels
        for item in draw_items:
            obj = item["obj"]
            x1, y1, x2, y2 = item["bbox"]
            label_prefix = item["label"]

            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), BBOX_LINE_THICKNESS)

            vel = obj.get("vel", float("nan"))
            v_str = f"{vel:.2f}km/h" if np.isfinite(vel) else "--km/h"
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
                    
                    if 0 <= u < w and 0 <= v < h:
                        if has_doppler:
                            speed_kmh = dopplers[i] * 3.6
                            # Main GUI Radar Speed Filters
                            if hasattr(self, 'chk_filter_low_speed') and self.chk_filter_low_speed.isChecked():
                                thr_kmh = float(self.spin_min_speed_kmh.value())
                                if abs(speed_kmh) < thr_kmh:
                                    continue
                            if speed_kmh >= 0 and hasattr(self, 'chk_show_approach') and (not self.chk_show_approach.isChecked()):
                                continue
                            if speed_kmh < 0 and hasattr(self, 'chk_show_recede') and (not self.chk_show_recede.isChecked()):
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
            speed_text = f"{np.median(speed_values):.2f}km/h"

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
        """
        try:
            rp = rospkg.RosPack()
            pkg_path = rp.get_path("perception_test")
            image_dir = os.path.join(pkg_path, "image")
            out_yaml = os.path.join(pkg_path, "config", "camera_intrinsic.yaml")

            if not os.path.isdir(image_dir):
                self.lbl_log.setText(f"[Intrinsic] image dir not found: {image_dir}")
                return

            import glob
            imgs = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
            if len(imgs) == 0:
                self.lbl_log.setText(f"[Intrinsic] No .jpg in {image_dir}")
                return

            if hasattr(self, "intrinsic_proc") and self.intrinsic_proc is not None:
                if self.intrinsic_proc.state() != QtCore.QProcess.NotRunning:
                    self.lbl_log.setText("[Intrinsic] already running...")
                    return

            self.txt_intrinsic_log.clear()
            self.pbar_intrinsic.setVisible(True)
            self.pbar_intrinsic.setRange(0, 0)  # 돌아가는 중 표시
            self.lbl_log.setText("[Intrinsic] Running... (GUI에서 로그 확인)")

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
            self.intrinsic_proc.finished.connect(self._on_intrinsic_finished)

            self.intrinsic_proc.start(cmd, args)

        except Exception as e:
            self.lbl_log.setText(f"[Intrinsic] start error: {e}")
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
                K = ci.get("K", [])
                D = ci.get("D", [])
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