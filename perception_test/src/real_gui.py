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
from PySide6 import QtWidgets, QtCore, QtGui
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
# ManualCalibWindow (Rviz-Style Visuals)
# ==============================================================================
# ==============================================================================
# ManualCalibWindow (Rviz-Style Visuals) - [수정됨: 스타일 및 간격 통일]
# ==============================================================================
class ManualCalibWindow(QtWidgets.QDialog):
    def __init__(self, gui: 'RealWorldGUI'):
        super().__init__(gui)
        self.gui = gui
        self.setWindowTitle("Manual Calibration")
        self.setModal(True)
        
        # 초기 좌표 및 크기 설정
        self.setGeometry(2222, 183, 3234, 1764)

        # =========================================================
        # 1. State Init (Extrinsic)
        # =========================================================
        self.T_init = np.eye(4, dtype=np.float64)
        self.T_init[:3, :3] = np.array(self.gui.Extr_R, dtype=np.float64)
        self.T_init[:3, 3]  = np.array(self.gui.Extr_t, dtype=np.float64).flatten()
        self.T_current = self.T_init.copy()

        # Step / Mode
        self.deg_step     = 0.5
        self.trans_step   = 0.1
        self.radar_zoom   = 1.0
        self.is_fine_mode = False

        # Accumulators (display only)
        self.acc_rot   = np.zeros(3, dtype=np.float64)   # Rx, Ry, Rz (deg)
        self.acc_trans = np.zeros(3, dtype=np.float64)   # Tx, Ty, Tz (m)

        # Live refresh timer
        self._timer = QtCore.QTimer(self)
        self._timer.setTimerType(QtCore.Qt.PreciseTimer)
        self._timer.setInterval(REFRESH_RATE_MS)
        self._timer.timeout.connect(self.update_view)
        self._timer.start()
        
        # =========================================================
        # 2. Visual Toggle Widgets
        # =========================================================
        self.chk_axis = QtWidgets.QCheckBox("Show Axis")
        self.chk_axis.setChecked(True)

        self.chk_grid = QtWidgets.QCheckBox("Show Grid")
        self.chk_grid.setChecked(True)

        self.chk_bbox = QtWidgets.QCheckBox("Show BBox")
        self.chk_bbox.setChecked(True)

        self.chk_raw = QtWidgets.QCheckBox("Show Raw Points")
        self.chk_raw.setChecked(True)

        self.chk_hide_static = QtWidgets.QCheckBox(
            f"Hide Static (|v| < {NOISE_MIN_SPEED_KMH:.1f} km/h)"
        )
        self.chk_hide_static.setChecked(False)

        self.s_min_kmh = QtWidgets.QDoubleSpinBox()
        self.s_min_kmh.setRange(0.0, 200.0)
        self.s_min_kmh.setSingleStep(0.5)
        self.s_min_kmh.setValue(NOISE_MIN_SPEED_KMH)
        self.s_min_kmh.valueChanged.connect(self._on_min_speed_changed)

        self.min_speed_kmh = float(self.s_min_kmh.value())

        # =========================================================
        # 3. Adjustment Widgets
        # =========================================================
        self.s_deg = QtWidgets.QDoubleSpinBox()
        self.s_deg.setRange(0.01, 10.0)
        self.s_deg.setSingleStep(0.1)
        self.s_deg.setValue(self.deg_step)
        self.s_deg.valueChanged.connect(lambda v: setattr(self, "deg_step", v))

        self.s_trans = QtWidgets.QDoubleSpinBox()
        self.s_trans.setRange(0.001, 1.0)
        self.s_trans.setSingleStep(0.01)
        self.s_trans.setValue(self.trans_step)
        self.s_trans.valueChanged.connect(lambda v: setattr(self, "trans_step", v))

        self.s_zoom = QtWidgets.QDoubleSpinBox()
        self.s_zoom.setRange(0.1, 5.0)
        self.s_zoom.setSingleStep(0.1)
        self.s_zoom.setValue(self.radar_zoom)
        self.s_zoom.valueChanged.connect(lambda v: setattr(self, "radar_zoom", v))

        # =========================================================
        # 4. Radar Clustering Widgets
        # =========================================================
        self.s_clu_dist = QtWidgets.QDoubleSpinBox()
        self.s_clu_dist.setRange(0.3, 10.0)
        self.s_clu_dist.setSingleStep(0.1)
        self.s_clu_dist.setValue(CLUSTER_MAX_DIST)

        self.s_clu_minpts = QtWidgets.QSpinBox()
        self.s_clu_minpts.setRange(1, 50)
        self.s_clu_minpts.setValue(CLUSTER_MIN_PTS)

        self.s_clu_velgate = QtWidgets.QDoubleSpinBox()
        self.s_clu_velgate.setRange(0.0, 50.0)
        self.s_clu_velgate.setSingleStep(0.2)
        self.s_clu_velgate.setValue(CLUSTER_VEL_GATE_MPS)

        self.s_bbox_area = QtWidgets.QSpinBox()
        self.s_bbox_area.setRange(0, 20000)
        self.s_bbox_area.setSingleStep(50)
        self.s_bbox_area.setValue(BBOX2D_MIN_AREA_PX2)

        self.s_bbox_margin = QtWidgets.QSpinBox()
        self.s_bbox_margin.setRange(0, 50)
        self.s_bbox_margin.setSingleStep(1)
        self.s_bbox_margin.setValue(BBOX2D_MARGIN_PX)

        # =========================================================
        # 5. State Labels
        # =========================================================
        self.lbl_mode = QtWidgets.QLabel("Mode: Normal")
        self.lbl_mode.setStyleSheet("color: blue; font-weight: bold;")

        self.v_labels = [QtWidgets.QLabel("0.00") for _ in range(6)]
        for l in self.v_labels:
            l.setStyleSheet("color:#FF0000; font-weight:bold;")

        # =========================================================
        # 6. Config Path
        # =========================================================
        try:
            rp = rospkg.RosPack()
            self.config_dir = os.path.join(rp.get_path("perception_test"), "config")
        except Exception:
            self.config_dir = os.path.dirname(self.gui.extrinsic_path)

        os.makedirs(self.config_dir, exist_ok=True)

        # =========================================================
        # 7. Control Panel UI Build
        # =========================================================
        ctrl_panel = QtWidgets.QWidget()
        ctrl_panel.setFixedWidth(420)
        
        # [수정] Main GUI와 동일한 스타일 적용 (폰트 20px, GroupBox 여백 확장)
        ctrl_panel.setStyleSheet("""
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
        QDoubleSpinBox, QSpinBox {
            min-height:32px; padding:2px 6px;
            border:1px solid #d0d0d0; border-radius:6px;
        }
        QPushButton {
            min-height:36px; border-radius:8px;
            font-weight:700; background:#efefef;
        }
        """)

        vbox = QtWidgets.QVBoxLayout(ctrl_panel)
        vbox.setContentsMargins(10, 10, 10, 10)
        vbox.setSpacing(20) # [수정] 그룹 간 간격 넓힘 (10 -> 20)
        vbox.setAlignment(QtCore.Qt.AlignTop)

        # ================= (1) Visual Aids =================
        gb_vis = QtWidgets.QGroupBox("1. Visual Aids")
        g_vis = QtWidgets.QGridLayout()
        g_vis.setHorizontalSpacing(15)
        g_vis.setVerticalSpacing(12) # [수정] 내부 항목 간격 조정

        g_vis.addWidget(self.chk_axis, 0, 0)
        g_vis.addWidget(self.chk_grid, 0, 1)
        g_vis.addWidget(self.chk_bbox, 1, 0)
        g_vis.addWidget(self.chk_raw, 1, 1)
        g_vis.addWidget(self.chk_hide_static, 2, 0)
        g_vis.addWidget(self.s_min_kmh, 2, 1)

        gb_vis.setLayout(g_vis)
        vbox.addWidget(gb_vis)

        # ================= (2) Adjustments =================
        gb_step = QtWidgets.QGroupBox("2. Adjustments")
        g_step = QtWidgets.QGridLayout()
        g_step.setHorizontalSpacing(12)
        g_step.setVerticalSpacing(12) # [수정]

        g_step.addWidget(QtWidgets.QLabel("Rot Step (°)"), 0, 0)
        g_step.addWidget(self.s_deg, 0, 1)
        g_step.addWidget(QtWidgets.QLabel("Move Step (m)"), 1, 0)
        g_step.addWidget(self.s_trans, 1, 1)
        g_step.addWidget(QtWidgets.QLabel("Zoom"), 2, 0)
        g_step.addWidget(self.s_zoom, 2, 1)

        gb_step.setLayout(g_step)
        vbox.addWidget(gb_step)

        # ================= (3) Radar Clustering =================
        gb_clu = QtWidgets.QGroupBox("3. Radar Clustering")
        g_clu = QtWidgets.QGridLayout()
        g_clu.setHorizontalSpacing(12)
        g_clu.setVerticalSpacing(12) # [수정]

        labels = [
            ("max_dist (m)", self.s_clu_dist),
            ("min_pts", self.s_clu_minpts),
            ("vel_gate (m/s)", self.s_clu_velgate),
            ("bbox min area", self.s_bbox_area),
            ("bbox margin", self.s_bbox_margin),
        ]
        for i, (t, w_) in enumerate(labels):
            g_clu.addWidget(QtWidgets.QLabel(t), i, 0)
            g_clu.addWidget(w_, i, 1)

        gb_clu.setLayout(g_clu)
        vbox.addWidget(gb_clu)

        # ================= (4) State =================
        gb_stat = QtWidgets.QGroupBox("4. State")
        g_stat = QtWidgets.QGridLayout()
        g_stat.setHorizontalSpacing(12)
        g_stat.setVerticalSpacing(8) # [수정]

        g_stat.addWidget(self.lbl_mode, 0, 0, 1, 2)
        for i, name in enumerate(["Rot X", "Rot Y", "Rot Z", "Trans X", "Trans Y", "Trans Z"]):
            g_stat.addWidget(QtWidgets.QLabel(name), i + 1, 0)
            g_stat.addWidget(self.v_labels[i], i + 1, 1)

        gb_stat.setLayout(g_stat)
        vbox.addWidget(gb_stat)

        # ================= (E) Extrinsic Control (TABLE) =================
        # [수정] 키캡 스타일 크기 증가 (글자 크기 20px 대응)
        def _keycap(t, w=36): 
            k = QtWidgets.QLabel(t)
            k.setAlignment(QtCore.Qt.AlignCenter)
            k.setFixedSize(w, 30) 
            k.setStyleSheet("background:#2b2b2b;color:white;border-radius:6px;font-weight:800;font-size:16px;")
            return k

        def _vline():
            l = QtWidgets.QFrame()
            l.setFrameShape(QtWidgets.QFrame.VLine)
            l.setStyleSheet("color:#d0d0d0;")
            return l

        def _hline():
            l = QtWidgets.QFrame()
            l.setFrameShape(QtWidgets.QFrame.HLine)
            l.setStyleSheet("color:#d0d0d0;")
            return l

        gb_help = QtWidgets.QGroupBox("Extrinsic Control")
        g = QtWidgets.QGridLayout()
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(10) # [수정]

        # ---- headers ----
        for i, h in enumerate(["X", "Y", "Z"]):
            lb = QtWidgets.QLabel(h)
            lb.setAlignment(QtCore.Qt.AlignCenter)
            lb.setStyleSheet("font-weight:800;color:#555;")
            g.addWidget(lb, 0, 1 + i * 2)

        g.addWidget(_vline(), 0, 2, 6, 1)
        g.addWidget(_vline(), 0, 4, 6, 1)

        # ---- Rotation ----
        g.addWidget(QtWidgets.QLabel("Rot +"), 1, 0)
        g.addWidget(_keycap("Q"), 1, 1)
        g.addWidget(_keycap("W"), 1, 3)
        g.addWidget(_keycap("E"), 1, 5)

        g.addWidget(QtWidgets.QLabel("Rot −"), 2, 0)
        g.addWidget(_keycap("A"), 2, 1)
        g.addWidget(_keycap("S"), 2, 3)
        g.addWidget(_keycap("D"), 2, 5)

        g.addWidget(_hline(), 3, 0, 1, 6)

        # ---- Translation ----
        g.addWidget(QtWidgets.QLabel("Move +"), 4, 0)
        g.addWidget(_keycap("R"), 4, 1)
        g.addWidget(_keycap("T"), 4, 3)
        g.addWidget(_keycap("Y"), 4, 5)

        g.addWidget(QtWidgets.QLabel("Move −"), 5, 0)
        g.addWidget(_keycap("F"), 5, 1)
        g.addWidget(_keycap("G"), 5, 3)
        g.addWidget(_keycap("H"), 5, 5)

        g.addWidget(_hline(), 6, 0, 1, 6)

        # ---- Zoom ----
        g.addWidget(QtWidgets.QLabel("Zoom Out"), 7, 0)
        g.addWidget(_keycap("Z"), 7, 1, 1, 2)

        g.addWidget(QtWidgets.QLabel("Zoom In"), 8, 0)
        g.addWidget(_keycap("X"), 8, 1, 1, 2)

        g.addWidget(_hline(), 9, 0, 1, 6)

        # ---- Fine ----
        g.addWidget(QtWidgets.QLabel("Fine(Hold)"), 10, 0)
        g.addWidget(_keycap("Shift", 60), 10, 1, 1, 2)

        gb_help.setLayout(g)
        vbox.addWidget(gb_help)

        # Buttons
        h_btns = QtWidgets.QHBoxLayout()
        
        btn_save = QtWidgets.QPushButton("💾 Save Extrinsic")
        btn_save.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        btn_save.clicked.connect(self._save_extrinsic)

        btn_reset = QtWidgets.QPushButton("Reset")
        btn_reset.setStyleSheet("background-color: #f44336; color: white; padding: 10px;")
        btn_reset.clicked.connect(self._reset_T)

        h_btns.addWidget(btn_save)
        h_btns.addWidget(btn_reset)
        
        vbox.addLayout(h_btns)

        # 빈 공간 채우기
        vbox.addStretch(1)

        # =========================================================
        # 8. Viewer + Main Layout
        # =========================================================
        self.img_view = ImageCanvasViewer(self)
        self.img_view.setMinimumSize(1200, 800)

        main = QtWidgets.QHBoxLayout(self)
        main.setContentsMargins(8, 8, 8, 8)
        main.setSpacing(10)
        main.addWidget(self.img_view, stretch=1)
        main.addWidget(ctrl_panel, stretch=0)

        # 첫 렌더 강제
        self._update_stat_ui()
        QtCore.QTimer.singleShot(0, self.update_view)

    def closeEvent(self, e):
        try:
            if hasattr(self, "_timer") and self._timer.isActive():
                self._timer.stop()
        except Exception:
            pass
        super().closeEvent(e)

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

        self.T_current[:3, :3] = self.T_current[:3, :3] @ dR
        self.acc_rot[axis] += (sign * step)

    def _mov(self, axis, sign, step):
        """Translate in radar-axis units (x,y,z) and convert to camera translation t."""
        d = np.zeros(3, dtype=np.float64)
        d[axis] = sign * step
        dt = self.T_current[:3, :3] @ d
        self.T_current[:3, 3] = self.T_current[:3, 3] + dt
        self.acc_trans[axis] += (sign * step)

    def _on_min_speed_changed(self, v):
        self.min_speed_kmh = float(v)
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
        for i, v in enumerate(vals):
            self.v_labels[i].setText(v)

    def _reset_T(self):
        self.T_current = self.T_init.copy()
        self.acc_rot[:] = 0
        self.acc_trans[:] = 0
        self.radar_zoom = 1.0
        self.s_zoom.setValue(1.0)
        self._update_stat_ui()
        self.update_view()

    def _save_extrinsic(self):
        R = self.T_current[:3, :3]
        t = self.T_current[:3, 3].reshape(3, 1)
        data = {"R": R.tolist(), "t": t.flatten().tolist()}
        try:
            with open(self.gui.extrinsic_path, 'w') as f:
                json.dump(data, f, indent=4)
            self.gui.load_extrinsic()
            QtWidgets.QMessageBox.information(self, "Saved", "Extrinsic Updated!")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save:\n{e}")

    # ------------------------------------------------
    # Visualization (Gizmo, Box, Grid)
    # ------------------------------------------------
    def update_view(self):
        cv_img = self.gui.cv_image
        if cv_img is None:
            return

        disp = cv_img.copy()
        h, w = disp.shape[:2]
        K = self.gui.cam_K
        cx, cy = (K[0, 2], K[1, 2]) if K is not None else (w / 2, h / 2)
        cx_opt, cy_opt = (K[0, 2], K[1, 2]) if K is not None else (w / 2, h / 2)

        if K is not None:
            if self.chk_grid.isChecked():
                self._draw_grid(disp, K, cx_opt, cy_opt)
            if self.chk_axis.isChecked():
                self._draw_axis(disp, K, cx_opt, cy_opt)

        if self.gui.radar_points is not None and K is not None:
            pts_r = self.gui.radar_points
            dopplers = self.gui.radar_doppler

            clusters = cluster_radar_points(
                pts_r,
                dopplers,
                max_dist=float(self.s_clu_dist.value()),
                min_pts=int(self.s_clu_minpts.value()),
                vel_gate_mps=float(self.s_clu_velgate.value()),
            )

            if self.chk_bbox.isChecked():
                for c in clusters:
                    self._draw_bbox_2d(disp, c, pts_r, dopplers, K, cx_opt, cy_opt)

            if self.chk_raw.isChecked():
                self._draw_raw_points(disp, pts_r, dopplers, K, cx_opt, cy_opt, w, h)

        self.img_view.update_image(disp)

    def _draw_raw_points(self, img, pts_r, dopplers, K, cx, cy, w, h):
        pts_r_h = np.hstack((pts_r, np.ones((pts_r.shape[0], 1))))
        pts_c = (self.T_current @ pts_r_h.T).T

        valid = pts_c[:, 2] > 0.5
        if not np.any(valid):
            return

        pts_c = pts_c[valid, :3]
        dops = dopplers[valid] if dopplers is not None else np.zeros(np.sum(valid))

        uvs = K @ pts_c.T
        uvs /= uvs[2, :]

        for i in range(uvs.shape[1]):
            u = (uvs[0, i] - cx) * self.radar_zoom + cx
            v = (uvs[1, i] - cy) * self.radar_zoom + cy

            vel = dops[i]
            if self.chk_hide_static.isChecked() and abs(vel) * 3.6 < float(self.min_speed_kmh):
                continue

            if abs(vel) < 0.2:
                color = (180, 180, 180)
            elif vel < SPEED_THRESHOLD_RED:
                color = (0, 0, 255)
            elif vel > SPEED_THRESHOLD_BLUE:
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)

            if 0 <= u < w and 0 <= v < h:
                cv2.circle(img, (int(u), int(v)), 2, color, -1)

    def _draw_bbox_2d(self, img, cluster, pts_r_all, dopplers_all, K, cx, cy):
        if pts_r_all is None or K is None:
            return

        idx = cluster.get('indices', None)
        if idx is None or len(idx) == 0:
            return

        vel = float(cluster.get('vel', 0.0))
        if self.chk_hide_static.isChecked():
            if abs(vel) * 3.6 < float(self.min_speed_kmh):
                return

        if (vel >= SPEED_THRESHOLD_RED) and (vel <= SPEED_THRESHOLD_BLUE):
            return

        pts_r = pts_r_all[np.array(idx, dtype=np.int64)]
        pts_h = np.hstack((pts_r, np.ones((pts_r.shape[0], 1), dtype=np.float64)))
        pts_c = (self.T_current @ pts_h.T).T

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

        margin = int(self.s_bbox_margin.value())
        umin -= margin
        umax += margin
        vmin -= margin
        vmax += margin

        umin = max(0, min(w - 1, int(round(umin))))
        umax = max(0, min(w - 1, int(round(umax))))
        vmin = max(0, min(h - 1, int(round(vmin))))
        vmax = max(0, min(h - 1, int(round(vmax))))

        if umax <= umin or vmax <= vmin:
            return

        area = (umax - umin) * (vmax - vmin)
        min_area = int(self.s_bbox_area.value())
        if area < min_area:
            return

        color = (0, 0, 255) if vel < 0 else (255, 0, 0)
        
        # 폰트 사이즈 키움
        cv2.rectangle(img, (umin, vmin), (umax, vmax), color, BBOX_LINE_THICKNESS)
        cv2.putText(img, f"{abs(vel*3.6):.0f} km/h",
                    (umin, max(0, vmin - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def _draw_axis(self, img, K, cx, cy):
        origin = np.array([0, 0, 0, 1])
        x_axis = np.array([AXIS_LENGTH, 0, 0, 1])
        y_axis = np.array([0, AXIS_LENGTH, 0, 1])
        z_axis = np.array([0, 0, AXIS_LENGTH, 1])

        pts = np.vstack([origin, x_axis, y_axis, z_axis])
        pts_c = (self.T_current @ pts.T).T

        if pts_c[0, 2] < 0.5:
            return

        uvs = K @ pts_c[:, :3].T
        uvs /= uvs[2, :]

        px = []
        for i in range(4):
            u = (uvs[0, i] - cx) * self.radar_zoom + cx
            v = (uvs[1, i] - cy) * self.radar_zoom + cy
            px.append((int(u), int(v)))

        o = px[0]
        cv2.arrowedLine(img, o, px[1], (0, 0, 255), 3)
        cv2.arrowedLine(img, o, px[2], (0, 255, 0), 3)
        cv2.arrowedLine(img, o, px[3], (255, 0, 0), 3)
        cv2.putText(img, "RADAR", o, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def _draw_grid(self, img, K, cx, cy):
        z_h = -1.5

        xs = np.linspace(-20, 20, 9)
        ys = np.linspace(0, 100, 11)

        for x in xs:
            p1 = np.array([x, 0, z_h, 1])
            p2 = np.array([x, 100, z_h, 1])
            self._draw_line_3d(img, p1, p2, K, cx, cy, (100, 100, 0))

        for y in ys:
            p1 = np.array([-20, y, z_h, 1])
            p2 = np.array([20, y, z_h, 1])
            self._draw_line_3d(img, p1, p2, K, cx, cy, (100, 100, 0))

    def _draw_line_3d(self, img, p1, p2, K, cx, cy, color):
        pts = np.vstack([p1, p2])
        pts_c = (self.T_current @ pts.T).T
        if pts_c[0, 2] < 0.5 or pts_c[1, 2] < 0.5:
            return

        uvs = K @ pts_c[:, :3].T
        uvs /= uvs[2, :]

        u1 = (uvs[0, 0] - cx) * self.radar_zoom + cx
        v1 = (uvs[1, 0] - cy) * self.radar_zoom + cy
        u2 = (uvs[0, 1] - cx) * self.radar_zoom + cx
        v2 = (uvs[1, 1] - cy) * self.radar_zoom + cy

        try:
            cv2.line(img, (int(u1), int(v1)), (int(u2), int(v2)), color, 1)
        except Exception:
            pass

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

        # 2. Lane Editor
        gb_lane = QtWidgets.QGroupBox("2. Lane Editor")
        v_l = QtWidgets.QVBoxLayout()
        btn_edit = QtWidgets.QPushButton("🖌️ Open Editor")
        btn_edit.setStyleSheet("background-color: #FFD700; color: black; font-weight: bold; padding: 10px;")
        btn_edit.clicked.connect(self.open_lane_editor)
        v_l.addWidget(btn_edit)
        gb_lane.setLayout(v_l)
        vbox.addWidget(gb_lane)

        # 3. View Options
        gb_vis = QtWidgets.QGroupBox("3. View Options")
        v_vis = QtWidgets.QVBoxLayout()
        self.chk_show_poly = QtWidgets.QCheckBox("Show Lane Polygons (ROI)")
        self.chk_show_poly.setChecked(False)
        v_vis.addWidget(self.chk_show_poly)

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

        # 4. Speed Estimation
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

        # 5. Radar Speed Filters
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
        self._maybe_reload_extrinsic()
        self._speed_frame_counter += 1
        do_speed_update = (self._speed_frame_counter % self.speed_update_period == 0)
        frame = self.latest_frame

        if frame is None:
            return

        self.cv_image = frame["cv_image"]
        self.radar_points = frame["radar_points"]
        self.radar_doppler = frame["radar_doppler"]

        disp = frame["cv_image"].copy()
        if hasattr(self, "chk_brightness") and self.chk_brightness.isChecked():
            gain = float(self.spin_brightness_gain.value()) if hasattr(self, "spin_brightness_gain") else 1.0
            if abs(gain - 1.0) > 1e-3:
                disp = cv2.convertScaleAbs(disp, alpha=gain, beta=0)

        radar_points = frame["radar_points"]
        radar_doppler = frame["radar_doppler"]

        # 1. 차선 필터 설정 확인
        active_lane_polys = {}
        for name, chk in self.chk_lanes.items():
            if not chk.isChecked():
                continue
            poly = self.lane_polys.get(name)
            if poly is not None and len(poly) > 2:
                active_lane_polys[name] = poly
        has_lane_filter = bool(active_lane_polys)

        # 2. 레이더 투영 준비
        proj_uvs = None
        proj_valid = None
        proj_dopplers = None
        if radar_points is not None and self.cam_K is not None and radar_doppler is not None:
            uvs, valid = project_points(self.cam_K, self.Extr_R, self.Extr_t, radar_points)
            proj_uvs = uvs
            proj_valid = valid
            proj_dopplers = radar_doppler

        def _pass_estimation_filters(dop_mps: float) -> bool:
            if hasattr(self, "chk_filter_low_speed") and self.chk_filter_low_speed.isChecked():
                thr = float(self.spin_min_speed_kmh.value()) if hasattr(self, "spin_min_speed_kmh") else 5.0
                if abs(dop_mps * 3.6) < thr:
                    return False
            return True

        # 3. 대상 객체 정리
        draw_items = []
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

            label_prefix = f"No: {self.global_to_local_ids.get(g_id, g_id)} ({target_lane})" if target_lane else f"ID: {g_id}"
            
            # 차선 기반 카운팅 로직
            if target_lane and g_id not in self.global_to_local_ids:
                self.lane_counters[target_lane] += 1
                self.global_to_local_ids[g_id] = self.lane_counters[target_lane]
                label_prefix = f"No: {self.global_to_local_ids[g_id]} ({target_lane})"

            draw_items.append({
                "obj": obj, "id": g_id, "bbox": (x1, y1, x2, y2),
                "label": label_prefix, "y2": y2
            })

        # 4. 속도 추정 및 마스킹
        used_point_mask = np.zeros(proj_uvs.shape[0], dtype=bool) if proj_uvs is not None else None
        draw_items.sort(key=lambda d: d["y2"], reverse=True) # 아래쪽 차량부터 점 할당

        now_ts = time.time()
        hold_sec = float(self.spin_hold_sec.value()) if hasattr(self, "spin_hold_sec") else 10.0
        smoothing = self.cmb_smoothing.currentText() if hasattr(self, "cmb_smoothing") else "EMA"
        ema_alpha = float(self.spin_ema_alpha.value()) if hasattr(self, "spin_ema_alpha") else 0.35

        for item in draw_items:
            g_id = item["id"]
            x1, y1, x2, y2 = item["bbox"]
            meas_kmh = None
            mem = self.vel_memory.get(g_id, {})
            last_ts = float(mem.get("ts", -1.0))
            last_vel = mem.get("vel_kmh", None)

            if do_speed_update and proj_uvs is not None and proj_dopplers is not None:
                inside = (proj_valid & (proj_uvs[:, 0] >= x1) & (proj_uvs[:, 0] <= x2) & (proj_uvs[:, 1] >= y1) & (proj_uvs[:, 1] <= y2))
                if used_point_mask is not None:
                    inside = inside & (~used_point_mask)

                idxs = np.where(inside)[0]
                if idxs.size > 0:
                    dops = proj_dopplers[idxs].astype(np.float64)
                    keep = np.array([_pass_estimation_filters(d) for d in dops], dtype=bool)
                    idxs = idxs[keep]
                    dops = dops[keep]

                    if idxs.size > 0:
                        if not hasattr(self, "radar_start_ts"): self.radar_start_ts = now_ts
                        
                        if (now_ts - self.radar_start_ts) >= 0.5:
                            if self.use_nearest_k_points:
                                cx_box, cy_box = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
                                dist2 = (proj_uvs[idxs, 0] - cx_box)**2 + (proj_uvs[idxs, 1] - cy_box)**2
                                k = min(int(self.nearest_k), idxs.size)
                                pick = np.argpartition(dist2, k)[:k] if dist2.size > k else np.arange(dist2.size)
                                dop_med = float(np.median(dops[pick]))
                            else:
                                fwd = radar_points[idxs, 1]
                                sel_dops = select_best_forward_cluster(fwd, dops, max_gap_m=2.0)
                                dop_med = float(np.median(sel_dops)) if sel_dops.size > 0 else None
                            
                            if dop_med is not None:
                                # [핵심] 멀어지는 파란점(양수)과 다가오는 빨간점(음수) 모두 절댓값 처리하여 속도 산출
                                meas_kmh = abs(dop_med * 3.6) 

                        if used_point_mask is not None:
                            used_point_mask[idxs] = True

            # [수정 지점] 백엔드(association_node)의 잘못된 0.00 값을 무시하고 
            # 위에서 GUI가 직접 계산한 meas_kmh를 그대로 사용하도록 덮어씌우기 로직 제거

            # 필터링 및 스무딩 로직
            if meas_kmh is not None:
                if meas_kmh > MAX_REASONABLE_KMH: meas_kmh = None
                if last_vel is not None and abs(meas_kmh - last_vel) > self.max_jump_kmh: meas_kmh = None

            vel_out = None
            if meas_kmh is not None:
                if smoothing == "EMA":
                    prev = float(mem.get("ema", meas_kmh))
                    vel_out = (ema_alpha * meas_kmh) + ((1.0 - ema_alpha) * prev)
                    mem["ema"] = vel_out
                elif smoothing == "Kalman":
                    kf = mem.get("kf", ScalarKalman(x0=meas_kmh))
                    kf.predict(); vel_out = kf.update(meas_kmh)
                    mem["kf"] = kf
                else:
                    vel_out = float(meas_kmh)
                mem["vel_kmh"] = vel_out; mem["ts"] = now_ts
                self.vel_memory[g_id] = mem
            elif last_vel is not None and (hold_sec <= 0.0 or (now_ts - last_ts) <= hold_sec):
                vel_out = last_vel

            item["obj"]["vel"] = round(abs(vel_out), 2) if vel_out is not None else float("nan")

        # 5. 그리기 (Draw)
        for item in draw_items:
            obj = item["obj"]; x1, y1, x2, y2 = item["bbox"]; label_prefix = item["label"]
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            vel = obj.get("vel", float("nan"))
            v_str = f"{vel:.2f}km/h" if np.isfinite(vel) else "--km/h"
            
            font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            line1, line2 = label_prefix, f"Vel: {v_str}"
            (w1, h1), _ = cv2.getTextSize(line1, font, scale, thick)
            (w2, h2), _ = cv2.getTextSize(line2, font, scale, thick)
            bg_top = y1 - (h1 + h2 + 20)
            cv2.rectangle(disp, (x1, bg_top), (x1 + max(w1, w2) + 10, y1), (0, 0, 0), -1)
            cv2.putText(disp, line1, (x1 + 5, y1 - h2 - 10), font, scale, (255, 255, 255), thick)
            cv2.putText(disp, line2, (x1 + 5, y1 - 5), font, scale, (255, 255, 255), thick)

        # 6. 레이더 오버레이 및 메트릭 표시
        projected_count = 0
        if radar_points is not None and self.cam_K is not None:
            pts_r = radar_points.T
            pts_c = self.Extr_R @ pts_r + self.Extr_t.reshape(3, 1)
            valid_idx = pts_c[2, :] > 0.5
            if np.any(valid_idx):
                uvs = (self.cam_K @ pts_c[:, valid_idx])
                uvs /= uvs[2, :]
                dopplers = radar_doppler[valid_idx]
                projected_count = int(np.sum(valid_idx))
                h, w = disp.shape[:2]
                for i in range(uvs.shape[1]):
                    u, v = int(uvs[0, i]), int(uvs[1, i])
                    if 0 <= u < w and 0 <= v < h:
                        speed_kmh = dopplers[i] * 3.6
                        if abs(speed_kmh) < self.spin_min_speed_kmh.value(): continue
                        color = (0, 0, 255) if speed_kmh < -0.5 else (255, 0, 0) if speed_kmh > 0.5 else (0, 255, 0)
                        cv2.circle(disp, (u, v), 4, color, -1)

        # 상단 정보 표시
        speed_vals = [obj["vel"] for obj in self.vis_objects if np.isfinite(obj["vel"])]
        med_speed = f"{np.median(speed_vals):.2f}km/h" if speed_vals else "--"
        overlay = [f"Radar in view: {projected_count}", f"Median speed: {med_speed}"]
        for idx, line in enumerate(overlay):
            cv2.putText(disp, line, (12, 60 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if self.chk_show_poly.isChecked():
            for name, pts in self.lane_polys.items():
                if name in self.chk_lanes and self.chk_lanes[name].isChecked() and pts.size > 0:
                    cv2.polylines(disp, [pts], True, (0, 255, 255), 2)

        self.viewer.update_image(disp)
        
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
        self.txt_extrinsic_log.appendPlainText("\n[Extrinsic] DONE.")

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
                D = ci.get("D", [])
                
                self.txt_intrinsic_log.appendPlainText("\n[Intrinsic] DONE. camera_intrinsic.yaml updated.\n")
            else:
                self.txt_intrinsic_log.appendPlainText(f"\n[Intrinsic] FAILED exit_code={exit_code}\n")

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
            # message는 str
            if message.startswith("QPainter::end: Painter ended with"):
                return
            # 나머지는 그대로 출력
            sys.stderr.write(message + "\n")

        QtCore.qInstallMessageHandler(_qt_msg_handler)

        gui = RealWorldGUI()
        gui.show()
        sys.exit(app.exec())
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()