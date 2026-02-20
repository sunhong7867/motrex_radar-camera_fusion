#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
역할:
- 실차 운영용 GUI에서 영상/레이더/트래킹/연관 결과를 통합 표시하고 수동 외부파라미터 조정 도구를 제공한다.
- 차선 편집, 캘리브레이션 실행, 진단 결과 수신, 로그 저장 기능을 한 화면에서 제어해 운영 절차를 단일 UI로 통합한다.
- 클러스터 대표점/자석선/정확도 지표를 시각화해 보정 상태를 직관적으로 확인하고 파라미터 튜닝 결정을 지원한다.

입력:
- 카메라/레이더/CameraInfo 토픽, 연관/트래킹/최종 출력 토픽, 진단 결과 토픽.
- `extrinsic.json`, `lane_polys.json`, 로그 저장 경로 설정.

출력:
- Qt 기반 운영 화면, 캘리브레이션 제어, 실시간 오버레이 시각화, 데이터 로그 저장.
"""

import sys
import os
import json
import time
import traceback
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Qt, Signal, Slot
import message_filters
import rospy
import rospkg
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from cv_bridge import CvBridge
from std_msgs.msg import String

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
from library.lane_editor import LaneEditorDialog
from library.logging_utils import DataLogger
from library.speed_utils import (
    build_record_row,
    find_target_lane_from_bbox,
    get_bbox_reference_point,
    parse_radar_pointcloud,
    project_points,
    select_representative_point,
    update_lane_tracking,
    update_speed_estimate,
)

from autocal.msg import AssociationArray, DetectionArray
from library import calibration_utils
from library import lane_utils

os.environ["QT_LOGGING_RULES"] = "qt.gui.painting=false"

# ---------------------------------------------------------
# GUI 동작 파라미터
# ---------------------------------------------------------
REFRESH_RATE_MS = 33           # 30 FPS
MAX_REASONABLE_KMH = 100.0     # 육교 고정형 기준 상한
SPEED_THRESHOLD_RED = -0.5     # m/s (이보다 작으면 접근/빨강)
SPEED_THRESHOLD_BLUE = 0.5     # m/s (이보다 크면 이탈/파랑)
NOISE_MIN_SPEED_KMH = 5.0      # 정지/저속(노이즈) 필터 기준 (km/h)
CLUSTER_MAX_DIST = 2.5         # 점 사이 거리 (m)
CLUSTER_MIN_PTS = 3            # 최소 점 개수
CLUSTER_VEL_GATE_MPS = 2.0     # 클러스터 내 속도 일관성 게이트 (m/s)
BBOX2D_MIN_AREA_PX2 = 250      # 2D bbox 최소 면적 (너무 작은 박스 제거)
BBOX2D_MARGIN_PX = 6           # 2D bbox 여유 픽셀
OVERLAY_POINT_RADIUS = 4       # 레이더 대표점 시각화 크기
BBOX_LINE_THICKNESS = 2        # 차량 박스 선 두께
AXIS_LENGTH = 3.0              # 좌표축 화살표 길이 (m)

TOPIC_IMAGE = "/camera/image_raw"
TOPIC_RADAR = "/point_cloud"
TOPIC_CAMERA_INFO = "/camera/camera_info"
TOPIC_FINAL_OUT = "/autocal/output"
TOPIC_TRACKS = "/autocal/tracks"
TOPIC_ASSOCIATED = "/autocal/associated"

START_ACTIVE_LANES = ["IN1", "IN2", "IN3", "OUT1", "OUT2", "OUT3"]

def cluster_radar_points(
    points: np.ndarray,
    doppler: Optional[np.ndarray],
    max_dist: float,
    min_pts: int,
    min_speed_mps: float,
) -> list:
    """
    레이더 점군을 클러스터로 묶어 대표점을 계산
    """
    if points is None or len(points) == 0:
        return []

    n = points.shape[0]
    visited = np.zeros(n, dtype=bool)
    clusters = []

    for i in range(n):
        if visited[i]:
            continue
        dist_sq = np.sum((points[:, :2] - points[i, :2]) ** 2, axis=1)
        neighbors = np.where((dist_sq <= max_dist**2) & (~visited))[0]
        if len(neighbors) < min_pts:
            visited[i] = True
            continue
        visited[neighbors] = True
        vel_med = None
        if doppler is not None and doppler.size > 0:
            vel_med = float(np.median(doppler[neighbors]))
            if abs(vel_med) < min_speed_mps:
                continue
        centroid = np.median(points[neighbors], axis=0)
        clusters.append((centroid, vel_med))

    return clusters


def score_point_to_bbox(target_pt: Tuple[int, int], ref_pt: Tuple[int, int], bbox: Tuple[int, int, int, int]) -> float:
    """
    평가 점수를 계산해 정합 품질을 수치화
    """
    if target_pt is None or ref_pt is None:
        return 0.0
    x1, y1, x2, y2 = bbox
    dist_px = np.hypot(target_pt[0] - ref_pt[0], target_pt[1] - ref_pt[1])
    diag_len = np.hypot(x2 - x1, y2 - y1)
    norm_len = max(120.0, float(diag_len))
    err_ratio = dist_px / norm_len
    return float(max(0, min(100, 100 - (err_ratio * 120))))


def select_cluster_point_for_bbox(
    cluster_uvs: np.ndarray,
    bbox: Tuple[int, int, int, int],
    ref_mode: str = "center",
) -> Optional[Tuple[int, int]]:
    """
    bbox 기준점과 가장 가까운 클러스터 포인트를 선택
    """
    if cluster_uvs is None or cluster_uvs.size == 0:
        return None
    x1, y1, x2, y2 = bbox
    target_u, target_v = get_bbox_reference_point(bbox, ref_mode)
    target_pt = np.array([target_u, target_v], dtype=np.float64)
    dists = np.hypot(cluster_uvs[:, 0] - target_pt[0], cluster_uvs[:, 1] - target_pt[1])
    idx = int(np.argmin(dists))
    return (int(cluster_uvs[idx, 0]), int(cluster_uvs[idx, 1]))

class ImageCanvasViewer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        """
        클래스 상태와 UI/리소스를 초기화
        """
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
        """
        입력 이미지를 QPixmap으로 변환해 갱신
        """
        if cv_img is None:
            return
        h, w, ch = cv_img.shape
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        q_img = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        self.pixmap = QtGui.QPixmap.fromImage(q_img)
        self._image_size = (w, h)
        self.update()

    def set_click_callback(self, callback):
        """
        클릭 이벤트 콜백 함수를 등록
        """
        self._click_callback = callback

    def set_zoom(self, zoom: float):
        """
        뷰어 확대/축소 비율을 설정
        """
        self.zoom = float(np.clip(zoom, 0.00001, 5.0))
        self.update()

    def paintEvent(self, event):
        """
        현재 프레임을 위젯에 렌더링
        """
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
        """
        마우스 클릭 위치를 처리해 상호작용을 수행
        """
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

class ManualCalibWindow(QtWidgets.QDialog):
    def __init__(self, gui: 'RealWorldGUI'):
        """
        클래스 상태와 UI/리소스를 초기화
        """
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

        self.init_ui()
        self._timer = QtCore.QTimer(self); self._timer.timeout.connect(self.update_view); self._timer.start(33)

    def init_ui(self):
        """
        init_ui 함수의 핵심 처리를 수행
        """
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

        L_GRAY, D_GRAY = "#e0e0e0", "#757575"

        # (2) Translation Pad
        gb_trans = QtWidgets.QGroupBox("2. Translation (0.1m)")
        t_grid = QtWidgets.QGridLayout(gb_trans)
        t_grid.addWidget(self._create_btn("FWD ▲ (T)", L_GRAY, lambda: self._move(1, 1)), 0, 1)
        t_grid.addWidget(self._create_btn("LEFT ◀ (R)", L_GRAY, lambda: self._move(0, -1)), 1, 0)
        t_grid.addWidget(self._create_btn("RIGHT ▶ (F)", L_GRAY, lambda: self._move(0, 1)), 1, 2)
        t_grid.addWidget(self._create_btn("BWD ▼ (G)", L_GRAY, lambda: self._move(1, -1)), 2, 1)
        t_grid.addWidget(self._create_btn("UP ⤒ (Y)", D_GRAY, lambda: self._move(2, 1), "white"), 0, 2)
        t_grid.addWidget(self._create_btn("DOWN ⤓ (H)", D_GRAY, lambda: self._move(2, -1), "white"), 2, 2)
        vbox.addWidget(gb_trans)

        # (3) Rotation Pad
        gb_rot = QtWidgets.QGroupBox("3. Rotation (0.5°)")
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
        """
        공통 스타일 버튼을 생성하고 콜백을 연결
        """
        btn = QtWidgets.QPushButton(text)
        btn.setStyleSheet(f"background-color: {bg_color}; color: {text_color}; font-size: 12px;")
        btn.clicked.connect(func); return btn

    def _move(self, axis, sign):
        """
        사용자 입력에 따라 외부파라미터 변환을 적용
        """
        move_vec = np.zeros(3); move_vec[axis] = sign * self.FIXED_MOV
        self.T_current[:3, 3] += self.T_current[:3, :3] @ move_vec
        self.update_view()

    def _rotate(self, axis, sign):
        """
        사용자 입력에 따라 외부파라미터 변환을 적용
        """
        rad = np.deg2rad(sign * self.FIXED_DEG)
        c, s = np.cos(rad), np.sin(rad)
        if axis == 0: R_inc = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif axis == 1: R_inc = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        else: R_inc = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        self.T_current[:3, :3] = self.T_current[:3, :3] @ R_inc
        self.update_view()

    def keyPressEvent(self, event):
        """
        키보드 입력을 해석해 대응 동작을 실행
        """
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
        """
        현재 상태를 기반으로 화면 오버레이와 위젯 표시를 갱신
        """
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
                    cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 0, 255), 2)
            if self.chk_raw.isChecked() and uvs_man is not None:
                for i in range(len(uvs_man)):
                    if not valid_man[i]: continue
                    vel = dops[i] if (dops is not None and i < len(dops)) else 0.0
                    if self.chk_noise_filter.isChecked() and abs(vel) < 1.4: continue
                    color = (0, 255, 255)
                    cv2.circle(disp, (int(uvs_man[i, 0]), int(uvs_man[i, 1])), 5, color, -1)
            self.img_view.update_image(disp)
        except Exception: pass

    def _draw_grid_and_axis(self, img, K, cx, cy):
        """
        화면 오버레이 도형/선을 안전하게 그린다
        """
        grid_z = -1.5; grid_color = (0, 255, 0)
        for x in np.arange(-15, 16, 3): self._draw_safe_line(img, [x, 0, grid_z], [x, 100, grid_z], K, cx, cy, grid_color)
        for y in range(0, 101, 10): self._draw_safe_line(img, [-15, y, grid_z], [15, y, grid_z], K, cx, cy, grid_color)


    def _draw_safe_line(self, img, p1, p2, K, cx, cy, color):
        """
        화면 오버레이 도형/선을 안전하게 그린다
        """
        pts = np.vstack([p1, p2]); uv, v = project_points(K, self.T_current[:3,:3], self.T_current[:3,3], pts)
        if v[0] and v[1]: cv2.line(img, (int(uv[0,0]), int(uv[0,1])), (int(uv[1,0]), int(uv[1,1])), color, 1)

    def _save_extrinsic(self):
        """
        현재 외부파라미터를 파일로 저장
        """
        data = { "R": self.T_current[:3, :3].tolist(), "t": self.T_current[:3, 3].flatten().tolist() }
        with open(self.gui.extrinsic_path, 'w') as f: json.dump(data, f, indent=4)
        self.gui.load_extrinsic(); self.close()

    def _reset_T(self): 
        """
        외부파라미터를 초기 상태로 복원
        """
        self.T_current = self.T_init.copy()
        self.update_view()

    def closeEvent(self, e): 
        """
        종료 시 리소스를 정리하고 마무리 처리
        """
        self._timer.stop() 
        super().closeEvent(e)

class ViewOptionsDialog(QtWidgets.QDialog):
    def __init__(self, gui: 'RealWorldGUI'):
        """
        클래스 상태와 UI/리소스를 초기화
        """
        super().__init__(gui)
        self.gui = gui
        self.setWindowTitle("View Options")
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.Window)

        layout = QtWidgets.QVBoxLayout(self)
        
        # [수정] 부모/레이아웃 저장 로직 삭제
        # 위젯을 이 다이얼로그로 영구적으로 가져옵니다.
        gui.gb_vis.setParent(self)
        gui.gb_vis.setVisible(True)
        layout.addWidget(gui.gb_vis)

        # 사이즈 및 위치 설정
        size = gui.gb_vis.sizeHint()
        self.setFixedSize(size.width() + 40, size.height() + 40)
        
        main_geo = gui.geometry()
        x = main_geo.x() + main_geo.width() - self.width() - 20
        y = main_geo.y() + main_geo.height() - self.height() - 60
        self.move(max(main_geo.x(), x), max(main_geo.y(), y))

    def closeEvent(self, event):
        """
        종료 시 리소스를 정리하고 마무리 처리
        """
        # [수정] 위젯을 돌려보내지 않고 창만 숨깁니다 (Hide)
        event.ignore()  # 창이 파괴되는 것을 막음
        self.hide()
        
        # 메인 윈도우 버튼 텍스트 복구
        if hasattr(self.gui, "btn_toggle_vis"):
            self.gui.btn_toggle_vis.setText("👁️ Show View Options")

    def closeEvent(self, event):
        """
        종료 시 리소스를 정리하고 마무리 처리
        """
        self.gui.gb_vis.setParent(self._original_parent)
        if self._original_layout is not None:
            self._original_layout.addWidget(self.gui.gb_vis)
        self.gui.gb_vis.setVisible(False)
        if hasattr(self.gui, "btn_toggle_vis"):
            self.gui.btn_toggle_vis.setText("👁️ Show View Options")
        super().closeEvent(event)

class RealWorldGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Motrex-SKKU Sensor Fusion GUI")
        
        # 전체 화면으로 시작
        self.showMaximized()

        rospy.init_node('real_gui_node', anonymous=True)
        self.bridge = CvBridge()
        pkg_path = rospkg.RosPack().get_path("autocal")
        self.nodes_dir = os.path.join(CURRENT_DIR, "nodes")
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
        self.lane_stable_cnt = {}       # {id: (candidate_lane, count)}
        self.LANE_STABLE_FRAMES = 45    # 약 1.5초 (30fps 기준). 2초를 원하시면 60으로 설정
        self.radar_track_hist = {}
        self.track_hist_len = 40
        self.pt_alpha = 0.15
        self.vel_gate_mps = 1.5 
        self.spatial_weight_sigma = 30.0 
        self.min_pts_for_rep = 2
        self.kf_vel = {}  # 차량 ID별 속도 칼만 필터 딕셔너리
        self.kf_score = {} # 차량 ID별 점수 칼만 필터 (점수 급락 방지용)
        self.bbox_ref_mode = "bottom"

        self.vel_memory = {}
        self.vel_hold_sec = 0.6
        self.vel_decay_sec = 0.8
        self.vel_decay_floor_kmh = 5.0
        self.vel_rate_limit_kmh_per_s = 12.0
        self.radar_frame_count = 0
        self.radar_warmup_frames = 8
        self.max_jump_kmh = 25.0
        self.use_nearest_k_points = True
        self.nearest_k = 3
        self.speed_update_period = 2
        self._speed_frame_counter = 0
        self.speed_hard_max_kmh = 280.0
        self.speed_soft_max_kmh = 100.0
        self.data_logger = DataLogger()
        self.latest_frame = None

        self.load_extrinsic()
        self.load_lane_polys()
        self.init_ui()

        image_sub = message_filters.Subscriber(TOPIC_IMAGE, Image)
        radar_sub = message_filters.Subscriber(TOPIC_RADAR, PointCloud2)

        self.ts = message_filters.ApproximateTimeSynchronizer([image_sub, radar_sub], 10, 0.1)
        self.ts.registerCallback(self.cb_sync)

        rospy.Subscriber(TOPIC_CAMERA_INFO, CameraInfo, self.cb_info, queue_size=1)
        rospy.Subscriber(TOPIC_FINAL_OUT, AssociationArray, self.cb_final_result, queue_size=1)
        rospy.Subscriber(TOPIC_ASSOCIATED, AssociationArray, self.cb_association_result, queue_size=1)
        rospy.Subscriber(TOPIC_TRACKS, DetectionArray, self.cb_tracker_result, queue_size=1)
        self.pub_diag_start = rospy.Publisher("/autocal/diagnosis/start", String, queue_size=1)
        self.sub_diag_result = rospy.Subscriber("/autocal/diagnosis/result", String, self._on_diag_msg)
        self.latest_diag_msg = ""
        self.diag_dirty = False

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(REFRESH_RATE_MS)

        self.last_update_time = time.time()
        self.view_options_dialog = None

    def init_ui(self):
        # 메인 위젯
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        
        # 전체 레이아웃: 가로 배치 (좌: 뷰어 | 우: 패널)
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

        try:
            rp = rospkg.RosPack()
            pkg_path = rp.get_path("autocal")
            img_dir = os.path.join(pkg_path, "image", "etc")
            
            lbl_skku = QtWidgets.QLabel()
            lbl_motrex = QtWidgets.QLabel()
            
            pix_skku = QtGui.QPixmap(os.path.join(img_dir, "SKKU.png"))
            pix_motrex = QtGui.QPixmap(os.path.join(img_dir, "Motrex.jpeg"))
            
            # 패널 너비(400) 안쪽으로 들어오게 설정
            if not pix_skku.isNull():
                # 성대 로고
                scaled_skku = pix_skku.scaledToWidth(400, Qt.SmoothTransformation)
                lbl_skku.setPixmap(scaled_skku)
                lbl_skku.setAlignment(Qt.AlignCenter)
            
            if not pix_motrex.isNull():
                # 모트렉스 로고
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
        v_c.addWidget(self.pbar_intrinsic)
        v_c.addWidget(self.txt_intrinsic_log)
        v_c.addWidget(self.pbar_extrinsic)
        v_c.addWidget(self.txt_extrinsic_log)

        gb_calib.setLayout(v_c)
        vbox.addWidget(gb_calib)

        # 2. Diagnostic Tool
        gb_diag = QtWidgets.QGroupBox("2. Diagnostic Tool")
        v_diag = QtWidgets.QVBoxLayout()
        self.btn_diag = QtWidgets.QPushButton("📈 RUN TRAJECTORY EVAL")
        self.btn_diag.setStyleSheet("background-color: #3498db; color: white;")
        self.btn_diag.clicked.connect(self._start_diagnostic_ros)

        self.cmb_diag_ref = QtWidgets.QComboBox()
        self.cmb_diag_ref.addItems(["Bottom Center", "Center", "Top Center"])
        self.cmb_diag_ref.setCurrentText(self.bbox_ref_label())
        self.cmb_diag_ref.currentTextChanged.connect(self.set_bbox_ref_mode_from_label)

        self.txt_diag_log = QtWidgets.QPlainTextEdit()
        self.txt_diag_log.setReadOnly(True)
        self.txt_diag_log.setPlaceholderText("Ready. Press button to evaluate trajectories.")
        self.txt_diag_log.setMaximumHeight(180)
        self.txt_diag_log.setStyleSheet("background:#222; color:#0f0; font-family:Consolas; font-size:14px;")
        v_diag.addWidget(self.btn_diag)
        v_diag.addWidget(self.cmb_diag_ref)
        v_diag.addWidget(self.txt_diag_log)
        gb_diag.setLayout(v_diag)
        vbox.addWidget(gb_diag)

        # 3. Calibration Accuracy (New)
        gb2 = QtWidgets.QGroupBox("3. Accuracy / Score")
        v2 = QtWidgets.QVBoxLayout()
        
        h_acc = QtWidgets.QHBoxLayout()
        lbl_calib = QtWidgets.QLabel("Calibration Accuracy")
        self.bar_calib_acc = QtWidgets.QProgressBar()
        self.bar_calib_acc.setRange(0, 100)
        self.bar_calib_acc.setValue(0)
        self.bar_calib_acc.setStyleSheet("QProgressBar::chunk { background-color: #4CAF50; }")
        lbl_emoji = QtWidgets.QLabel("😐")
        lbl_emoji.setStyleSheet("font-size: 30px;")
        self.lbl_emoji = lbl_emoji
        h_acc.addWidget(lbl_calib)
        h_acc.addStretch(1)
        h_acc.addWidget(self.lbl_emoji)

        h_score = QtWidgets.QHBoxLayout()
        lbl_speed_score = QtWidgets.QLabel("Track Speed Score")
        self.bar_speed_score = QtWidgets.QProgressBar()
        self.bar_speed_score.setRange(0, 100)
        self.bar_speed_score.setValue(0)
        self.bar_speed_score.setStyleSheet("QProgressBar::chunk { background-color: #2196F3; }")
        h_score.addWidget(lbl_speed_score)
        
        v2.addLayout(h_acc)
        v2.addWidget(self.bar_calib_acc)
        v2.addLayout(h_score)
        v2.addWidget(self.bar_speed_score)
        gb2.setLayout(v2); vbox.addWidget(gb2)

        # 4. Lane Editor
        gb_lane = QtWidgets.QGroupBox("4. Lane Editor")
        v_l = QtWidgets.QVBoxLayout()
        btn_edit = QtWidgets.QPushButton("🖌️ Open Editor")
        btn_edit.setStyleSheet("background-color: #FFD700; color: black; font-weight: bold; padding: 10px;")
        btn_edit.clicked.connect(self.open_lane_editor)
        v_l.addWidget(btn_edit)
        gb_lane.setLayout(v_l)
        vbox.addWidget(gb_lane)

        # 5. Data Logging
        gb_logging = QtWidgets.QGroupBox("5. Data Logging")
        v_log = QtWidgets.QVBoxLayout()
        v_log.setSpacing(10)

        self.btn_main_record = QtWidgets.QPushButton("🔴 START MAIN RECORDING")
        self.btn_main_record.setCheckable(True)
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
        
        vbox.addWidget(gb_logging)

        # 6. View Options
        gb_vis = QtWidgets.QGroupBox("6. View Options")
        v_vis = QtWidgets.QVBoxLayout()
        gb_vis.setVisible(False)
        self._gb_vis_parent_layout = vbox
        self.btn_toggle_vis = QtWidgets.QPushButton("👁️ Show View Options")
        self.btn_toggle_vis.clicked.connect(self._toggle_view_options)
        vbox.addWidget(self.btn_toggle_vis)
        
        self.chk_show_id = QtWidgets.QCheckBox("Show ID Text")
        self.chk_show_id.setChecked(True)
        self.chk_show_speed = QtWidgets.QCheckBox("Show Speed Text")
        self.chk_show_speed.setChecked(True)

        self.chk_show_boxes = QtWidgets.QCheckBox("Show BBoxes")
        self.chk_show_boxes.setChecked(True)
        self.chk_show_targets = QtWidgets.QCheckBox("Show BBox Reference Points")
        self.chk_show_targets.setChecked(True)

        self.chk_show_radar = QtWidgets.QCheckBox("Show Radar Points")
        self.chk_show_radar.setChecked(False)
        
        self.chk_show_rep_points = QtWidgets.QCheckBox("Show Speed Radar Points")
        self.chk_show_rep_points.setChecked(False)
        self.chk_show_cluster_points = QtWidgets.QCheckBox("Show Cluster Centers")
        self.chk_show_cluster_points.setChecked(True)
        self.chk_show_radar_tracks = QtWidgets.QCheckBox("Show Radar Tracks")
        self.chk_show_radar_tracks.setChecked(False)
        self.chk_magnet = QtWidgets.QCheckBox("Show Magnet Line")
        self.chk_magnet.setChecked(False)

        self.chk_show_poly = QtWidgets.QCheckBox("Show Lane Polygons")
        self.chk_show_poly.setChecked(False)

        self.cmb_bbox_ref = QtWidgets.QComboBox()
        self.cmb_bbox_ref.addItems(["Bottom Center", "Center", "Top Center"])
        self.cmb_bbox_ref.setCurrentText(self.bbox_ref_label())
        self.cmb_bbox_ref.currentTextChanged.connect(self.set_bbox_ref_mode_from_label)

        self.cmb_magnet_mode = QtWidgets.QComboBox()
        self.cmb_magnet_mode.addItems([
            "BBox Representative (Speed)",
            "Cluster Center (Calibration)",
        ])

        v_vis.addWidget(QtWidgets.QLabel("Text / Labels"))
        v_vis.addWidget(self.chk_show_id)
        v_vis.addWidget(self.chk_show_speed)

        sep1 = QtWidgets.QFrame()
        sep1.setFrameShape(QtWidgets.QFrame.HLine)
        sep1.setFrameShadow(QtWidgets.QFrame.Sunken)
        v_vis.addWidget(sep1)

        v_vis.addWidget(QtWidgets.QLabel("Boxes / Reference"))
        v_vis.addWidget(self.chk_show_boxes)
        v_vis.addWidget(self.chk_show_targets)

        sep2 = QtWidgets.QFrame()
        sep2.setFrameShape(QtWidgets.QFrame.HLine)
        sep2.setFrameShadow(QtWidgets.QFrame.Sunken)
        v_vis.addWidget(sep2)

        v_vis.addWidget(QtWidgets.QLabel("Radar / Tracks"))
        v_vis.addWidget(self.chk_show_radar)
        v_vis.addWidget(self.chk_show_rep_points)
        v_vis.addWidget(self.chk_show_cluster_points)
        v_vis.addWidget(self.chk_show_radar_tracks)
        v_vis.addWidget(self.chk_magnet)

        sep3 = QtWidgets.QFrame()
        sep3.setFrameShape(QtWidgets.QFrame.HLine)
        sep3.setFrameShadow(QtWidgets.QFrame.Sunken)
        v_vis.addWidget(sep3)

        v_vis.addWidget(QtWidgets.QLabel("Association Settings"))
        v_vis.addWidget(QtWidgets.QLabel("BBox Reference (Auto)"))
        v_vis.addWidget(self.cmb_bbox_ref)
        v_vis.addWidget(QtWidgets.QLabel("Magnet Source"))
        v_vis.addWidget(self.cmb_magnet_mode)

        sep4 = QtWidgets.QFrame()
        sep4.setFrameShape(QtWidgets.QFrame.HLine)
        sep4.setFrameShadow(QtWidgets.QFrame.Sunken)
        v_vis.addWidget(sep4)

        v_vis.addWidget(QtWidgets.QLabel("Lane Overlay"))
        v_vis.addWidget(self.chk_show_poly)

        self.chk_lanes = {}
        display_order = ["IN1", "OUT1", "IN2", "OUT2", "IN3", "OUT3"]
        g_vis = QtWidgets.QGridLayout()

        for i, name in enumerate(display_order):
            chk = QtWidgets.QCheckBox(name)
            is_active = True
            chk.setChecked(is_active)
            self.chk_lanes[name] = chk
            g_vis.addWidget(chk, i // 2, i % 2)

        v_vis.addLayout(g_vis)

        btn_reset_id = QtWidgets.QPushButton("Reset Local IDs")
        btn_reset_id.clicked.connect(self.reset_ids)
        v_vis.addWidget(btn_reset_id)
        
        gb_vis.setLayout(v_vis)
        self.gb_vis = gb_vis
        vbox.addWidget(gb_vis)

        # 하단 여백 추가
        vbox.addStretch()
        layout.addWidget(panel, stretch=0)
                
        # 내부적으로 사용하는 speed parameter 들 (UI에는 없지만 로직 유지를 위해 필요)
        self.spin_hold_sec = type('obj', (object,), {'value': lambda: 10.0})
        self.cmb_smoothing = type('obj', (object,), {'currentText': lambda: "EMA"})
        self.spin_ema_alpha = type('obj', (object,), {'value': lambda: 0.35})

    def reset_ids(self):
        """
        트래킹/로깅 식별자 상태를 초기화
        """
        self.lane_counters = {name: 0 for name in self.lane_counters.keys()}
        self.global_to_local_ids = {}
        self.vehicle_lane_paths = {}

    def _toggle_view_options(self):
        """
        뷰 옵션 토글 상태를 반영
        """
        # 1. 다이얼로그가 없으면 딱 한 번만 생성
        if self.view_options_dialog is None:
            self.view_options_dialog = ViewOptionsDialog(self)
        
        # 2. 이미 열려있으면 숨김
        if self.view_options_dialog.isVisible():
            self.view_options_dialog.hide()
            if hasattr(self, "btn_toggle_vis"):
                self.btn_toggle_vis.setText("👁️ Show View Options")
        
        # 3. 닫혀있으면 보임
        else:
            self.view_options_dialog.show()
            self.view_options_dialog.raise_()
            self.view_options_dialog.activateWindow()
            if hasattr(self, "btn_toggle_vis"):
                self.btn_toggle_vis.setText("👁️ Hide View Options")

    def _start_diagnostic_ros(self):
        """
        백그라운드 작업 또는 프로세스를 시작
        """
        if hasattr(self, "btn_diag") and self.btn_diag is not None:
            self.btn_diag.setEnabled(False)
        if hasattr(self, "txt_diag_log") and self.txt_diag_log is not None:
            self.txt_diag_log.setPlainText(">> [REQUEST] Starting trajectory evaluation...\n>> Collecting track data...")
        self.pub_diag_start.publish(String(data=str(self.bbox_ref_mode)))

    def _on_diag_msg(self, msg):
        """
        진단 결과 메시지를 수신해 UI에 표시
        """
        self.latest_diag_msg = msg.data
        self.diag_dirty = True
        if hasattr(self, "btn_diag") and self.btn_diag is not None:
            self.btn_diag.setEnabled(True)

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

    def cb_sync(self, img_msg, radar_msg):
        """
        ROS 콜백 입력 메시지를 내부 상태로 반영
        """
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
        """
        ROS 콜백 입력 메시지를 내부 상태로 반영
        """
        if self.cam_K is None:
            self.cam_K = np.array(msg.K).reshape(3, 3)

    def cb_final_result(self, msg):
        """
        ROS 콜백 입력 메시지를 내부 상태로 반영
        """
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
        """
        ROS 콜백 입력 메시지를 내부 상태로 반영
        """
        assoc_speeds = {}
        for obj in msg.objects:
            if np.isfinite(obj.speed_kph):
                assoc_speeds[obj.id] = obj.speed_kph
        self.assoc_speed_by_id = assoc_speeds
        self.assoc_last_update_time = time.time()

    def cb_tracker_result(self, msg):
        """
        ROS 콜백 입력 메시지를 내부 상태로 반영
        """
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
        """
        텍스트 오버레이 표시 좌표를 계산
        """
        sorted_indices = np.argsort(pts[:, 1])[::-1]
        bottom_two = pts[sorted_indices[:2]]
        left_idx = np.argmin(bottom_two[:, 0])
        return tuple(bottom_two[left_idx])

    def update_loop(self):
        """
        현재 상태를 기반으로 화면 오버레이와 위젯 표시를 갱신
        """
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

            # 2. 레이더 투영
            proj_uvs = None
            proj_valid = None
            projected_count = 0 
            cluster_uvs = None
            cluster_valid = None

            if pts_raw is not None and self.cam_K is not None:
                uvs, valid = project_points(self.cam_K, self.Extr_R, self.Extr_t, pts_raw)
                proj_uvs = uvs
                proj_valid = valid
                projected_count = int(np.sum(valid))

                if self.Extr_R is not None and self.Extr_t is not None:
                    min_speed_mps = NOISE_MIN_SPEED_KMH / 3.6
                    clusters = cluster_radar_points(
                        pts_raw,
                        dop_raw,
                        CLUSTER_MAX_DIST,
                        CLUSTER_MIN_PTS,
                        min_speed_mps,
                    )
                    if clusters:
                        centers = np.array([c[0] for c in clusters], dtype=np.float64)
                        cuv, cvalid = project_points(self.cam_K, self.Extr_R, self.Extr_t, centers)
                        cluster_uvs = cuv
                        cluster_valid = cvalid

            # 3. 차선 필터 준비
            active_lane_polys = {}
            for name, chk in self.chk_lanes.items():
                if chk.isChecked():
                    poly = self.lane_polys.get(name)
                    if poly is not None and len(poly) > 2:
                        active_lane_polys[name] = poly
            has_lane_filter = bool(active_lane_polys)

            draw_items = []
            calib_scores = []
            speed_scores = []
            now_ts = time.time()

            # 4. 객체별 데이터 처리 루프
            confirmed_objects = [] # 실제로 2초 이상 검증된 객체들만 담을 리스트
            active_ids = set()
            
            for obj in self.vis_objects:
                g_id = obj['id']
                
                # 연속 감지 카운트 업데이트
                self.track_confirmed_cnt[g_id] = self.track_confirmed_cnt.get(g_id, 0) + 1
                
                # 차선 영역 진입 즉시 박스 표시
                if self.track_confirmed_cnt[g_id] < 20:
                    continue
                
                confirmed_objects.append(obj)
                
            # 검증된 객체들로 다시 루프 수행
            for obj in confirmed_objects:
                g_id = obj['id']
                x1, y1, x2, y2 = obj['bbox']
                active_ids.add(g_id)
                
                # 1. 차선 필터링 (기존 로직 유지)
                target_lane = None
                if has_lane_filter:
                    target_lane = find_target_lane_from_bbox((x1, y1, x2, y2), active_lane_polys)
                    if target_lane is None:
                        self.radar_track_hist.pop(g_id, None)
                        self.vehicle_lane_paths.pop(g_id, None)
                        self.global_to_local_ids.pop(g_id, None)
                        self.track_confirmed_cnt.pop(g_id, None)
                        active_ids.discard(g_id)
                        continue
                
                if g_id in self.vehicle_lane_paths:
                    # 현재 확정된 마지막 차선 (예: "IN2->OUT2" 에서 "OUT2")
                    last_confirmed = self.vehicle_lane_paths[g_id].split("->")[-1]
                    
                    # 감지된 차선(target_lane)이 확정된 차선과 다르다면? (차선 변경 시도)
                    if target_lane != last_confirmed:
                        # 후보 차선과 카운트 가져오기
                        cand_lane, cnt = self.lane_stable_cnt.get(g_id, (target_lane, 0))
                        
                        if cand_lane == target_lane:
                            # 같은 후보 차선이 계속 유지되면 카운트 증가
                            cnt += 1
                        else:
                            # 후보 차선이 또 바뀌었으면(왔다갔다 하면) 리셋
                            cand_lane = target_lane
                            cnt = 1
                        
                        self.lane_stable_cnt[g_id] = (cand_lane, cnt)

                        # 임계값(1.5초)을 넘지 못했으면 아직 '변경 안 됨'으로 간주하고 기존 차선 유지
                        if cnt < self.LANE_STABLE_FRAMES:
                            target_lane = last_confirmed
                        else:
                            # 임계값 넘으면 변경 인정! (카운터 제거하여 다음 변경 대기)
                            self.lane_stable_cnt.pop(g_id, None)
                    else:
                        # 다시 원래 차선으로 돌아오면 카운터 리셋
                        self.lane_stable_cnt.pop(g_id, None)
                else:
                    # 처음 발견된 차량은 안정화 없이 즉시 반영
                    self.lane_stable_cnt.pop(g_id, None)

                curr_lane_str, lane_path, local_no, label = update_lane_tracking(
                    g_id,
                    target_lane,
                    self.vehicle_lane_paths,
                    self.lane_counters,
                    self.global_to_local_ids,
                )

                obj['lane_path'] = lane_path
                obj['lane'] = curr_lane_str

                target_pt = get_bbox_reference_point((x1, y1, x2, y2), self.bbox_ref_mode)
                
                meas_kmh = None
                rep_pt = None
                rep_vel_raw = None
                score = 0
                radar_dist = -1.0
                cluster_pt = None
                cluster_score = 0.0

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
                        NOISE_MIN_SPEED_KMH,
                        self.bbox_ref_mode,
                    )
                if cluster_uvs is not None and cluster_valid is not None:
                    valid_clusters = cluster_uvs[cluster_valid]
                    cluster_pt = select_cluster_point_for_bbox(valid_clusters, (x1, y1, x2, y2), self.bbox_ref_mode)
                    if cluster_pt is not None:
                        cluster_score = score_point_to_bbox(target_pt, cluster_pt, (x1, y1, x2, y2))
                        if cluster_score < 30:
                            cluster_pt = None
                            cluster_score = 0.0

                if rep_vel_raw is None or abs(rep_vel_raw) < NOISE_MIN_SPEED_KMH:
                    cluster_pt = None
                    cluster_score = 0.0

                if cluster_score > 0:
                    calib_scores.append(cluster_score)

                vel_out, score_out = update_speed_estimate(
                    g_id,
                    meas_kmh,
                    score,
                    now_ts,
                    self.vel_memory,
                    self.kf_vel,
                    self.kf_score,
                    hold_sec=self.vel_hold_sec,
                    decay_sec=self.vel_decay_sec,
                    decay_floor_kmh=self.vel_decay_floor_kmh,
                    rate_limit_kmh_per_s=self.vel_rate_limit_kmh_per_s,
                )

                obj['vel'] = round(vel_out, 1) if vel_out is not None else float('nan')
                score = score_out
                if score_out > 0:
                    speed_scores.append(score_out)

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
                    "rep_pt": rep_pt,
                    "rep_vel": rep_vel_raw,
                    "target_pt": target_pt,
                    "score": score,
                    "cluster_pt": cluster_pt,
                    "cluster_score": cluster_score,
                })

                if cluster_pt is not None:
                    self.radar_track_hist.setdefault(g_id, deque(maxlen=self.track_hist_len)).append(cluster_pt)

            # 5. 전역 UI 업데이트
            if calib_scores:
                avg = sum(calib_scores) / len(calib_scores)
                self.bar_calib_acc.setValue(int(avg))
                self.lbl_emoji.setText("😊" if avg >= 80 else "😐" if avg >= 50 else "😟")
            else:
                self.bar_calib_acc.setValue(0)
                self.lbl_emoji.setText("😐")

            if speed_scores:
                avg_speed = sum(speed_scores) / len(speed_scores)
                self.bar_speed_score.setValue(int(avg_speed))
            else:
                self.bar_speed_score.setValue(0)

            if self.data_logger.is_recording:
                 self.lbl_record_status.setText(f"Recording... [{self.data_logger.record_count} pts]")

            # 6. 그리기 (Draw)
            if self.chk_show_radar.isChecked() and proj_uvs is not None:
                for i in range(len(proj_uvs)):
                    if not proj_valid[i]: continue
                    u, v = int(proj_uvs[i, 0]), int(proj_uvs[i, 1])
                    spd = dop_raw[i] * 3.6
                    
                    if abs(spd) < NOISE_MIN_SPEED_KMH: continue
                    
                    col = (0, 255, 255)
                    cv2.circle(disp, (u, v), 4, col, -1)

            if self.chk_show_radar_tracks.isChecked():
                for g_id, pts in self.radar_track_hist.items():
                    if len(pts) >= 2:
                        arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
                        cv2.polylines(disp, [arr], False, (0, 255, 0), 2)
                        if arr.shape[0] >= 2:
                            p1 = tuple(arr[-2][0])
                            p2 = tuple(arr[-1][0])
                            cv2.arrowedLine(disp, p1, p2, (0, 255, 0), 2, tipLength=0.3)

            for it in draw_items:
                x1, y1, x2, y2 = it["bbox"]
                if self.chk_show_boxes.isChecked():
                    cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 0, 255), 2)
                if self.chk_show_targets.isChecked():
                    cv2.circle(disp, it["target_pt"], 5, (0, 255, 255), -1) 

                magnet_mode = self.cmb_magnet_mode.currentText()
                magnet_pt = None
                magnet_score = 0.0
                if magnet_mode == "Cluster Center (Calibration)":
                    magnet_pt = it["cluster_pt"]
                    magnet_score = it["cluster_score"]
                else:
                    magnet_pt = it["rep_pt"]
                    magnet_score = it["score"]

                if magnet_pt is not None and self.chk_magnet.isChecked():
                    lc = (255, 0, 0) if magnet_score > 50 else (0, 0, 255)
                    cv2.line(disp, it["target_pt"], magnet_pt, lc, 2)

                if (
                    it["cluster_pt"] is not None
                    and magnet_mode == "Cluster Center (Calibration)"
                    and self.chk_show_cluster_points.isChecked()
                ):
                    cv2.circle(disp, it["cluster_pt"], 5, (255, 0, 255), -1)
                    cv2.circle(disp, it["cluster_pt"], 6, (255, 255, 255), 1)

                if it["rep_pt"] is not None and self.chk_show_rep_points.isChecked():
                    rv = it["rep_vel"]
                    if rv is not None and abs(rv) >= NOISE_MIN_SPEED_KMH:
                        col = (0, 255, 255)
                        cv2.circle(disp, it["rep_pt"], 6, col, -1)
                        cv2.circle(disp, it["rep_pt"], 7, (255, 255, 255), 1)

                lines = []
                if self.chk_show_id.isChecked(): lines.append(it['label'])
                if self.chk_show_speed.isChecked():
                    v = it["obj"].get("vel", float("nan"))
                    lines.append(f"Vel: {int(v)}km/h" if np.isfinite(v) else "Vel: --km/h")

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

            visible_ids = {o['id'] for o in self.vis_objects}
            self.track_confirmed_cnt = {k: v for k, v in self.track_confirmed_cnt.items() if k in visible_ids}
            self.radar_track_hist = {k: v for k, v in self.radar_track_hist.items() if k in active_ids}
            self.lane_stable_cnt = {k: v for k, v in self.lane_stable_cnt.items() if k in visible_ids}
            if self.diag_dirty and hasattr(self, "txt_diag_log"):
                self.txt_diag_log.setPlainText(self.latest_diag_msg)
                self.diag_dirty = False

            self.viewer.update_image(disp)

        except Exception as e:
            print(f"[GUI Error] update_loop failed: {e}")
            traceback.print_exc()
            
    def open_lane_editor(self):
        """
        차선 편집 다이얼로그를 연다
        """
        if self.cv_image is None:
            return
        dlg = LaneEditorDialog(self.cv_image, self.lane_polys, self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self.lane_polys = dlg.get_polys()
            self.save_lane_polys()

    def save_lane_polys(self):
        """
        현재 차선 폴리곤을 파일로 저장
        """
        lane_utils.save_lane_polys(self.lane_json_path, self.lane_polys)

    def load_lane_polys(self):
        """
        설정 파일을 읽어 내부 데이터 구조를 갱신
        """
        try:
            self.lane_polys = lane_utils.load_lane_polys(self.lane_json_path)
        except Exception:
            self.lane_polys = {}

    def load_extrinsic(self):
        """
        설정 파일을 읽어 내부 데이터 구조를 갱신
        """
        calibration_utils.load_extrinsic(self)

    def bbox_ref_label(self) -> str:
        """
        현재 bbox 기준점 모드 라벨을 반환
        """
        if self.bbox_ref_mode == "top":
            return "Top Center"
        if self.bbox_ref_mode == "center":
            return "Center"
        return "Bottom Center"

    def set_bbox_ref_mode_from_label(self, label: str) -> None:
        """
        라벨 문자열을 기준점 모드로 변환해 적용
        """
        label = (label or "").lower()
        if "top" in label:
            self.bbox_ref_mode = "top"
        elif "center" in label and "bottom" not in label:
            self.bbox_ref_mode = "center"
        else:
            self.bbox_ref_mode = "bottom"
        if hasattr(self, "cmb_bbox_ref") and self.cmb_bbox_ref is not None:
            target_label = self.bbox_ref_label()
            if self.cmb_bbox_ref.currentText() != target_label:
                self.cmb_bbox_ref.setCurrentText(target_label)

    def run_calibration(self):
        """
        캘리브레이션 실행 절차를 시작
        """
        calibration_utils.run_calibration(self, ManualCalibWindow)

    def run_autocalibration(self):
        """
        캘리브레이션 실행 절차를 시작
        """
        calibration_utils.run_autocalibration(self)

    def _on_extrinsic_stdout(self):
        """
        외부 프로세스 표준출력 로그를 수신해 화면에 반영
        """
        if not hasattr(self, "extrinsic_proc") or self.extrinsic_proc is None:
            return
        data = bytes(self.extrinsic_proc.readAllStandardOutput()).decode("utf-8", errors="ignore")
        if not data:
            return
        self.txt_extrinsic_log.appendPlainText(data.rstrip())

    def _on_extrinsic_stderr(self):
        """
        외부 프로세스 표준에러 로그를 수신해 화면에 반영
        """
        if not hasattr(self, "extrinsic_proc") or self.extrinsic_proc is None:
            return
        data = bytes(self.extrinsic_proc.readAllStandardError()).decode("utf-8", errors="ignore")
        if not data:
            return
        self.txt_extrinsic_log.appendPlainText(data.rstrip())

    def _on_extrinsic_finished(self):
        """
        외부 프로세스 종료 결과를 처리하고 후속 상태를 갱신
        """
        if hasattr(self, "pbar_extrinsic") and self.pbar_extrinsic is not None:
            self.pbar_extrinsic.setVisible(False)
        # 오류 수정: 문자열 내 실제 줄바꿈 제거
        self.txt_extrinsic_log.appendPlainText("[Extrinsic] DONE.")

    def run_intrinsic_calibration(self):
        """
        캘리브레이션 실행 절차를 시작
        """
        calibration_utils.run_intrinsic_calibration(self)

    def _on_intrinsic_stdout(self):
        """
        외부 프로세스 표준출력 로그를 수신해 화면에 반영
        """
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
        """
        외부 프로세스 표준에러 로그를 수신해 화면에 반영
        """
        if not hasattr(self, "intrinsic_proc") or self.intrinsic_proc is None:
            return
        data = bytes(self.intrinsic_proc.readAllStandardError()).decode("utf-8", errors="ignore")
        if not data:
            return
        self.txt_intrinsic_log.appendPlainText(data.rstrip())

    def _on_intrinsic_finished(self, exit_code, exit_status):
        """
        외부 프로세스 종료 결과를 처리하고 후속 상태를 갱신
        """
        try:
            self.pbar_intrinsic.setVisible(False)
            rp = rospkg.RosPack()
            pkg_path = rp.get_path("autocal")
            out_json = os.path.join(pkg_path, "config", "intrinsic.json")

            if exit_code == 0 and os.path.exists(out_json):
                import json
                with open(out_json, "r", encoding="utf-8") as f:
                    ci = json.load(f) or {}
                self.txt_intrinsic_log.appendPlainText("[Intrinsic] DONE. intrinsic.json updated.")
            else:
                self.txt_intrinsic_log.appendPlainText(f"[Intrinsic] FAILED exit_code={exit_code}")

        except Exception as e:
            print(f"[Intrinsic] finish error: {e}")
            traceback.print_exc()

    def _maybe_reload_extrinsic(self):
        """
        파일 변경 여부를 확인해 필요 시 설정을 재로딩
        """
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
