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
BEV_LATERAL_LIMIT = (-70, 70)  # 그래프 가로축 (Screen X) 범위
BEV_FORWARD_LIMIT = (0, 120)   # 그래프 세로축 (Screen Y) 범위

# 4. 토글 킬 차선
START_ACTIVE_LANES = ["IN1"]

# 5. 토픽 이름
TOPIC_IMAGE = "/camera/image_raw"
TOPIC_RADAR = "/point_cloud"
TOPIC_CAMERA_INFO = "/camera/camera_info"
TOPIC_FINAL_OUT = "/perception_test/output"
TOPIC_TRACKS = "/perception_test/tracks"

# ==============================================================================
# Setup
# ==============================================================================
try:
    import matplotlib
    matplotlib.use('qtagg')
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle
except ImportError:
    print("[Error] Matplotlib backend_qtagg 로드 실패.")
    sys.exit(1)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

try:
    from PySide6 import QtWidgets, QtGui, QtCore
    from PySide6.QtCore import Qt
    import rospy
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

class RadarGraphCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#101010')
        super(RadarGraphCanvas, self).__init__(self.fig)
        self.setParent(parent)

        # 리사이즈/레이아웃 반영 안정화
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.ax_bev = self.fig.add_subplot(111, facecolor='#202020')
        self.fig.tight_layout()

    @staticmethod
    def cluster_points(points_xy, max_dist, min_points):
        """간단한 거리 기반 클러스터링(DFS/BFS).
        points_xy: (N,2)
        return: [np.array(indices), ...]
        """
        n_points = points_xy.shape[0]
        visited = np.zeros(n_points, dtype=bool)
        clusters = []

        for i in range(n_points):
            if visited[i]:
                continue
            queue = [i]
            visited[i] = True
            cluster_indices = []

            while queue:
                idx = queue.pop()
                cluster_indices.append(idx)
                dists = np.linalg.norm(points_xy - points_xy[idx], axis=1)
                neighbors = np.where(dists <= max_dist)[0]
                for nb in neighbors:
                    if not visited[nb]:
                        visited[nb] = True
                        queue.append(nb)

            if len(cluster_indices) >= min_points:
                clusters.append(np.array(cluster_indices, dtype=int))

        return clusters

    def update_graphs(self, points, dopplers, powers):
        self.ax_bev.clear()

        # ---------------------------------------------------------
        # 1. BEV (Top View) - 전방을 위쪽으로 설정
        # ---------------------------------------------------------
        self.ax_bev.set_title("Radar Point Cloud (Top View)", color='white')
        self.ax_bev.tick_params(colors='white')
        self.ax_bev.grid(True, color='#404040')
        self.ax_bev.set_facecolor('#202020')

        # 분석 결과에 따라 축 설정
        self.ax_bev.set_xlim(BEV_LATERAL_LIMIT)  # Screen X = Data X (Lateral)
        self.ax_bev.set_ylim(BEV_FORWARD_LIMIT)  # Screen Y = Data Y (Forward)
        self.ax_bev.set_xlabel("Lateral (X) [m]", color='gray')
        self.ax_bev.set_ylabel("Forward (Y) [m]", color='gray')

        # 비율 고정 (1:1 비율로 보이게 하여 왜곡 방지)
        self.ax_bev.set_aspect('equal', adjustable='box')

        if points is None or len(points) == 0:
            self.fig.tight_layout()
            self.draw_idle()
            return

        colors = []
        valid_indices = []

        for i in range(len(points)):
            speed_kmh = dopplers[i] * 3.6
            if abs(speed_kmh) < NOISE_MIN_SPEED_KMH:
                continue

            valid_indices.append(i)
            if speed_kmh < SPEED_THRESHOLD_RED:
                colors.append('red')
            elif speed_kmh > SPEED_THRESHOLD_BLUE:
                colors.append('blue')
            else:
                colors.append('white')

        if not valid_indices:
            self.fig.tight_layout()
            self.draw_idle()
            return

        idx = valid_indices
        pts_x = points[idx, 0]  # Data X = Lateral
        pts_y = points[idx, 1]  # Data Y = Forward

        self.ax_bev.scatter(pts_x, pts_y, c=colors, s=GRAPH_POINT_SIZE)

        moving_mask = np.abs(dopplers[idx] * 3.6) >= MOVING_CLUSTER_MIN_SPEED_KMH
        moving_points = np.column_stack((pts_x[moving_mask], pts_y[moving_mask]))
        if len(moving_points) > 0:
            clusters = RadarGraphCanvas.cluster_points(moving_points, CLUSTER_MAX_DIST_M, CLUSTER_MIN_POINTS)
            for cluster in clusters:
                cluster_pts = moving_points[cluster]
                min_x, min_y = np.min(cluster_pts, axis=0)
                max_x, max_y = np.max(cluster_pts, axis=0)
                rect = Rectangle(
                    (min_x, min_y),
                    max_x - min_x,
                    max_y - min_y,
                    linewidth=2,
                    edgecolor='lime',
                    facecolor='none'
                )
                self.ax_bev.add_patch(rect)

        # Qt+Matplotlib 조합에서 draw_idle이 안정적
        self.fig.tight_layout()
        self.draw_idle()


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
        self.resize(1280, 950)

        rospy.init_node('real_gui_node', anonymous=True)
        self.bridge = CvBridge()
        self.nodes_dir = os.path.join(CURRENT_DIR, "nodes")
        self.extrinsic_path = os.path.join(self.nodes_dir, "extrinsic.json")
        self.lane_json_path = os.path.join(self.nodes_dir, "lane_polys.json")

        self.cv_image = None
        self.radar_points = None
        self.cam_K = None
        self.lane_polys = {}
        self.Extr_R = np.eye(3)
        self.Extr_t = np.zeros((3, 1))
        self.vis_objects = []
        self.lane_counters = {name: 0 for name in ["IN1", "IN2", "IN3", "OUT1", "OUT2", "OUT3"]}
        self.global_to_local_ids = {}
        self.is_paused = False

        # Pause 동안 들어오는 동기 데이터 버퍼(프레임 단위). Resume 시 버퍼를 먼저 재생.
        self.pause_queue = deque(maxlen=900)  # 대략 30FPS 기준 30초
        self.play_from_queue = False
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
        self.graph_canvas = RadarGraphCanvas(width=5, height=3)
        left_layout.addWidget(self.viewer, stretch=2)
        left_layout.addWidget(self.graph_canvas, stretch=1)
        layout.addWidget(left_widget, stretch=4)

        panel = QtWidgets.QWidget()
        panel.setFixedWidth(280)
        vbox = QtWidgets.QVBoxLayout(panel)
        vbox.setAlignment(Qt.AlignTop)

        gb_calib = QtWidgets.QGroupBox("1. Calibration")
        v_c = QtWidgets.QVBoxLayout()
        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_pause.clicked.connect(self.toggle_pause)
        btn_calib = QtWidgets.QPushButton("Run Calibration")
        btn_calib.clicked.connect(self.run_calibration)
        btn_reload = QtWidgets.QPushButton("Reload JSON")
        btn_reload.clicked.connect(self.load_extrinsic)
        v_c.addWidget(self.btn_pause)
        v_c.addWidget(btn_calib)
        v_c.addWidget(btn_reload)
        gb_calib.setLayout(v_c)
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

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.btn_pause.setText("Resume")
            self.lbl_log.setText("Paused.")
        else:
            # Resume 시: Pause 동안 쌓인 프레임을 먼저 소모(멈춘 지점부터 이어 재생처럼 보이게)
            self.play_from_queue = True
            self.btn_pause.setText("Pause")
            self.lbl_log.setText("Resumed.")

    # ------------------ Callback (Synchronized) ------------------
    def cb_sync(self, img_msg, radar_msg):
        # 1) 이미지
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except:
            return

        # 2) 레이더
        try:
            target_fields = ("x", "y", "z", "power", "doppler")
            field_names = [f.name for f in radar_msg.fields]
            if "power" not in field_names and "intensity" in field_names:
                target_fields = ("x", "y", "z", "intensity", "doppler")

            g = pc2.read_points(radar_msg, field_names=target_fields, skip_nans=True)
            self.radar_points_all = np.array(list(g))

            if self.radar_points_all.shape[0] > 0:
                radar_points = self.radar_points_all[:, :3]
                if self.radar_points_all.shape[1] >= 5:
                    radar_power = self.radar_points_all[:, 3]
                    radar_doppler = self.radar_points_all[:, 4]
                else:
                    radar_power = np.zeros(len(radar_points))
                    radar_doppler = np.zeros(len(radar_points))
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

        # Pause 중에는 현재 화면을 덮어쓰지 않고 큐에만 저장
        if self.is_paused:
            self.pause_queue.append(frame)
            return

        self.latest_frame = frame

    def cb_info(self, msg):
        if self.cam_K is None:
            self.cam_K = np.array(msg.K).reshape(3, 3)

    def cb_final_result(self, msg):
        objects = []
        for obj in msg.objects:
            objects.append({
                'id': obj.id,
                'bbox': [int(obj.bbox.xmin), int(obj.bbox.ymin), int(obj.bbox.xmax), int(obj.bbox.ymax)],
                'vel': obj.speed_kph
            })
        self.vis_objects = objects
        self.last_update_time = time.time()

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
        if self.is_paused:
            return

        # Resume 직후에는 pause_queue를 먼저 재생
        frame = None
        if self.play_from_queue and len(self.pause_queue) > 0:
            frame = self.pause_queue.popleft()
            if len(self.pause_queue) == 0:
                self.play_from_queue = False
        else:
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
        for obj in self.vis_objects:
            g_id = obj['id']
            x1, y1, x2, y2 = obj['bbox']
            cx, cy = (x1 + x2) // 2, y2

            target_lane = None
            for name, chk in self.chk_lanes.items():
                if chk.isChecked():
                    poly = self.lane_polys.get(name)
                    if poly is not None and len(poly) > 2:
                        if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                            target_lane = name
                            break

            if target_lane:
                if g_id not in self.global_to_local_ids:
                    self.lane_counters[target_lane] += 1
                    self.global_to_local_ids[g_id] = self.lane_counters[target_lane]
                local_id = self.global_to_local_ids[g_id]

                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), BBOX_LINE_THICKNESS)
                vel = obj['vel']
                v_str = f"{int(vel)}km/h" if np.isfinite(vel) else "--km/h"
                line1 = f"No: {local_id} ({target_lane})"
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
        if radar_points is not None and self.cam_K is not None:
            pts_r = radar_points.T
            pts_c = self.Extr_R @ pts_r + self.Extr_t.reshape(3, 1)
            valid_idx = pts_c[2, :] > 0.5

            if np.any(valid_idx):
                pts_c = pts_c[:, valid_idx]
                dopplers = radar_doppler[valid_idx] if radar_doppler is not None else np.zeros(pts_c.shape[1])

                uvs = self.cam_K @ pts_c
                uvs /= uvs[2, :]
                h, w = disp.shape[:2]

                for i in range(uvs.shape[1]):
                    u, v = int(uvs[0, i]), int(uvs[1, i])
                    if 0 <= u < w and 0 <= v < h:
                        speed_kmh = dopplers[i] * 3.6
                        if abs(speed_kmh) < NOISE_MIN_SPEED_KMH:
                            continue

                        if speed_kmh < SPEED_THRESHOLD_RED:
                            color = (0, 0, 255)
                        elif speed_kmh > SPEED_THRESHOLD_BLUE:
                            color = (255, 0, 0)
                        else:
                            color = (255, 255, 255)

                        cv2.circle(disp, (u, v), OVERLAY_POINT_RADIUS, color, -1)

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

        # 같은 frame으로 레이더 그래프도 동시에 업데이트
        if radar_points is not None:
            d = radar_doppler if radar_doppler is not None else np.zeros(len(radar_points))
            p = radar_power if radar_power is not None else np.zeros(len(radar_points))
            self.graph_canvas.update_graphs(radar_points, d, p)

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
                if hasattr(self, "lbl_log"):
                    self.lbl_log.setText("Extrinsic Loaded.")
            except:
                pass

    def run_calibration(self):
        calib_path = os.path.join(CURRENT_DIR, "perception_lib", "calibration_manager.py")
        subprocess.Popen(["python3", calib_path])


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
