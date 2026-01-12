#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
real_gui.py (Local ID Mapping Version)

[업데이트 사항]
- 차선별 로컬 ID 관리: 활성화된 차선에 진입한 차량 순서대로 1, 2, 3... 번호 부여
- 글로벌-로컬 ID 매핑: 동일 차량(Global ID)에 대해 로컬 번호 유지
"""

import sys
import os
import json
import time
import subprocess
import traceback
import numpy as np
import cv2

# ==============================================================================
# 경로 및 임포트
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

try:
    from PySide6 import QtWidgets, QtGui, QtCore
    from PySide6.QtCore import Qt

    import rospy
    import rospkg
    from sensor_msgs.msg import Image, PointCloud2, CameraInfo
    from cv_bridge import CvBridge
    import sensor_msgs.point_cloud2 as pc2
    
    from perception.msg import AssociationArray, DetectionArray
    from perception_lib import perception_utils
    from perception_lib import lane_utils

except ImportError as e:
    print(f"\n[GUI Error] 라이브러리 로드 실패: {e}")
    sys.exit(1)


# ==============================================================================
# UI Classes (Canvas & Editor)
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
        if len(self._editing_pts) < 3: return
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
        qimg = QtGui.QImage(self.bg_img.data, self.w, self.h, 3*self.w, QtGui.QImage.Format_BGR888)
        p.drawImage(0, 0, qimg)
        for name, poly in self._lane_polys.items():
            if poly is None or len(poly) == 0: continue
            if name == self._current_lane_name:
                p.setPen(QtGui.QPen(QtGui.QColor(255, 50, 50), 3))
            else:
                p.setPen(self.pen_boundary)
            p.setBrush(self.brush_fill)
            path = QtGui.QPainterPath()
            pts = [QtCore.QPoint(int(x), int(y)) for x, y in poly]
            if not pts: continue
            path.moveTo(pts[0])
            for pt in pts[1:]: path.lineTo(pt)
            path.closeSubpath()
            p.drawPath(path)
            if len(poly) > 0:
                cx, cy = int(np.mean(poly[:,0])), int(np.mean(poly[:,1]))
                p.setPen(QtGui.QColor(255, 255, 255))
                p.setFont(self.font_label)
                p.drawText(cx, cy, name)
        if self._editing_pts:
            p.setPen(self.pen_editing)
            pts = [QtCore.QPoint(int(x), int(y)) for x, y in self._editing_pts]
            for i in range(len(pts)-1): p.drawLine(pts[i], pts[i+1])
            for pt in pts: p.drawEllipse(pt, 3, 3)

class LaneEditorDialog(QtWidgets.QDialog):
    def __init__(self, bg_img, current_polys, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Lane Polygon Editor (Pop-up)")
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
    def get_polys(self): return self.polys

class ImageCanvasViewer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = None
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #101010;")
    def update_image(self, cv_img):
        if cv_img is None: return
        h, w, ch = cv_img.shape
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        q_img = QtGui.QImage(rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888)
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
# 메인 GUI Class
# ==============================================================================
class RealWorldGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Motrex-SKKU Sensor Fusion GUI")
        self.resize(1280, 850)

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
        self.Extr_t = np.zeros((3,1))
        self.vis_objects = [] 
        
        # [신규 추가] ID 매핑을 위한 변수
        self.lane_counters = {name: 0 for name in ["IN1", "IN2", "IN3", "OUT1", "OUT2", "OUT3"]}
        self.global_to_local_ids = {} # {global_id: local_id}

        self.load_extrinsic()
        self.load_lane_polys()
        self.init_ui()

        rospy.Subscriber("/camera/image_raw", Image, self.cb_image, queue_size=1, buff_size=2**24)
        rospy.Subscriber("/point_cloud", PointCloud2, self.cb_radar, queue_size=1)
        rospy.Subscriber("/camera/camera_info", CameraInfo, self.cb_info, queue_size=1)
        rospy.Subscriber("/perception/output", AssociationArray, self.cb_final_result, queue_size=1)
        rospy.Subscriber("/perception/tracks", DetectionArray, self.cb_tracker_result, queue_size=1)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(33)
        self.last_update_time = time.time()

    def init_ui(self):
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        self.viewer = ImageCanvasViewer()
        layout.addWidget(self.viewer, stretch=4)
        
        panel = QtWidgets.QWidget(); panel.setFixedWidth(280)
        vbox = QtWidgets.QVBoxLayout(panel); vbox.setAlignment(Qt.AlignTop)
        
        gb_calib = QtWidgets.QGroupBox("1. Calibration")
        v_c = QtWidgets.QVBoxLayout()
        btn_calib = QtWidgets.QPushButton("Run Calibration")
        btn_calib.clicked.connect(self.run_calibration)
        btn_reload = QtWidgets.QPushButton("Reload JSON")
        btn_reload.clicked.connect(self.load_extrinsic)
        v_c.addWidget(btn_calib); v_c.addWidget(btn_reload)
        gb_calib.setLayout(v_c); vbox.addWidget(gb_calib)

        gb_lane = QtWidgets.QGroupBox("2. Lane Editor")
        v_l = QtWidgets.QVBoxLayout()
        btn_edit = QtWidgets.QPushButton("🖌️ Open Editor (Pop-up)")
        btn_edit.setStyleSheet("background-color: #FFD700; color: black; font-weight: bold; padding: 10px;")
        btn_edit.clicked.connect(self.open_lane_editor)
        v_l.addWidget(btn_edit)
        gb_lane.setLayout(v_l); vbox.addWidget(gb_lane)

        gb_vis = QtWidgets.QGroupBox("3. View Options")
        v_vis = QtWidgets.QVBoxLayout()
        
        self.chk_show_poly = QtWidgets.QCheckBox("Show Lane Polygons (ROI)")
        self.chk_show_poly.setChecked(True)
        v_vis.addWidget(self.chk_show_poly)
        v_vis.addWidget(QtWidgets.QLabel("--- Filter Lane (Box & Poly) ---"))

        self.chk_lanes = {}
        display_order = ["IN1", "OUT1", "IN2", "OUT2", "IN3", "OUT3"]
        g_vis = QtWidgets.QGridLayout()
        for i, name in enumerate(display_order):
            chk = QtWidgets.QCheckBox(name)
            chk.setChecked(True)
            self.chk_lanes[name] = chk
            g_vis.addWidget(chk, i//2, i%2)
            
        v_vis.addLayout(g_vis)
        
        # [추가] ID 리셋 버튼
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
        """로컬 ID 카운터 및 매핑 초기화"""
        self.lane_counters = {name: 0 for name in self.lane_counters.keys()}
        self.global_to_local_ids = {}
        self.lbl_log.setText("Local IDs Reset.")

    # ------------------ Callback Logic ------------------
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

    def cb_image(self, msg):
        try:
            f = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if len(f.shape) == 2: f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
            self.cv_image = f
        except: pass
    
    def cb_radar(self, msg):
        try:
            g = pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True)
            self.radar_points = np.array(list(g))
        except: pass

    def cb_info(self, msg):
        if self.cam_K is None: self.cam_K = np.array(msg.K).reshape(3,3)

    def get_text_position(self, pts):
        sorted_indices = np.argsort(pts[:, 1])[::-1] 
        bottom_two = pts[sorted_indices[:2]] 
        left_idx = np.argmin(bottom_two[:, 0])
        return tuple(bottom_two[left_idx])
    
    # ------------------ Update Loop (Rendering) ------------------
    def update_loop(self):
        if self.cv_image is None: return
        disp = self.cv_image.copy()

        # 1. Objects Draw
        for obj in self.vis_objects:
            g_id = obj['id'] # 트래커의 글로벌 ID
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
            
            # [필터링 로직] 활성화된 차선에 있는 경우만 표시
            if target_lane:
                # [로컬 ID 매핑] 처음 보는 차라면 해당 차선 번호 부여
                if g_id not in self.global_to_local_ids:
                    self.lane_counters[target_lane] += 1
                    self.global_to_local_ids[g_id] = self.lane_counters[target_lane]
                
                local_id = self.global_to_local_ids[g_id]

                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                vel = obj['vel']
                v_str = f"{int(vel)}km/h" if np.isfinite(vel) else "--km/h"
                
                # 글로벌 ID 대신 로컬 ID(local_id) 표시
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

        # 2. Radar
        if self.radar_points is not None and self.cam_K is not None:
            pts_r = self.radar_points.T
            pts_c = self.Extr_R @ pts_r + self.Extr_t.reshape(3,1)
            valid = pts_c[2, :] > 0.5
            if np.any(valid):
                pts_c = pts_c[:, valid]
                uvs = self.cam_K @ pts_c
                uvs /= uvs[2, :]
                h, w = disp.shape[:2]
                for i in range(uvs.shape[1]):
                    u, v = int(uvs[0, i]), int(uvs[1, i])
                    if 0 <= u < w and 0 <= v < h:
                        cv2.circle(disp, (u, v), 2, (0, 255, 255), -1)

        # 3. Lane Polygons
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
        if self.cv_image is None: return
        dlg = LaneEditorDialog(self.cv_image, self.lane_polys, self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            self.lane_polys = dlg.get_polys()
            self.save_lane_polys()
            self.lbl_log.setText("Lane Polygons Updated.")

    def save_lane_polys(self):
        lane_utils.save_lane_polys(self.lane_json_path, self.lane_polys)

    def load_lane_polys(self):
        try: self.lane_polys = lane_utils.load_lane_polys(self.lane_json_path)
        except: self.lane_polys = {}

    def load_extrinsic(self):
        if os.path.exists(self.extrinsic_path):
            try:
                with open(self.extrinsic_path, 'r') as f:
                    data = json.load(f)
                self.Extr_R = np.array(data['R'])
                self.Extr_t = np.array(data['t'])
                self.lbl_log.setText("Extrinsic Loaded.")
            except: pass

    def run_calibration(self):
        calib_path = os.path.join(self.nodes_dir, "..", "perception_lib", "calibration_manager.py")
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