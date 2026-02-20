#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GUI 캘리브레이션 실행 제어
- extrinsic/intrinsic 프로세스 실행 관리
- 로그 UI 초기화 및 ROS 명령 구성
"""

import json
import os
import time
import traceback

import numpy as np
from PySide6 import QtCore
import rospkg

# 외부파라미터 JSON 로드 후 GUI 상태 갱신
def load_extrinsic(gui):
    if os.path.exists(gui.extrinsic_path):
        try:
            with open(gui.extrinsic_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            gui.Extr_R = np.array(data["R"])
            gui.Extr_t = np.array(data["t"])
            gui.extrinsic_mtime = os.path.getmtime(gui.extrinsic_path)
            gui.extrinsic_last_loaded = time.time()
        except Exception:
            pass


# 전달된 다이얼로그 클래스를 생성해 모달 실행
def run_calibration(gui, dialog_cls):
    dlg = dialog_cls(gui)
    dlg.exec()


# Extrinsic 캘리브레이션 노드를 백그라운드에서 실행
def run_autocalibration(gui):
    try:
        # 중복 실행 방지
        if hasattr(gui, "extrinsic_proc") and gui.extrinsic_proc is not None:
            if gui.extrinsic_proc.state() != QtCore.QProcess.NotRunning:
                print("[Extrinsic] already running...")
                return

        # 실행 프로세스 및 환경 설정
        gui.extrinsic_proc = QtCore.QProcess()
        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("QT_QPA_PLATFORM", "offscreen")  # GUI 없이 백그라운드 실행
        env.remove("DISPLAY")  # 디스플레이 출력 비활성화
        gui.extrinsic_proc.setProcessEnvironment(env)
        gui.extrinsic_proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)

        # UI 로그/진행상태 초기화
        gui.txt_extrinsic_log.clear()
        gui.txt_extrinsic_log.appendPlainText("[Extrinsic] START")
        gui.pbar_extrinsic.setVisible(False)
        gui.pbar_extrinsic.setRange(0, 0)

        ws = os.path.expanduser("~/motrex/catkin_ws")
        setup_bash = os.path.join(ws, "devel", "setup.bash")

        # ROS 실행 명령 구성
        cmd = "bash"
        ros_cmd = (
            f"source {setup_bash} && "
            f"rosrun autocal calibration_ex_node.py "
            f"_extrinsic_path:={gui.extrinsic_path} "
            f"_bbox_ref_mode:={gui.bbox_ref_mode}"
        )
        args = ["-c", ros_cmd]
        gui.extrinsic_proc.readyReadStandardOutput.connect(gui._on_extrinsic_stdout)
        gui.extrinsic_proc.readyReadStandardError.connect(gui._on_extrinsic_stderr)
        gui.extrinsic_proc.finished.connect(gui._on_extrinsic_finished)
        gui.extrinsic_proc.start(cmd, args)
    except Exception as exc:
        print(f"[Extrinsic] start error: {exc}")
        traceback.print_exc()


# Intrinsic 캘리브레이션 노드를 백그라운드에서 실행
def run_intrinsic_calibration(gui):
    try:
        rp = rospkg.RosPack()
        pkg_path = rp.get_path("autocal")
        image_dir = os.path.join(pkg_path, "image")
        out_json = os.path.join(pkg_path, "config", "intrinsic.json")

        # 입력 데이터 유효성 확인
        if not os.path.isdir(image_dir):
            print(f"[Intrinsic] image dir not found: {image_dir}")
            return

        import glob

        imgs = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        if len(imgs) == 0:
            print(f"[Intrinsic] No .jpg in {image_dir}")
            return

        # 중복 실행 방지
        if hasattr(gui, "intrinsic_proc") and gui.intrinsic_proc is not None:
            if gui.intrinsic_proc.state() != QtCore.QProcess.NotRunning:
                print("[Intrinsic] already running...")
                return

        # UI 로그/진행상태 초기화
        gui.txt_intrinsic_log.clear()
        gui.txt_intrinsic_log.appendPlainText("[Intrinsic] START")
        gui.pbar_intrinsic.setVisible(False)
        gui.pbar_intrinsic.setRange(0, 0)

        # 실행 프로세스 및 환경 설정
        gui.intrinsic_proc = QtCore.QProcess()
        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("QT_QPA_PLATFORM", "offscreen")
        env.remove("DISPLAY")
        gui.intrinsic_proc.setProcessEnvironment(env)
        gui.intrinsic_proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)

        ws = os.path.expanduser("~/motrex/catkin_ws")
        setup_bash = os.path.join(ws, "devel", "setup.bash")

        # ROS 실행 명령 구성
        cmd = "bash"
        ros_cmd = (
            f"source {setup_bash} && "
            f"rosrun autocal calibration_in_node.py "
            f"_image_dir:={image_dir} "
            f"_output_path:={out_json} "
            f"_board/width:=10 _board/height:=7 _board/square_size:=0.025 "
            f"_min_samples:=20 _max_selected:=45 _grid_size_mm:=50 "
            f"_save_selected:=true _save_undistort:=true"
        )

        args = ["-c", ros_cmd]
        gui.intrinsic_proc.readyReadStandardOutput.connect(gui._on_intrinsic_stdout)
        gui.intrinsic_proc.readyReadStandardError.connect(gui._on_intrinsic_stderr)
        gui.intrinsic_proc.finished.connect(gui._on_intrinsic_finished)
        gui.intrinsic_proc.start(cmd, args)
    except Exception as exc:
        print(f"[Intrinsic] start error: {exc}")
        traceback.print_exc()