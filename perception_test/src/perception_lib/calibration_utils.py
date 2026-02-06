import json
import os
import time
import traceback

import numpy as np
from PySide6 import QtCore
import rospkg

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


def run_calibration(gui, dialog_cls):
    dlg = dialog_cls(gui)
    dlg.exec()


def run_autocalibration(gui):
    try:
        if hasattr(gui, "extrinsic_proc") and gui.extrinsic_proc is not None:
            if gui.extrinsic_proc.state() != QtCore.QProcess.NotRunning:
                print("[Extrinsic] already running...")
                return

        gui.extrinsic_proc = QtCore.QProcess()
        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("QT_QPA_PLATFORM", "offscreen")  # Qt가 화면을 그리지 않고 백그라운드에서만 돌게 함
        env.remove("DISPLAY")  # 디스플레이 접근 차단
        gui.extrinsic_proc.setProcessEnvironment(env)
        gui.extrinsic_proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)

        gui.txt_extrinsic_log.clear()
        gui.txt_extrinsic_log.appendPlainText("[Extrinsic] START")
        gui.pbar_extrinsic.setVisible(False)
        gui.pbar_extrinsic.setRange(0, 0)

        ws = os.path.expanduser("~/motrex/catkin_ws")
        setup_bash = os.path.join(ws, "devel", "setup.bash")

        cmd = "bash"
        ros_cmd = (
            f"source {setup_bash} && "
            f"rosrun perception_test calibration_ex_node.py "
            f"_extrinsic_path:={gui.extrinsic_path}"
        )
        args = ["-c", ros_cmd]
        gui.extrinsic_proc.readyReadStandardOutput.connect(gui._on_extrinsic_stdout)
        gui.extrinsic_proc.readyReadStandardError.connect(gui._on_extrinsic_stderr)
        gui.extrinsic_proc.finished.connect(gui._on_extrinsic_finished)
        gui.extrinsic_proc.start(cmd, args)
    except Exception as exc:
        print(f"[Extrinsic] start error: {exc}")
        traceback.print_exc()


def run_intrinsic_calibration(gui):
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

        if hasattr(gui, "intrinsic_proc") and gui.intrinsic_proc is not None:
            if gui.intrinsic_proc.state() != QtCore.QProcess.NotRunning:
                print("[Intrinsic] already running...")
                return

        gui.txt_intrinsic_log.clear()
        gui.txt_intrinsic_log.appendPlainText("[Intrinsic] START")
        gui.pbar_intrinsic.setVisible(False)
        gui.pbar_intrinsic.setRange(0, 0)

        gui.intrinsic_proc = QtCore.QProcess()
        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("QT_QPA_PLATFORM", "offscreen")
        env.remove("DISPLAY")
        gui.intrinsic_proc.setProcessEnvironment(env)
        gui.intrinsic_proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)

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

        args = ["-c", ros_cmd]
        gui.intrinsic_proc.readyReadStandardOutput.connect(gui._on_intrinsic_stdout)
        gui.intrinsic_proc.readyReadStandardError.connect(gui._on_intrinsic_stderr)
        gui.intrinsic_proc.finished.connect(gui._on_intrinsic_finished)
        gui.intrinsic_proc.start(cmd, args)
    except Exception as exc:
        print(f"[Intrinsic] start error: {exc}")
        traceback.print_exc()
