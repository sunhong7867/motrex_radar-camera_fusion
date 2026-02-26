#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Batch AutoCal Runner (real_gui cycle mode).

요청 반영 동작(매 iteration):
1) extrinsic 초기값 복원
2) real_gui 포함 파이프라인 실행
3) real_gui에서 N초 후 자동으로 Run Extrinsic(Auto) 버튼 동작
4) 진단 결과 수신/CSV 저장
5) 창/파이프라인 종료
"""

import atexit
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import signal
import time
import shlex
from datetime import datetime
from typing import Dict, List, Tuple

import rospy
import rospkg
from std_msgs.msg import String


class DiagnosisCache:
    def __init__(self):
        self._lock = threading.Lock()
        self._seq = 0
        self._text = ""

    def cb(self, msg: String):
        with self._lock:
            self._seq += 1
            self._text = str(msg.data)

    def snapshot(self) -> Tuple[int, str]:
        with self._lock:
            return self._seq, self._text


def _resolve_defaults() -> Tuple[str, str, str]:
    rp = rospkg.RosPack()
    pkg_path = rp.get_path("autocal")
    cfg = os.path.join(pkg_path, "config")
    return (
        os.path.join(cfg, "extrinsic_initial.json"),
        os.path.join(cfg, "extrinsic.json"),
        os.path.join(pkg_path, "calibration_logs", "batch_autocal_results.csv"),
    )


def _safe_float(s: str):
    try:
        return float(s)
    except Exception:
        return None


def _parse_diag_text(text: str) -> Dict[str, object]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    out: Dict[str, object] = {
        "valid_tracks": None,
        "rmse_abs_mean": None,
        "rmse_abs_median": None,
        "rmse_db_mean": None,
        "rmse_db_median": None,
        "ru_abs_mean": None,
        "rv_abs_mean": None,
        "angle_diff_mean_deg": None,
        "global_bias_u": None,
        "global_bias_v": None,
        "distance_bins": {},
        "lane_rmse": {},
        "distance_bias_v": {},
    }

    key_patterns = {
        "valid_tracks": re.compile(r"^- 유효 트랙 수:\s*([0-9]+)"),
        "rmse_abs_mean": re.compile(r"^- 평균 RMSE\(abs, px\):\s*([-0-9.]+)"),
        "rmse_abs_median": re.compile(r"^- 중앙값 RMSE\(abs, px\):\s*([-0-9.]+)"),
        "rmse_db_mean": re.compile(r"^- 평균 RMSE\(debiased, px\):\s*([-0-9.]+)"),
        "rmse_db_median": re.compile(r"^- 중앙값 RMSE\(debiased, px\):\s*([-0-9.]+)"),
        "ru_abs_mean": re.compile(r"^- \|ru\| 평균:\s*([-0-9.]+)"),
        "rv_abs_mean": re.compile(r"^- \|rv\| 평균:\s*([-0-9.]+)"),
        "angle_diff_mean_deg": re.compile(r"^- 평균 방향 각도 차\(°\):\s*([-0-9.]+)"),
    }
    re_global = re.compile(r"^- 전역 Bias\(u,v\) px:\s*\(([-0-9.]+),\s*([-0-9.]+)\)")
    re_dist = re.compile(
        r"^-\s*([0-9]+-[0-9]+m):\s*RMSE\s*([-0-9.]+)px,\s*Bias\(([-0-9.]+),([-0-9.]+)\)\s*\(N=([0-9]+)\)"
    )
    re_lane = re.compile(
        r"^-\s*([A-Za-z0-9_]+):\s*RMSE\s*([-0-9.]+)px,\s*Bias\(([-0-9.]+),([-0-9.]+)\)\s*\(N=([0-9]+)\)"
    )

    for ln in lines:
        for k, rgx in key_patterns.items():
            m = rgx.search(ln)
            if m:
                out[k] = _safe_float(m.group(1))

        m = re_global.search(ln)
        if m:
            out["global_bias_u"] = _safe_float(m.group(1))
            out["global_bias_v"] = _safe_float(m.group(2))

        m = re_dist.search(ln)
        if m:
            key = m.group(1)
            rmse = _safe_float(m.group(2))
            bu = _safe_float(m.group(3))
            bv = _safe_float(m.group(4))
            n = int(m.group(5))
            out["distance_bins"][key] = {"rmse_abs": rmse, "bias_u": bu, "bias_v": bv, "n": n}
            out["distance_bias_v"][key] = bv

        m = re_lane.search(ln)
        if m:
            lane = m.group(1)
            rmse = _safe_float(m.group(2))
            bu = _safe_float(m.group(3))
            bv = _safe_float(m.group(4))
            n = int(m.group(5))
            out["lane_rmse"][lane] = {"rmse_abs": rmse, "bias_u": bu, "bias_v": bv, "n": n}

    return out


def _load_extrinsic(path: str) -> Tuple[List[List[float]], List[float]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("R", []), data.get("t", [])
    except Exception:
        return [], []


def _rewrite_extrinsic_pretty(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        payload = {
            "R": data.get("R", []),
            "t": data.get("t", []),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=4)
    except Exception:
        pass


def _fmt_rt(mat: List[List[float]], vec: List[float]) -> str:
    r = mat if mat else []
    t = vec if vec else []

    if not r:
        r_block = "[]"
    else:
        row_lines = [json.dumps(row, ensure_ascii=False) for row in r]
        r_block = "[" + row_lines[0]
        for row in row_lines[1:]:
            r_block += ",\n          " + row
        r_block += "]"

    t_block = json.dumps(t, ensure_ascii=False)
    return "{\n    \"R\": " + r_block + ",\n    \"t\": " + t_block + "\n}"


def _fmt_distance_bins(distance_bins: Dict[str, Dict[str, float]]) -> str:
    if not distance_bins:
        return ""
    rows = []
    for k in sorted(distance_bins.keys(), key=lambda x: int(x.split("-")[0])):
        d = distance_bins[k]
        rows.append(f"{k}: RMSE {d.get('rmse_abs')}, Bias({d.get('bias_u')},{d.get('bias_v')}), N={d.get('n')}")
    return "\n".join(rows)


def _fmt_lane_rmse(lane_rmse: Dict[str, Dict[str, float]]) -> str:
    if not lane_rmse:
        return ""
    rows = []
    for lane in sorted(lane_rmse.keys()):
        d = lane_rmse[lane]
        rows.append(f"{lane}: RMSE {d.get('rmse_abs')}, Bias({d.get('bias_u')},{d.get('bias_v')}), N={d.get('n')}")
    return "\n".join(rows)


def _fmt_dist_bias_v(distance_bias_v: Dict[str, float]) -> str:
    if not distance_bias_v:
        return ""
    rows = []
    for k in sorted(distance_bias_v.keys(), key=lambda x: int(x.split("-")[0])):
        rows.append(f"{k}: {distance_bias_v[k]}")
    return "\n".join(rows)


def _ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _prepare_csv_schema(csv_path: str, fieldnames: List[str]):
    if not os.path.exists(csv_path):
        return

    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            header_line = f.readline().strip()
    except Exception:
        return

    if not header_line:
        return

    if "	" in header_line and "," not in header_line:
        existing = [h.strip() for h in header_line.split("	")]
    else:
        existing = [h.strip() for h in header_line.split(",")]

    target = list(fieldnames)
    if existing == target:
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"{csv_path}.schema_mismatch_{ts}.bak"
    shutil.move(csv_path, backup)
    rospy.logwarn(f"[BatchAutoCal] existing csv schema mismatch. moved old file -> {backup}")


def _append_csv(csv_path: str, row: Dict[str, object]):
    fieldnames = [
        "timestamp",
        "iter_idx",
        "iterations_total",
        "bbox_ref_mode",
        "calib_exit_code",
        "Rt_multiline",
        "valid_tracks",
        "rmse_abs_mean",
        "rmse_abs_median",
        "rmse_db_mean",
        "rmse_db_median",
        "ru_abs_mean",
        "rv_abs_mean",
        "angle_diff_mean_deg",
        "global_bias_u",
        "global_bias_v",
        "distance_bins_text",
        "lane_rmse_text",
        "distance_bias_v_text",
    ]
    _ensure_parent(csv_path)
    _prepare_csv_schema(csv_path, fieldnames)
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            wr.writeheader()
        wr.writerow(row)


def _wait_new_diag(cache: DiagnosisCache, prev_seq: int, timeout_sec: float) -> Tuple[bool, str]:
    deadline = rospy.Time.now().to_sec() + max(0.0, timeout_sec)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown() and rospy.Time.now().to_sec() <= deadline:
        seq, text = cache.snapshot()
        if seq > prev_seq:
            return True, text
        rate.sleep()
    return False, ""


def _start_pipeline_for_iteration(
    launch_file: str,
    source_name: str,
    autostart_delay_sec: float,
    autostart_min_tracks: int,
):
    cmd = [
        "roslaunch", "autocal", launch_file,
        "launch_real_gui:=true",
        f"offline_source_name:={source_name}",
        f"autostart_extrinsic_delay_sec:={autostart_delay_sec}",
        f"autostart_min_tracks:={autostart_min_tracks}",
    ]
    return subprocess.Popen(cmd, start_new_session=True)


def _stop_proc(proc):
    if proc is None:
        return
    if proc.poll() is not None:
        return

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        proc.terminate()

    try:
        proc.wait(timeout=8.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()
        try:
            proc.wait(timeout=3.0)
        except Exception:
            pass


def _close_gui_window_alt_f4_style(window_title: str):
    title = (window_title or "").strip()
    if not title:
        return

    # 1) wmctrl close by title
    try:
        subprocess.run(["wmctrl", "-c", title],
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,
                       check=False)
    except Exception:
        pass

    # 2) xdotool windowclose / Alt+F4 for all matched windows
    try:
        cmd = f"xdotool search --name {shlex.quote(title)}"
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
        ids = [ln.strip() for ln in (res.stdout or "").splitlines() if ln.strip()]
        for wid in ids:
            subprocess.run(["xdotool", "windowactivate", wid],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,
                           check=False)
            subprocess.run(["xdotool", "key", "--window", wid, "Alt+F4"],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,
                           check=False)
            subprocess.run(["xdotool", "windowclose", wid],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,
                           check=False)
    except Exception:
        pass


def _force_close_real_gui_node():
    for node_name in ["/real_gui", "/real_gui_node"]:
        try:
            subprocess.run(["rosnode", "kill", node_name],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,
                           check=False)
        except Exception:
            pass


def _kill_lingering_gui_processes():
    patterns = [
        "autocal/src/real_gui.py",
        "devel/lib/autocal/real_gui.py",
        "__name:=real_gui",
        "real_gui_node",
    ]

    for pat in patterns:
        try:
            subprocess.run(["pkill", "-f", pat], check=False)
        except Exception:
            pass

    time.sleep(0.3)
    for pat in patterns:
        try:
            subprocess.run(["pkill", "-9", "-f", pat], check=False)
        except Exception:
            pass


def _init_node_or_exit():
    try:
        rospy.init_node("autocal_batch_runner", anonymous=False)
    except Exception as exc:
        print("[BatchAutoCal] ERROR: ROS master connection failed.", file=sys.stderr)
        print(f"[BatchAutoCal] details: {exc}", file=sys.stderr)
        sys.exit(1)


def main():
    _init_node_or_exit()

    d_initial, d_target, d_csv = _resolve_defaults()
    iterations = int(rospy.get_param("~iterations", 100))
    bbox_ref_mode = str(rospy.get_param("~bbox_ref_mode", "bottom"))
    diag_topic = str(rospy.get_param("~diagnosis_topic", "/autocal/diagnosis/result"))

    # real_gui 버튼 자동 실행 지연(요청: 10초)
    bag_warmup_sec = float(rospy.get_param("~bag_warmup_sec", 10.0))
    autostart_min_tracks = int(rospy.get_param("~autostart_min_tracks", 1))
    # real_gui 자동실행 + 진단 완료까지 기다릴 총 시간
    diagnosis_timeout_sec = float(rospy.get_param("~diagnosis_timeout_sec", 240.0))
    min_valid_tracks_for_success = int(rospy.get_param("~min_valid_tracks_for_success", 20))
    gui_window_title = str(rospy.get_param("~gui_window_title", "Motrex-SKKU Sensor Fusion GUI"))

    pipeline_launch_file = str(rospy.get_param("~pipeline_launch_file", "autocal_dev.launch"))
    offline_source_name = str(rospy.get_param("~offline_source_name", "default"))

    extrinsic_initial_path = str(rospy.get_param("~extrinsic_initial_path", d_initial))
    extrinsic_path = str(rospy.get_param("~extrinsic_path", d_target))
    output_csv = str(rospy.get_param("~output_csv", d_csv))

    if not os.path.exists(extrinsic_initial_path):
        rospy.logerr(f"[BatchAutoCal] initial extrinsic not found: {extrinsic_initial_path}")
        sys.exit(2)

    rospy.loginfo(f"[BatchAutoCal] start iterations={iterations}, csv={output_csv}, ref={bbox_ref_mode}")

    cache = DiagnosisCache()
    rospy.Subscriber(diag_topic, String, cache.cb, queue_size=5)
    rospy.sleep(0.2)

    saved = 0
    for i in range(1, iterations + 1):
        if rospy.is_shutdown():
            break

        print(f"[BatchAutoCal] {i}/{iterations}", flush=True)

        shutil.copyfile(extrinsic_initial_path, extrinsic_path)
        _rewrite_extrinsic_pretty(extrinsic_path)

        prev_seq, _ = cache.snapshot()

        pipeline_proc = _start_pipeline_for_iteration(
            pipeline_launch_file,
            offline_source_name,
            bag_warmup_sec,
            autostart_min_tracks,
        )
        atexit.register(_stop_proc, pipeline_proc)

        got_diag, diag_text = _wait_new_diag(cache, prev_seq, diagnosis_timeout_sec)
        if not got_diag:
            rospy.logwarn(f"[BatchAutoCal] iter {i}: diagnosis timeout ({diagnosis_timeout_sec}s)")

        _close_gui_window_alt_f4_style(gui_window_title)
        _force_close_real_gui_node()
        _stop_proc(pipeline_proc)
        _close_gui_window_alt_f4_style(gui_window_title)
        _force_close_real_gui_node()
        _kill_lingering_gui_processes()

        _rewrite_extrinsic_pretty(extrinsic_path)
        parsed = _parse_diag_text(diag_text)
        valid_tracks = parsed.get("valid_tracks")
        calib_code = 0 if got_diag else -1
        if got_diag and (valid_tracks is None or int(valid_tracks) < min_valid_tracks_for_success):
            calib_code = -1
            rospy.logwarn(
                f"[BatchAutoCal] iter {i}: valid_tracks={valid_tracks} < {min_valid_tracks_for_success}; mark exit=-1"
            )

        r_mat, t_vec = _load_extrinsic(extrinsic_path)

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "iter_idx": i,
            "iterations_total": iterations,
            "bbox_ref_mode": bbox_ref_mode,
            "calib_exit_code": calib_code,
            "Rt_multiline": _fmt_rt(r_mat, t_vec),
            "valid_tracks": parsed["valid_tracks"],
            "rmse_abs_mean": parsed["rmse_abs_mean"],
            "rmse_abs_median": parsed["rmse_abs_median"],
            "rmse_db_mean": parsed["rmse_db_mean"],
            "rmse_db_median": parsed["rmse_db_median"],
            "ru_abs_mean": parsed["ru_abs_mean"],
            "rv_abs_mean": parsed["rv_abs_mean"],
            "angle_diff_mean_deg": parsed["angle_diff_mean_deg"],
            "global_bias_u": parsed["global_bias_u"],
            "global_bias_v": parsed["global_bias_v"],
            "distance_bins_text": _fmt_distance_bins(parsed["distance_bins"]),
            "lane_rmse_text": _fmt_lane_rmse(parsed["lane_rmse"]),
            "distance_bias_v_text": _fmt_dist_bias_v(parsed["distance_bias_v"]),
        }
        _append_csv(output_csv, row)
        saved += 1

        if not got_diag:
            state = "timeout"
        elif calib_code == 0:
            state = "ok"
        else:
            state = f"low_tracks({valid_tracks})"
        print(f"[BatchAutoCal] iter {i}: exit={calib_code}, diagnosis={state}", flush=True)

    rospy.loginfo(f"[BatchAutoCal] done. saved_rows={saved} -> {output_csv}")


if __name__ == "__main__":
    main()
