#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Dict, Tuple

import pandas as pd

LOG_BASE_DIR = os.path.expanduser("~/motrex/catkin_ws/src/autocal/calibration_logs")


def _get_next_log_dir(log_base_dir: str) -> Tuple[str, int]:
    if not os.path.exists(log_base_dir):
        os.makedirs(log_base_dir)

    existing = [d for d in os.listdir(log_base_dir) if d.isdigit()]
    if existing:
        next_num = max([int(d) for d in existing]) + 1
    else:
        next_num = 1

    target_dir = os.path.join(log_base_dir, str(next_num))
    os.makedirs(target_dir, exist_ok=True)
    return target_dir, next_num


def save_sequential_data(record_buffer, log_base_dir: str = LOG_BASE_DIR) -> Dict[str, object]:
    """GUI 로그 저장 유틸 (pandas.to_csv 사용으로 csv 모듈은 불필요)."""
    if not record_buffer:
        return {"status": "no_data"}

    try:
        target_dir, next_num = _get_next_log_dir(log_base_dir)
        df = pd.DataFrame(record_buffer)

        raw_path = os.path.join(target_dir, "raw_main.csv")
        df.to_csv(raw_path, index=False)

        if "Unique_Key" in df.columns:
            stats_id = df.groupby("Unique_Key").agg({
                "Local_ID": "first",
                "Lane_Path": "first",
                "Final_Lane": "last",
                "Radar_Dist": ["count", "mean", "min", "max"],
                "Radar_Vel": "mean",
                "Track_Vel": "mean",
                "Score": "mean"
            }).reset_index()

            stats_id.columns = [
                "Unique_Key", "Local_ID", "Lane_Path", "Final_Lane",
                "Count", "Mean_Dist", "Min_Dist", "Max_Dist",
                "Mean_Radar_Vel", "Mean_Track_Vel", "Mean_Score"
            ]

            id_path = os.path.join(target_dir, "id_analysis.csv")
            stats_id.to_csv(id_path, index=False)

        if "Radar_Dist" in df.columns:
            df["Dist_Bin"] = (df["Radar_Dist"] // 10) * 10
            stats_dist = df.groupby("Dist_Bin").agg({
                "Radar_Vel": ["count", "mean", "std"],
                "Track_Vel": ["mean", "std"],
                "Score": "mean"
            }).reset_index()

            stats_dist.columns = [
                "Dist_Bin_m",
                "Count", "Radar_Vel_Mean", "Radar_Vel_Std",
                "Track_Vel_Mean", "Track_Vel_Std",
                "Score_Mean"
            ]

            dist_path = os.path.join(target_dir, "dist_analysis.csv")
            stats_dist.to_csv(dist_path, index=False)

        return {"status": "saved", "folder": next_num, "target_dir": target_dir}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
