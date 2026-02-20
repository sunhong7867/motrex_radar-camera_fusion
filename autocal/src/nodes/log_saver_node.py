#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
역할:
- GUI에서 누적한 레코드 버퍼를 순번 폴더에 CSV로 저장하고, ID별/거리구간별 통계 CSV를 함께 생성한다.
- 저장 폴더를 자동으로 증가시키며(`1`, `2`, `3` ...), 기존 로그를 덮어쓰지 않고 세션 단위로 분리 보관한다.
- 원본 데이터(`raw_main.csv`)와 분석용 요약(`id_analysis.csv`, `dist_analysis.csv`)을 일관된 포맷으로 내보낸다.

입력:
- `record_buffer` (`list[dict]`): 저장할 레코드 목록.
- `log_base_dir` (`str`, 선택): 로그 루트 디렉터리 경로.

출력:
- 반환값(`dict`): `status`, `folder`, `target_dir` 또는 에러 메시지.
- 파일 출력: `raw_main.csv`, `id_analysis.csv`, `dist_analysis.csv`.
"""

import os
from typing import Dict, Tuple

import pandas as pd

LOG_BASE_DIR = os.path.expanduser("~/motrex/catkin_ws/src/autocal/calibration_logs")  # 로그 루트 디렉터리


def _get_next_log_dir(log_base_dir: str) -> Tuple[str, int]:
    """
    다음 저장 순번 폴더를 생성하고 경로와 폴더 번호를 반환
    """
    if not os.path.exists(log_base_dir):
        os.makedirs(log_base_dir)

    existing = [d for d in os.listdir(log_base_dir) if d.isdigit()]
    next_num = max([int(d) for d in existing]) + 1 if existing else 1

    target_dir = os.path.join(log_base_dir, str(next_num))
    os.makedirs(target_dir, exist_ok=True)
    return target_dir, next_num


def save_sequential_data(record_buffer, log_base_dir: str = LOG_BASE_DIR) -> Dict[str, object]:
    """
    레코드 버퍼를 순번 폴더에 원본/요약 CSV 형태로 저장
    """
    if not record_buffer:
        return {"status": "no_data"}

    try:
        # 저장 디렉터리 생성 및 원본 CSV 저장
        target_dir, next_num = _get_next_log_dir(log_base_dir)
        df = pd.DataFrame(record_buffer)

        raw_path = os.path.join(target_dir, "raw_main.csv")
        df.to_csv(raw_path, index=False)

        # ID 기준 집계 통계 생성
        if "Unique_Key" in df.columns:
            stats_id = df.groupby("Unique_Key").agg({
                "Local_ID": "first",
                "Lane_Path": "first",
                "Final_Lane": "last",
                "Radar_Dist": ["count", "mean", "min", "max"],
                "Radar_Vel": "mean",
                "Track_Vel": "mean",
                "Score": "mean",
            }).reset_index()

            stats_id.columns = [
                "Unique_Key", "Local_ID", "Lane_Path", "Final_Lane",
                "Count", "Mean_Dist", "Min_Dist", "Max_Dist",
                "Mean_Radar_Vel", "Mean_Track_Vel", "Mean_Score",
            ]

            id_path = os.path.join(target_dir, "id_analysis.csv")
            stats_id.to_csv(id_path, index=False)

        # 거리 구간 기준 집계 통계 생성
        if "Radar_Dist" in df.columns:
            df["Dist_Bin"] = (df["Radar_Dist"] // 10) * 10
            stats_dist = df.groupby("Dist_Bin").agg({
                "Radar_Vel": ["count", "mean", "std"],
                "Track_Vel": ["mean", "std"],
                "Score": "mean",
            }).reset_index()

            stats_dist.columns = [
                "Dist_Bin_m",
                "Count", "Radar_Vel_Mean", "Radar_Vel_Std",
                "Track_Vel_Mean", "Track_Vel_Std",
                "Score_Mean",
            ]

            dist_path = os.path.join(target_dir, "dist_analysis.csv")
            stats_dist.to_csv(dist_path, index=False)

        return {"status": "saved", "folder": next_num, "target_dir": target_dir}

    except Exception as exc:
        return {"status": "error", "error": str(exc)}
