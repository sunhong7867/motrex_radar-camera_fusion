#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
역할:
- GUI에서 누적한 레코드 버퍼를 순번 폴더에 CSV로 저장하고, ID별/거리구간별 통계 CSV를 함께 생성한다.
- 저장 폴더를 자동으로 증가시키며(`1`, `2`, `3` ...), 기존 로그를 덮어쓰지 않고 세션 단위로 분리 보관한다.
- 원본 데이터(`raw_main.csv`)와 분석용 요약(`id_analysis.csv`, `dist_analysis.csv`)을 일관된 포맷으로 내보낸다.
- 픽셀 정합 기반 GT-free 평가 지표(RMSE/중앙값/방향성/일관성/정규화 바이어스) CSV를 추가 생성한다.

입력:
- `record_buffer` (`list[dict]`): 저장할 레코드 목록.
- `log_base_dir` (`str`, 선택): 로그 루트 디렉터리 경로.

출력:
- 반환값(`dict`): `status`, `folder`, `target_dir` 또는 에러 메시지.
- 파일 출력: `raw_main.csv`, `id_analysis.csv`, `dist_analysis.csv`, `eval_main.csv`, `eval_id_analysis.csv`, `eval_session_summary.csv`.
"""

import os
from typing import Dict, Optional, Tuple
import numpy as np
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

def _fit_direction(points_xy: np.ndarray) -> Optional[np.ndarray]:
    """2D 점 집합의 주성분 방향 벡터를 반환"""
    if points_xy is None or points_xy.shape[0] < 2:
        return None

    centered = points_xy - points_xy.mean(axis=0)
    cov = np.cov(centered.T)
    if cov.shape != (2, 2):
        return None

    eigvals, eigvecs = np.linalg.eigh(cov)
    vec = eigvecs[:, int(np.argmax(eigvals))]
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return None
    return vec / norm


def _build_eval_csvs(df: pd.DataFrame, target_dir: str) -> None:
    """
    기록 데이터프레임에서 평가 지표용 CSV를 생성

    생성 파일:
    - eval_main.csv: 프레임 단위 오차(매칭된 행 위주)
    - eval_id_analysis.csv: 트랙/ID 단위 RMSE, median, sigma, ru/rv, 방향각
    - eval_session_summary.csv: 세션 단위 중앙값/백분위 요약
    """
    needed = {
        "Unique_Key", "Global_ID", "Final_Lane", "Lane_Path",
        "Ref_U", "Ref_V", "Proj_U", "Proj_V", "Delta_U", "Delta_V",
        "Pixel_Error", "RU", "RV", "Is_Matched"
    }
    if not needed.issubset(set(df.columns)):
        return

    eval_df = df.copy()
    eval_df = eval_df[pd.to_numeric(eval_df["Is_Matched"], errors="coerce") > 0].copy()

    if eval_df.empty:
        return

    numeric_cols = ["Ref_U", "Ref_V", "Proj_U", "Proj_V", "Delta_U", "Delta_V", "Pixel_Error", "RU", "RV"]
    for col in numeric_cols:
        eval_df[col] = pd.to_numeric(eval_df[col], errors="coerce")

    eval_main_cols = [
        "Time", "Global_ID", "Local_ID", "Unique_Key", "Final_Lane", "Lane_Path",
        "Ref_U", "Ref_V", "Proj_U", "Proj_V", "Delta_U", "Delta_V",
        "Pixel_Error", "RU", "RV", "BBox_W", "BBox_H"
    ]
    eval_main_cols = [c for c in eval_main_cols if c in eval_df.columns]
    eval_main = eval_df[eval_main_cols].copy()
    eval_main.to_csv(os.path.join(target_dir, "eval_main.csv"), index=False)

    id_rows = []
    for key, g in eval_df.groupby("Unique_Key"):
        du = g["Delta_U"].to_numpy(dtype=np.float64)
        dv = g["Delta_V"].to_numpy(dtype=np.float64)
        e = g["Pixel_Error"].to_numpy(dtype=np.float64)

        rmse = float(np.sqrt(np.mean((du ** 2) + (dv ** 2)))) if du.size > 0 else np.nan
        med_err = float(np.nanmedian(e)) if e.size > 0 else np.nan
        p75_err = float(np.nanpercentile(e, 75)) if e.size > 0 else np.nan
        sigma_u = float(np.nanstd(du)) if du.size > 1 else np.nan
        sigma_v = float(np.nanstd(dv)) if dv.size > 1 else np.nan
        mean_ru = float(np.nanmean(g["RU"].to_numpy(dtype=np.float64)))
        mean_rv = float(np.nanmean(g["RV"].to_numpy(dtype=np.float64)))

        ref_pts = g[["Ref_U", "Ref_V"]].to_numpy(dtype=np.float64)
        proj_pts = g[["Proj_U", "Proj_V"]].to_numpy(dtype=np.float64)
        dir_ref = _fit_direction(ref_pts)
        dir_proj = _fit_direction(proj_pts)
        angle_deg = np.nan
        if dir_ref is not None and dir_proj is not None:
            dot = float(np.clip(np.abs(np.dot(dir_ref, dir_proj)), -1.0, 1.0))
            angle_deg = float(np.degrees(np.arccos(dot)))

        id_rows.append({
            "Unique_Key": key,
            "Global_ID": g["Global_ID"].iloc[0] if "Global_ID" in g.columns else np.nan,
            "Final_Lane": g["Final_Lane"].iloc[-1] if "Final_Lane" in g.columns else "",
            "Count": int(len(g)),
            "RMSE_px": rmse,
            "Median_Error_px": med_err,
            "P75_Error_px": p75_err,
            "Sigma_U_px": sigma_u,
            "Sigma_V_px": sigma_v,
            "Mean_RU": mean_ru,
            "Mean_RV": mean_rv,
            "Direction_Angle_Deg": angle_deg,
        })

    id_eval_df = pd.DataFrame(id_rows)
    id_eval_df.to_csv(os.path.join(target_dir, "eval_id_analysis.csv"), index=False)

    session_summary = {
        "Matched_Row_Count": int(len(eval_df)),
        "Valid_Track_Count": int(len(id_eval_df)),
        "Median_RMSE_px": float(np.nanmedian(id_eval_df["RMSE_px"])) if not id_eval_df.empty else np.nan,
        "P75_RMSE_px": float(np.nanpercentile(id_eval_df["RMSE_px"], 75)) if len(id_eval_df) > 0 else np.nan,
        "Median_Direction_Deg": float(np.nanmedian(id_eval_df["Direction_Angle_Deg"])) if not id_eval_df.empty else np.nan,
        "Median_Sigma_U_px": float(np.nanmedian(id_eval_df["Sigma_U_px"])) if not id_eval_df.empty else np.nan,
        "Median_Sigma_V_px": float(np.nanmedian(id_eval_df["Sigma_V_px"])) if not id_eval_df.empty else np.nan,
        "Median_Abs_RU": float(np.nanmedian(np.abs(id_eval_df["Mean_RU"]))) if not id_eval_df.empty else np.nan,
        "Median_Abs_RV": float(np.nanmedian(np.abs(id_eval_df["Mean_RV"]))) if not id_eval_df.empty else np.nan,
    }
    pd.DataFrame([session_summary]).to_csv(os.path.join(target_dir, "eval_session_summary.csv"), index=False)

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

        _build_eval_csvs(df, target_dir)
        
        return {"status": "saved", "folder": next_num, "target_dir": target_dir}

    except Exception as exc:
        return {"status": "error", "error": str(exc)}
