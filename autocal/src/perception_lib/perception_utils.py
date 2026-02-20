#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2

def project_radar_to_image(radar_xyz, K, D, R, t, img_w, img_h):
    """
    레이더 3D 포인트들을 카메라 이미지 2D 좌표로 투영합니다.
    
    Args:
        radar_xyz (np.ndarray): (N, 3) 레이더 좌표계 점들 (x, y, z)
        K (np.ndarray): (3, 3) 카메라 내부 파라미터 (Intrinsic)
        D (np.ndarray): (5,) 또는 (N,) 왜곡 계수 (Distortion)
        R (np.ndarray): (3, 3) 회전 행렬 (Extrinsic Rotation)
        t (np.ndarray): (3, 1) 평행이동 벡터 (Extrinsic Translation)
        img_w (int): 이미지 너비
        img_h (int): 이미지 높이
        
    Returns:
        uv_valid (np.ndarray): (M, 2) 이미지 내부에 들어오는 픽셀 좌표 (정수형)
        valid_indices (np.ndarray): (M,) 유효한 점들의 원본 인덱스
    """
    if radar_xyz is None or len(radar_xyz) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=int)

    # 1. Extrinsic Transform (Radar -> Camera Coordinate)
    # P_cam = R * P_rad + t
    # (N, 3) @ (3, 3).T + (1, 3)
    pts_cam = (radar_xyz @ R.T) + t.reshape(1, 3)

    # 2. Z축 필터링 (카메라 뒤쪽에 있는 점 제거)
    # 일반적으로 카메라 좌표계에서 Z가 전방
    valid_z = pts_cam[:, 2] > 0.1
    
    # 인덱스 추적을 위해
    indices = np.arange(len(radar_xyz))
    pts_cam = pts_cam[valid_z]
    indices = indices[valid_z]

    if len(pts_cam) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=int)

    # 3. Project to 2D (Intrinsic + Distortion)
    # cv2.projectPoints는 (N, 1, 3) 형태를 기대함
    pts_cam_reshaped = pts_cam.reshape(-1, 1, 3)
    
    # rvec, tvec는 0으로 설정 (이미 pts_cam이 카메라 좌표계 기준이므로)
    zero_vec = np.zeros((3, 1), dtype=float)
    
    image_points, _ = cv2.projectPoints(
        pts_cam_reshaped, 
        zero_vec, 
        zero_vec, 
        K, 
        D
    )
    
    image_points = image_points.reshape(-1, 2)

    # 4. 이미지 경계 확인 (FOV 필터링)
    u = image_points[:, 0]
    v = image_points[:, 1]

    mask = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    
    uv_valid = image_points[mask].astype(int)
    valid_indices = indices[mask]

    return uv_valid, valid_indices

def iou(bbox1, bbox2):
    """
    두 BBox의 IoU(Intersection over Union) 계산
    bbox: [x1, y1, x2, y2]
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    union = area1 + area2 - intersection
    if union <= 0: return 0.0
    return intersection / union