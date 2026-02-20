#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
레이더-카메라 기하 연산 유틸리티
- 3D 레이더 포인트 2D 투영
- bbox IoU 계산
"""

import numpy as np
import cv2


# 레이더 3D 포인트를 이미지 좌표로 투영하고 유효 인덱스 반환
def project_radar_to_image(radar_xyz, K, D, R, t, img_w, img_h):
    if radar_xyz is None or len(radar_xyz) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=int)

    # 1) Radar -> Camera 좌표계 변환
    pts_cam = (radar_xyz @ R.T) + t.reshape(1, 3)

    # 2) 카메라 전방(Z>0) 포인트만 선별
    valid_z = pts_cam[:, 2] > 0.1

    indices = np.arange(len(radar_xyz))
    pts_cam = pts_cam[valid_z]
    indices = indices[valid_z]

    if len(pts_cam) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=int)

    # 3) OpenCV 투영 입력 형태(N, 1, 3) 변환
    pts_cam_reshaped = pts_cam.reshape(-1, 1, 3)

    # 이미 카메라 좌표계 기준이므로 rvec/tvec는 0 벡터 사용
    zero_vec = np.zeros((3, 1), dtype=float)

    image_points, _ = cv2.projectPoints(
        pts_cam_reshaped,
        zero_vec,
        zero_vec,
        K,
        D
    )

    image_points = image_points.reshape(-1, 2)

    # 4) 이미지 경계 내부 포인트 필터링
    u = image_points[:, 0]
    v = image_points[:, 1]

    mask = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)

    uv_valid = image_points[mask].astype(int)
    valid_indices = indices[mask]

    return uv_valid, valid_indices


# 두 bbox([x1, y1, x2, y2]) IoU 계산
def iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    union = area1 + area2 - intersection
    if union <= 0:
        return 0.0
    return intersection / union