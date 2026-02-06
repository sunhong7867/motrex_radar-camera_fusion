#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def project_points(K, R, t, points_3d):
    """
    3D Points (N,3) -> 2D Pixels (N,2) projection
    """
    if points_3d.shape[0] == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=bool)

    P = K @ np.hstack([R, t.reshape(3, 1)])
    pts_h = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    proj = (P @ pts_h.T).T

    z = proj[:, 2]
    valid = z > 1e-4
    u = proj[:, 0] / np.maximum(z, 1e-9)
    v = proj[:, 1] / np.maximum(z, 1e-9)

    uv = np.vstack([u, v]).T
    return uv, valid


def cluster_1d_by_gap(values: np.ndarray, max_gap: float):
    if values is None or len(values) == 0:
        return []

    values = np.sort(values)
    clusters = []
    start = values[0]
    last = values[0]
    cluster = [values[0]]

    for v in values[1:]:
        if (v - last) <= max_gap:
            cluster.append(v)
        else:
            clusters.append(cluster)
            cluster = [v]
        last = v

    clusters.append(cluster)
    return clusters


def select_best_forward_cluster(forward_vals: np.ndarray, dopplers: np.ndarray, max_gap_m: float = 2.0):
    clusters = cluster_1d_by_gap(forward_vals, max_gap=max_gap_m)
    if not clusters:
        return None

    means = [np.mean(c) for c in clusters]
    best_idx = int(np.argmax(means))
    best_cluster = clusters[best_idx]

    mask = np.isin(forward_vals, best_cluster)
    best_dop = dopplers[mask]

    if best_dop.size == 0:
        return None
    return float(np.median(best_dop))
