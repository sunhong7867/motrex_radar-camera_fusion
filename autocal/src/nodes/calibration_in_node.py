#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modified camera intrinsic calibration node with improved logging.

This node performs chessboard-based camera intrinsic calibration either
offline (using images from a directory) or online (subscribing to a
camera topic). Compared to the original implementation, this version
emits clearer start and progress messages so that users can see when
calibration is underway and when it completes.

Key improvements:
  * Logs a `[Intrinsic] START` message when the node begins, indicating
    offline or online mode and the relevant parameters.
  * In offline calibration, logs a message just before running
    `cv2.calibrateCamera` to indicate that computation is in progress.
  * Retains all original functionality, including AutoImagePicker
    behavior and intrinsic.json writing.

Usage parameters mirror the original `calibration_in_node.py`.
"""

import os
import re
import glob
import math
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rospy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rospkg import RosPack

_FIND_RE = re.compile(r"\$\(\s*find\s+([A-Za-z0-9_]+)\s*\)")


def resolve_ros_path(path: str) -> str:
    """Resolve ROS package-based paths like $(find pkg)/..."""
    if not isinstance(path, str):
        return path
    if "$(" not in path:
        return path
    rp = RosPack()

    def _sub(match):
        pkg = match.group(1)
        try:
            return rp.get_path(pkg)
        except Exception:
            return match.group(0)

    return _FIND_RE.sub(_sub, path)


@dataclass
class BoardSquare:
    p1: np.ndarray
    p2: np.ndarray
    p3: np.ndarray
    p4: np.ndarray
    area: float
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    midpoint: np.ndarray
    angle_left_top: float
    angle_right_top: float
    angle_left_bottom: float
    angle_right_bottom: float


def _poly_area(pts: np.ndarray) -> float:
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba < 1e-9 or nbc < 1e-9:
        return 0.0
    cosv = float(np.dot(ba, bc) / (nba * nbc))
    cosv = max(-1.0, min(1.0, cosv))
    return math.degrees(math.acos(cosv))


def make_board_square(p1, p2, p3, p4) -> BoardSquare:
    pts = np.array([p1, p2, p4, p3], dtype=np.float64)  # polygon order
    area = _poly_area(pts)
    min_x = float(np.min(pts[:, 0]))
    max_x = float(np.max(pts[:, 0]))
    min_y = float(np.min(pts[:, 1]))
    max_y = float(np.max(pts[:, 1]))
    midpoint = np.mean(np.array([p1, p2, p3, p4], dtype=np.float64), axis=0)

    angle_lt = _angle_deg(np.array(p2), np.array(p1), np.array(p3))
    angle_rt = _angle_deg(np.array(p1), np.array(p2), np.array(p4))
    angle_lb = _angle_deg(np.array(p1), np.array(p3), np.array(p4))
    angle_rb = _angle_deg(np.array(p2), np.array(p4), np.array(p3))

    return BoardSquare(
        p1=np.array(p1, dtype=np.float64),
        p2=np.array(p2, dtype=np.float64),
        p3=np.array(p3, dtype=np.float64),
        p4=np.array(p4, dtype=np.float64),
        area=area,
        min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y,
        midpoint=np.array(midpoint, dtype=np.float64),
        angle_left_top=angle_lt,
        angle_right_top=angle_rt,
        angle_left_bottom=angle_lb,
        angle_right_bottom=angle_rb,
    )


class AutoImagePickerPy:
    """Ported from C++ AutoImagePicker.cpp with identical logic."""
    CHESSBOARD_MIN_AREA_PORTION = 0.04
    IMAGE_MARGIN_PERCENT = 0.1
    CHESSBOARD_MIN_AREA_CHANGE_PARTION_THRESH = 0.005
    CHESSBOARD_MIN_MOVE_THRESH = 0.08
    CHESSBOARD_MIN_ANGLE_CHANGE_THRESH = 5.0
    CHESSBOARD_MIN_ANGLE = 40.0

    def __init__(self, img_width: int, img_height: int, board_width: int, board_height: int, max_selected: int = 45):
        self.img_width = int(img_width)
        self.img_height = int(img_height)
        self.board_width = int(board_width)
        self.board_height = int(board_height)
        self.max_selected = int(max_selected)

        self.area_box = np.full((self.img_height, self.img_width), 5, dtype=np.int16)
        d_imgh = int(self.IMAGE_MARGIN_PERCENT * self.img_height)
        d_imgw = int(self.IMAGE_MARGIN_PERCENT * self.img_width)
        self.area_box[d_imgh:self.img_height - d_imgh, d_imgw:self.img_width - d_imgw] = 1

        self.candidate_board: List[BoardSquare] = []

    def status(self) -> bool:
        return len(self.candidate_board) >= self.max_selected

    def _check_validity(self, board: BoardSquare) -> bool:
        if board.area < self.CHESSBOARD_MIN_AREA_PORTION * self.img_width * self.img_height:
            return False
        if (
            board.angle_left_top < self.CHESSBOARD_MIN_ANGLE or
            board.angle_right_top < self.CHESSBOARD_MIN_ANGLE or
            board.angle_left_bottom < self.CHESSBOARD_MIN_ANGLE or
            board.angle_right_bottom < self.CHESSBOARD_MIN_ANGLE
        ):
            return False
        return True

    def _check_move_thresh(self, board: BoardSquare) -> bool:
        thresh = self.CHESSBOARD_MIN_MOVE_THRESH * float(self.img_width + self.img_height) / 2.0
        for cand in self.candidate_board:
            dx = board.midpoint[0] - cand.midpoint[0]
            dy = board.midpoint[1] - cand.midpoint[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < thresh:
                return False
        return True

    def _check_area_thresh(self, board: BoardSquare) -> bool:
        x0 = max(0, int(board.min_x))
        x1 = min(self.img_width, int(board.max_x))
        y0 = max(0, int(board.min_y))
        y1 = min(self.img_height, int(board.max_y))
        if x1 <= x0 or y1 <= y0:
            return False
        score = int(np.sum(self.area_box[y0:y1, x0:x1]))
        if score < self.CHESSBOARD_MIN_AREA_CHANGE_PARTION_THRESH * self.img_width * self.img_height:
            return False
        return True

    def _check_pose_angle_thresh(self, board: BoardSquare) -> bool:
        move_thresh = self.CHESSBOARD_MIN_MOVE_THRESH * float(self.img_width + self.img_height) / 2.0
        for cand in self.candidate_board:
            dx = board.midpoint[0] - cand.midpoint[0]
            dy = board.midpoint[1] - cand.midpoint[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < move_thresh:
                if (
                    abs(board.angle_left_top - cand.angle_left_top) < self.CHESSBOARD_MIN_ANGLE_CHANGE_THRESH and
                    abs(board.angle_right_top - cand.angle_right_top) < self.CHESSBOARD_MIN_ANGLE_CHANGE_THRESH and
                    abs(board.angle_left_bottom - cand.angle_left_bottom) < self.CHESSBOARD_MIN_ANGLE_CHANGE_THRESH and
                    abs(board.angle_right_bottom - cand.angle_right_bottom) < self.CHESSBOARD_MIN_ANGLE_CHANGE_THRESH
                ):
                    return False
        return True

    def _fill_area_box(self, board: BoardSquare) -> None:
        x0 = max(0, int(board.min_x))
        x1 = min(self.img_width, int(board.max_x))
        y0 = max(0, int(board.min_y))
        y1 = min(self.img_height, int(board.max_y))
        if x1 <= x0 or y1 <= y0:
            return
        self.area_box[y0:y1, x0:x1] = 0

    def add_image(self, image_corners: np.ndarray) -> bool:
        p1 = image_corners[0]
        p2 = image_corners[0 + self.board_width - 1]
        p3 = image_corners[len(image_corners) - self.board_width]
        p4 = image_corners[len(image_corners) - 1]
        sq = make_board_square(p1, p2, p3, p4)

        if not self._check_validity(sq):
            return False

        if self._check_move_thresh(sq) or self._check_area_thresh(sq):
            self.candidate_board.append(sq)
            self._fill_area_box(sq)
            return True

        if not self._check_pose_angle_thresh(sq):
            return False

        self.candidate_board.append(sq)
        self._fill_area_box(sq)
        return True


class CameraIntrinsicCalibrator:
    def __init__(self):
        rospy.init_node('calibration_in_node', anonymous=False)

        self.camera_topic = rospy.get_param('~camera_topic', '/camera/image_raw')
        self.board_w = int(rospy.get_param('~board/width', 10))
        self.board_h = int(rospy.get_param('~board/height', 7))
        self.square_size = float(rospy.get_param('~board/square_size', 0.025))

        self.grid_size_mm = int(rospy.get_param('~grid_size_mm', 50))
        self.min_samples = int(rospy.get_param('~min_samples', 20))
        self.max_selected = int(rospy.get_param('~max_selected', 45))

        self.image_dir = rospy.get_param('~image_dir', '')

        default_json = os.path.join(RosPack().get_path('autocal'), 'config', 'intrinsic.json')
        self.output_path = resolve_ros_path(rospy.get_param('~output_path', default_json))

        self.save_selected = bool(rospy.get_param('~save_selected', True))
        self.save_undistort = bool(rospy.get_param('~save_undistort', True))

        self.pattern_size = (self.board_w, self.board_h)

        # object points (C++ ordering 느낌 유지)
        objp = []
        for i in range(self.board_h):
            for j in range(self.board_w):
                objp.append([float(i) * self.square_size, float(j) * self.square_size, 0.0])
        self.objp = np.array(objp, dtype=np.float32)

        self.bridge = CvBridge()

        self.img_size: Optional[Tuple[int, int]] = None  # (w,h)
        self.obj_points: List[np.ndarray] = []
        self.img_points: List[np.ndarray] = []

        self._online_images: List[np.ndarray] = []
        self._online_names: List[str] = []

        # START 로그: intrinsic calibration 노드 시작을 알린다.
        if self.image_dir:
            rospy.loginfo(f"[Intrinsic] START offline calibration: image_dir={self.image_dir}")
            rospy.loginfo(f"[Intrinsic] Offline mode: image_dir={self.image_dir}")
            self.run_offline_calibration()
            rospy.signal_shutdown("offline intrinsic calibration complete")
        else:
            rospy.loginfo(f"[Intrinsic] START online calibration: topic={self.camera_topic}, pattern={self.board_w}x{self.board_h}")
            rospy.loginfo(f"[Intrinsic] Online mode: topic={self.camera_topic}, pattern={self.board_w}x{self.board_h}")
            rospy.Subscriber(self.camera_topic, Image, self.image_callback, queue_size=1)

    def _get_output_dirs(self, base_dir: str) -> Tuple[str, str]:
        return os.path.join(base_dir, "selected"), os.path.join(base_dir, "undistorted")

    def _corner_subpix(self, gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, int(self.grid_size_mm), 0.001)
        return cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

    def _safe_float_list(self, arr) -> List[float]:
        a = np.array(arr).reshape(-1)
        return [float(x) for x in a.tolist()]

    def _write_intrinsic_json(self, K: np.ndarray, D: np.ndarray, img_w: int, img_h: int) -> None:
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        K3 = [[float(K[r, c]) for c in range(3)] for r in range(3)]
        P3x4 = [
            [fx, 0.0, cx, 0.0],
            [0.0, fy, cy, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
        Dlist = self._safe_float_list(D.reshape(-1))

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "image_size": {"width": int(img_w), "height": int(img_h)},
                "distortion_model": "plumb_bob",
                "K": K3,
                "D": Dlist,
                "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "P": P3x4
            }, f, indent=2)

        rospy.loginfo(f"[Intrinsic] JSON saved: {self.output_path}")

    def _undistort_and_save(self, img_paths: List[str], out_dir: str, K: np.ndarray, D: np.ndarray, img_size: Tuple[int, int]):
        os.makedirs(out_dir, exist_ok=True)
        w, h = img_size
        map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), cv2.CV_32FC1)
        for p in img_paths:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            und = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(out_dir, os.path.basename(p)), und)

    def _select_images_with_picker(self, gray_images: List[np.ndarray], names: List[str]) -> Tuple[List[int], List[np.ndarray], List[str]]:
        h, w = gray_images[0].shape[:2]
        self.img_size = (w, h)
        picker = AutoImagePickerPy(w, h, self.board_w, self.board_h, max_selected=self.max_selected)

        selected_idx: List[int] = []
        selected_corners_raw: List[np.ndarray] = []
        selected_names: List[str] = []

        for i, (g, nm) in enumerate(zip(gray_images, names)):
            found, corners = cv2.findChessboardCorners(g, self.pattern_size)
            if not found or corners is None:
                continue
            corners_np = corners.reshape(-1, 2).astype(np.float64)
            if picker.add_image(corners_np):
                selected_idx.append(i)
                selected_corners_raw.append(corners)  # (N,1,2)
                selected_names.append(nm)
                rospy.loginfo(f"[Intrinsic] Select image: {nm} ({len(selected_idx)}/{self.max_selected})")
                if picker.status():
                    rospy.loginfo("[Intrinsic] Enough selected images (max_selected reached).")
                    break
        return selected_idx, selected_corners_raw, selected_names

    def run_offline_calibration(self):
        img_dir = self.image_dir
        patterns = sorted(glob.glob(os.path.join(img_dir, "calib_*.jpg")))
        if not patterns:
            rospy.logerr(f"[Intrinsic] No calib_*.jpg found in: {img_dir}")
            return
        gray_imgs: List[np.ndarray] = []
        color_paths: List[str] = []
        names: List[str] = []
        for p in patterns:
            g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            gray_imgs.append(g)
            color_paths.append(p)
            names.append(os.path.basename(p))
        if not gray_imgs:
            rospy.logerr("[Intrinsic] Failed to read any calib_*.jpg images.")
            return
        selected_idx, selected_corners_raw, selected_names = self._select_images_with_picker(gray_imgs, names)
        if len(selected_idx) == 0:
            rospy.logerr("[Intrinsic] No images selected by AutoImagePicker (check board size / images).")
            return
        if len(selected_idx) < self.min_samples:
            rospy.logwarn(
                f"[Intrinsic] Selected {len(selected_idx)} < min_samples({self.min_samples}). Still calibrating (may be unstable)."
            )
        selected_dir, undist_dir = self._get_output_dirs(img_dir)
        if self.save_selected:
            os.makedirs(selected_dir, exist_ok=True)
        if self.save_undistort:
            os.makedirs(undist_dir, exist_ok=True)
        self.obj_points = []
        self.img_points = []
        used_color_paths: List[str] = []
        for idx, corners in zip(selected_idx, selected_corners_raw):
            g = gray_imgs[idx]
            corners2 = self._corner_subpix(g, corners)
            self.obj_points.append(self.objp.copy())
            self.img_points.append(corners2.reshape(-1, 2).astype(np.float32))
            used_color_paths.append(color_paths[idx])
            if self.save_selected:
                src = color_paths[idx]
                dst = os.path.join(selected_dir, os.path.basename(src))
                if not os.path.exists(dst):
                    img_color = cv2.imread(src, cv2.IMREAD_COLOR)
                    if img_color is not None:
                        cv2.imwrite(dst, img_color)
        img_w, img_h = self.img_size
        # 캘리브레이션 시작 안내. cv2.calibrateCamera는 시간이 걸리므로 안내 메세지를 출력한다.
        rospy.loginfo(
            f"[Intrinsic] Computing intrinsic parameters from {len(self.obj_points)} images... this may take a while."
        )
        ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
            [op.astype(np.float32) for op in self.obj_points],
            [ip.astype(np.float32) for ip in self.img_points],
            (img_w, img_h),
            None,
            None
        )
        if not ret:
            rospy.logerr("[Intrinsic] cv2.calibrateCamera failed.")
            return
        rospy.loginfo(
            f"[Intrinsic] Calibration done. Selected={len(used_color_paths)} | image_size=({img_w},{img_h})"
        )
        self._write_intrinsic_json(K, D, img_w, img_h)
        if self.save_undistort:
            self._undistort_and_save(used_color_paths, undist_dir, K, D, (img_w, img_h))
        rospy.loginfo("[Intrinsic] Offline calibration complete.")

    def image_callback(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr(f"cv_bridge error: {e}")
            return
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        if self.img_size is None:
            self.img_size = (w, h)
            self._picker = AutoImagePickerPy(w, h, self.board_w, self.board_h, max_selected=self.max_selected)
            rospy.loginfo(f"[Intrinsic] Online picker initialized: img={w}x{h}, pattern={self.board_w}x{self.board_h}")
        found, corners = cv2.findChessboardCorners(gray, self.pattern_size)
        if not found or corners is None:
            return
        corners_np = corners.reshape(-1, 2).astype(np.float64)
        if not self._picker.add_image(corners_np):
            return
        corners2 = self._corner_subpix(gray, corners)
        self.obj_points.append(self.objp.copy())
        self.img_points.append(corners2.reshape(-1, 2).astype(np.float32))
        name = f"calib_{len(self.obj_points):02d}.jpg"
        self._online_images.append(bgr.copy())
        self._online_names.append(name)
        rospy.loginfo(f"[Intrinsic] Online selected {len(self.obj_points)} samples")
        if len(self.obj_points) >= self.min_samples or self._picker.status():
            self._compute_and_save_online()

    def _compute_and_save_online(self):
        img_w, img_h = self.img_size
        ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
            [op.astype(np.float32) for op in self.obj_points],
            [ip.astype(np.float32) for ip in self.img_points],
            (img_w, img_h),
            None,
            None
        )
        if not ret:
            rospy.logerr("[Intrinsic] cv2.calibrateCamera failed (online).")
            return
        self._write_intrinsic_json(K, D, img_w, img_h)
        base_dir = os.path.join(RosPack().get_path('autocal'), 'calibration_images')
        selected_dir, undist_dir = self._get_output_dirs(base_dir)
        if self.save_selected:
            os.makedirs(selected_dir, exist_ok=True)
            for img, nm in zip(self._online_images, self._online_names):
                cv2.imwrite(os.path.join(selected_dir, nm), img)
        if self.save_undistort:
            paths = [os.path.join(selected_dir, nm) for nm in self._online_names] if self.save_selected else []
            if paths:
                self._undistort_and_save(paths, undist_dir, K, D, (img_w, img_h))
        rospy.loginfo("[Intrinsic] Online calibration complete. Shutting down.")
        rospy.signal_shutdown("intrinsic calibration complete")


def main():
    CameraIntrinsicCalibrator()
    rospy.spin()


if __name__ == '__main__':
    main()