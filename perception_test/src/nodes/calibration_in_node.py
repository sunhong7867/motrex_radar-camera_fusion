#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
calibration_in_node.py

카메라 내부 파라미터(초점거리, 주점, 왜곡)를 캘리브레이션하는 노드입니다.
체커보드 이미지를 여러 장 수집한 뒤 OpenCV의 cv2.calibrateCamera를 통해
내부 파라미터(K)와 왜곡 계수(D)를 계산합니다. 결과는 YAML 형식으로
config/camera_intrinsic.yaml에 저장됩니다.

ROS 파라미터:
- ~camera_topic (기본: /camera/image_raw) : 이미지 토픽
- ~board/width, ~board/height (기본: 7, 5) : 체커보드의 내부 코너 수
- ~board/square_size (기본: 0.1) : 한 칸의 실제 길이(m)
- ~min_samples (기본: 20) : 필요 샘플 수
- ~output_path (기본: $(find perception_test)/config/camera_intrinsic.yaml)
- ~save_images/enable (bool) : 검출된 이미지를 저장할지 여부
- ~save_images/path (str) : 저장 경로 (기본: calibration_images)
"""

import os
import re
import cv2
import numpy as np
import rospy
import yaml
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rospkg import RosPack

_FIND_RE = re.compile(r"\$\(\s*find\s+([A-Za-z0-9_]+)\s*\)")


def resolve_ros_path(path: str) -> str:
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

class CameraIntrinsicCalibrator:
    def __init__(self):
        """Initialize the intrinsic calibrator.

        This class supports two modes of operation:

        * Online mode (default): Subscribe to a camera image topic and collect
          chessboard images in real-time. When a minimum number of samples
          (``min_samples``) have been collected, it computes the intrinsic
          parameters.

        * Offline mode: If the parameter ``~image_dir`` is provided and points
          to a directory on disk, all images in this directory will be used for
          calibration. In this case the node does not subscribe to any ROS
          topics and will exit after saving the results. This replicates the
          behavior of PJLab's SensorsCalibration intrinsic calibrator, which
          takes a directory of calibration images as input【478033828322678†L320-L355】.
        """

        rospy.init_node('calibration_in_node', anonymous=False)

        # ROS parameters
        self.camera_topic = rospy.get_param('~camera_topic', '/camera/image_raw')
        self.board_w = int(rospy.get_param('~board/width', 7))
        self.board_h = int(rospy.get_param('~board/height', 5))
        self.square_size = float(rospy.get_param('~board/square_size', 0.1))
        self.min_samples = int(rospy.get_param('~min_samples', 20))

        # Offline calibration directory. If provided, calibration will run
        # immediately using images in this directory rather than subscribing
        # to a camera topic.
        self.image_dir = rospy.get_param('~image_dir', '')

        # Result paths
        default_out = os.path.join(RosPack().get_path('perception_test'), 'config', 'camera_intrinsic.yaml')
        self.output_path = resolve_ros_path(rospy.get_param('~output_path', default_out))

        # Image saving options (for online mode)
        self.save_images = bool(rospy.get_param('~save_images/enable', False))
        default_img_path = os.path.join(RosPack().get_path('perception_test'), 'calibration_images')
        self.save_images_path = resolve_ros_path(rospy.get_param('~save_images/path', default_img_path))

        # Undistorted image saving options (for offline mode)
        self.save_undistort = bool(rospy.get_param('~undistort/enable', False))
        default_undistort_path = os.path.join(RosPack().get_path('perception_test'), 'calibration_images', 'undistort')
        self.undistort_path = resolve_ros_path(rospy.get_param('~undistort/path', default_undistort_path))

        # Prepare chessboard object points
        self.pattern_size = (self.board_w, self.board_h)
        objp = np.zeros((self.board_h * self.board_w, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.board_w, 0:self.board_h].T.reshape(-1, 2)
        objp *= self.square_size
        self.objp = objp

        # Storage for object and image points
        self.obj_points = []
        self.img_points = []
        self.img_size = None
        self.sample_count = 0

        self.bridge = CvBridge()

        # Create directories if needed
        if self.save_images:
            os.makedirs(self.save_images_path, exist_ok=True)
        if self.save_undistort:
            os.makedirs(self.undistort_path, exist_ok=True)

        # Decide mode based on image_dir
        if self.image_dir:
            # Offline mode: run calibration immediately using images in directory
            rospy.loginfo(f"[Intrinsic] Offline calibration using images in {self.image_dir}")
            self.run_offline_calibration()
            rospy.signal_shutdown("offline intrinsic calibration complete")
        else:
            # Online mode: subscribe to camera topic
            rospy.Subscriber(self.camera_topic, Image, self.image_callback, queue_size=1)
            rospy.loginfo(
                f"[Intrinsic] Online mode: topic={self.camera_topic}, pattern={self.board_w}x{self.board_h}, "
                f"min_samples={self.min_samples}"
            )

    def image_callback(self, msg: Image):
        if self.sample_count >= self.min_samples:
            return
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except CvBridgeError as e:
            rospy.logerr(f"cv_bridge error: {e}")
            return

        if self.img_size is None:
            self.img_size = (cv_img.shape[1], cv_img.shape[0])

        found, corners = cv2.findChessboardCorners(cv_img, self.pattern_size,
                                                   flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if found:
            # 코너 정밀화
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(cv_img, corners, (11, 11), (-1, -1), criteria)
            self.obj_points.append(self.objp.copy())
            self.img_points.append(corners2.reshape(-1, 2))
            self.sample_count += 1
            # 이미지 저장
            if self.save_images:
                vis = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(vis, self.pattern_size, corners2, True)
                filename = f"calib_{self.sample_count:02d}.png"
                cv2.imwrite(os.path.join(self.save_images_path, filename), vis)
            rospy.loginfo(f"[Intrinsic] 샘플 {self.sample_count}/{self.min_samples} 확보")
            if self.sample_count >= self.min_samples:
                self.compute_and_save()

    def compute_and_save(self):
        if not self.obj_points or not self.img_points:
            rospy.logerr("샘플이 부족해 캘리브레이션 불가")
            return
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.obj_points, self.img_points,
                                                            self.img_size, None, None)
        if not ret:
            rospy.logerr("cv2.calibrateCamera 실패")
            return
        data = {
            'camera_spec': {
                'name': 'camera',
                'resolution': {
                    'width': self.img_size[0],
                    'height': self.img_size[1]
                }
            },
            'camera_info': {
                'distortion_model': 'plumb_bob',
                'D': dist.flatten().tolist(),
                'K': mtx.flatten().tolist(),
                'R': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                'P': [mtx[0, 0], 0.0, mtx[0, 2], 0.0,
                      0.0, mtx[1, 1], mtx[1, 2], 0.0,
                      0.0, 0.0, 1.0, 0.0]
            }
        }
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        rospy.loginfo(f"[Intrinsic] 캘리브레이션 결과 저장: {self.output_path}")
        rospy.signal_shutdown("카메라 내부 캘리브레이션 완료")

    def run_offline_calibration(self):
        """Run intrinsic calibration using images stored in a directory.

        This function walks through all files in ``self.image_dir`` matching common
        image extensions (jpg, png, jpeg, bmp). For each image, it performs
        chessboard corner detection. Successful detections contribute to the
        intrinsic calibration. After collecting detections, it computes the
        camera matrix and distortion coefficients and writes them to the
        configured output path. Optionally, it saves undistorted versions of
        each input image to ``self.undistort_path``.
        """
        import glob
        # Accept multiple image extensions
        patterns = []
        for ext in ('*.jpg', '*.png', '*.jpeg', '*.bmp', '*.JPG', '*.PNG', '*.JPEG', '*.BMP'):
            patterns.extend(glob.glob(os.path.join(self.image_dir, ext)))
        if not patterns:
            rospy.logerr(f"[Intrinsic] No image files found in {self.image_dir}")
            return
        # Reset state
        self.obj_points = []
        self.img_points = []
        self.img_size = None
        self.sample_count = 0
        for img_path in sorted(patterns):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                rospy.logwarn(f"[Intrinsic] Failed to read image: {img_path}")
                continue
            if self.img_size is None:
                self.img_size = (img.shape[1], img.shape[0])
            found, corners = cv2.findChessboardCorners(
                img,
                self.pattern_size,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if found:
                # refine corners
                criteria = (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001,
                )
                corners2 = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
                self.obj_points.append(self.objp.copy())
                self.img_points.append(corners2.reshape(-1, 2))
                self.sample_count += 1
        if self.sample_count == 0:
            rospy.logerr(f"[Intrinsic] No chessboard corners detected in {self.image_dir}")
            return
        rospy.loginfo(f"[Intrinsic] Detected corners in {self.sample_count} images")
        # Compute calibration and save YAML
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.obj_points, self.img_points, self.img_size, None, None
        )
        if not ret:
            rospy.logerr("[Intrinsic] cv2.calibrateCamera failed during offline calibration")
            return
        data = {
            'camera_spec': {
                'name': 'camera',
                'resolution': {'width': self.img_size[0], 'height': self.img_size[1]},
            },
            'camera_info': {
                'distortion_model': 'plumb_bob',
                'D': dist.flatten().tolist(),
                'K': mtx.flatten().tolist(),
                'R': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                'P': [mtx[0, 0], 0.0, mtx[0, 2], 0.0, 0.0, mtx[1, 1], mtx[1, 2], 0.0, 0.0, 0.0, 1.0, 0.0],
            },
        }
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        rospy.loginfo(f"[Intrinsic] Offline calibration saved to {self.output_path}")
        # Optionally save undistorted images
        if self.save_undistort:
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, self.img_size, 1, self.img_size)
            for img_path in sorted(patterns):
                img = cv2.imread(img_path)
                if img is None:
                    continue
                dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
                # Crop if ROI provided
                x, y, w, h = roi
                dst = dst[y : y + h, x : x + w]
                base = os.path.basename(img_path)
                out_path = os.path.join(self.undistort_path, base)
                cv2.imwrite(out_path, dst)
            rospy.loginfo(f"[Intrinsic] Undistorted images saved to {self.undistort_path}")

def main():
    calibrator = CameraIntrinsicCalibrator()
    rospy.spin()

if __name__ == '__main__':
    main()