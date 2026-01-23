#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
calibration_ex_node.py

레이더-카메라 외부 파라미터(R, t)를 실시간으로 추정하는 노드입니다.
DetectionArray(2D bbox)와 PointCloud2(3D 점) 데이터를 시간 동기화하여
각 프레임에서 가장 신뢰도 높은 차량을 선택하고, 레이더 점들의
무게중심을 3D 좌표로 사용합니다. 일정 샘플을 수집하면 solvePnP로
R,t를 계산한 후 extrinsic.json으로 저장합니다.

ROS 파라미터:
- ~detections_topic (기본: /perception_test/detections)
- ~radar_topic      (기본: /point_cloud)
- ~camera_info_topic (기본: /camera/camera_info)
- ~min_samples      (기본: 20)
- ~detection_threshold (기본: 0.5)
- ~extrinsic_path    (기본: $(find perception_test)/config/extrinsic.json)
"""

import json
import os
from typing import List
import numpy as np
import rospy
import rospkg
import cv2
from sensor_msgs.msg import CameraInfo, PointCloud2
import sensor_msgs.point_cloud2 as pc2
import message_filters
from perception_test.msg import DetectionArray

class ExtrinsicCalibrationManager:
    def __init__(self):
        rospy.init_node('calibration_ex_node', anonymous=False)
        self.detections_topic = rospy.get_param('~detections_topic', '/perception_test/detections')
        self.radar_topic = rospy.get_param('~radar_topic', '/point_cloud')
        self.camera_info_topic = rospy.get_param('~camera_info_topic', '/camera/camera_info')
        self.min_samples = int(rospy.get_param('~min_samples', 20))
        self.detection_thresh = float(rospy.get_param('~detection_threshold', 0.5))

        # extrinsic 저장 경로: config/extrinsic.json
        default_pkg = 'perception_test'
        rp = rospkg.RosPack()
        default_path = os.path.join(rp.get_path(default_pkg), 'config', 'extrinsic.json')
        self.save_path = rospy.get_param('~extrinsic_path', default_path)

        self.samples = 0
        self.obj_pts: List[np.ndarray] = []
        self.img_pts: List[np.ndarray] = []
        self.K = None
        self.D = None

        # 동기화 설정
        sub_det = message_filters.Subscriber(self.detections_topic, DetectionArray)
        sub_rad = message_filters.Subscriber(self.radar_topic, PointCloud2)
        self.ts = message_filters.ApproximateTimeSynchronizer([sub_det, sub_rad], 10, 0.1)
        self.ts.registerCallback(self._callback)
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self._info_callback, queue_size=1)
        rospy.loginfo(f"[Extrinsic] 데이터 수집 중... det={self.detections_topic}, radar={self.radar_topic}")

    def _info_callback(self, msg: CameraInfo):
        if self.K is None:
            self.K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
            self.D = np.array(msg.D, dtype=np.float64) if msg.D else None
            rospy.loginfo("[Extrinsic] CameraInfo 수신")

    def _callback(self, det_msg: DetectionArray, rad_msg: PointCloud2):
        # CameraInfo 수신 전이면 리턴
        if self.K is None:
            return
        # detection 없으면 패스
        if not det_msg.detections:
            return
        best_det = max(det_msg.detections, key=lambda d: d.score)
        if best_det.score < self.detection_thresh:
            return
        # 2D bbox 중심 (u,v)
        bb = best_det.bbox
        u = (bb.xmin + bb.xmax) * 0.5
        v = (bb.ymin + bb.ymax) * 0.5
        # 3D 레이더 포인트 추출
        pts = []
        for p in pc2.read_points(rad_msg, field_names=('x','y','z','power'), skip_nans=True):
            pts.append([p[0], p[1], p[2], p[3]])
        if not pts:
            return
        pts = np.array(pts, dtype=np.float64)
        # power 기준 간단한 필터
        valid = pts[:, 3] > 0
        xyz = pts[valid, :3]
        if xyz.shape[0] < 3:
            return
        centroid = np.mean(xyz, axis=0)
        # 너무 가까운 샘플 제외
        dist = np.linalg.norm(centroid)
        if dist < 5.0:
            return
        self.obj_pts.append(centroid)
        self.img_pts.append([u, v])
        self.samples += 1
        rospy.loginfo(f"[Extrinsic] 샘플 {self.samples}/{self.min_samples} 저장 (거리 {dist:.1f}m)")
        if self.samples >= self.min_samples:
            self.run_calibration()

    def run_calibration(self):
        obj_pts = np.array(self.obj_pts, dtype=np.float64)
        img_pts = np.array(self.img_pts, dtype=np.float64)
        success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, self.K, self.D,
                                           flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            rospy.logwarn("[Extrinsic] solvePnP 실패, 샘플 더 모음")
            self.min_samples += 10
            return
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3, 1)
        data = {'R': R.tolist(), 't': t.flatten().tolist()}
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        rospy.loginfo(f"[Extrinsic] extrinsic.json 저장 완료: {self.save_path}")
        rospy.signal_shutdown("외부 캘리브레이션 완료")

def main():
    ExtrinsicCalibrationManager()
    rospy.spin()

if __name__ == '__main__':
    main()
