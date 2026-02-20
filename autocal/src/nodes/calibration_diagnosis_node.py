#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
역할:
- 트리거 수신 후 일정 시간 동안 트랙 bbox 기준점과 레이더 투영점의 대응을 수집해 외부파라미터 품질을 진단한다.
- 트랙별 RMSE와 진행 방향 차이를 계산하여 캘리브레이션 결과의 정합도를 텍스트 리포트로 발행한다.

입력:
- `~topics/detections` (`autocal/DetectionArray`): 트랙 ID와 bbox 정보.
- `~topics/radar` (`sensor_msgs/PointCloud2`): x, y, z, doppler를 포함한 레이더 포인트.
- `~topics/camera_info` (`sensor_msgs/CameraInfo`): 카메라 내부 파라미터 K.
- `/autocal/diagnosis/start` (`std_msgs/String`): 진단 시작 명령(top/center/bottom 기준점 모드).

출력:
- `/autocal/diagnosis/result` (`std_msgs/String`): 트랙 수, RMSE 통계, 상위 오차 트랙 요약.
"""

import json
import os
from collections import defaultdict, deque
from typing import Dict, Optional, Tuple

import message_filters
import numpy as np
import rospy
import rospkg
import sensor_msgs.point_cloud2 as pc2
from autocal.msg import DetectionArray
from sensor_msgs.msg import CameraInfo, PointCloud2
from std_msgs.msg import String

# ---------------------------------------------------------
# 진단 파라미터
# ---------------------------------------------------------
COLLECTION_DURATION_SEC = 30.0      # 진단 데이터 수집 시간(초)
MIN_TRACK_LEN = 15                  # 유효 트랙 최소 길이(프레임)
MIN_TRACKS = 3                      # 리포트 생성 최소 트랙 수
CLUSTER_TOL_M = 3.0                 # 레이더 XY 클러스터 반경(m)
MIN_SPEED_MPS = 2.0                 # 정적 물체 제거 속도 임계값(m/s)
MATCH_MAX_DIST_PX = 300.0           # bbox 기준점-투영점 매칭 허용 거리(px)


class TrajectoryEvaluationNode:
    def __init__(self):
        rospy.init_node("trajectory_evaluation_node", anonymous=False)

        self.detections_topic = rospy.get_param("~topics/detections", "/autocal/tracks")
        self.radar_topic = rospy.get_param("~topics/radar", "/point_cloud")
        self.camera_info_topic = rospy.get_param("~topics/camera_info", "/camera/camera_info")

        self.trigger_topic = "/autocal/diagnosis/start"
        self.result_topic = "/autocal/diagnosis/result"

        rp = rospkg.RosPack()
        default_ext_path = os.path.join(rp.get_path("autocal"), "config", "extrinsic.json")
        self.extrinsic_path = rospy.get_param("~extrinsic_path", default_ext_path)

        self.collection_duration = float(rospy.get_param("~collection_duration", COLLECTION_DURATION_SEC))
        self.min_track_len = int(rospy.get_param("~min_track_len", MIN_TRACK_LEN))
        self.min_tracks = int(rospy.get_param("~min_tracks", MIN_TRACKS))
        self.cluster_tol = float(rospy.get_param("~cluster_tol", CLUSTER_TOL_M))
        self.min_speed_mps = float(rospy.get_param("~min_speed_mps", MIN_SPEED_MPS))
        self.match_max_dist_px = float(rospy.get_param("~match_max_dist_px", MATCH_MAX_DIST_PX))

        self.collecting = False
        self.start_time = rospy.Time(0)
        self.ref_mode = "bottom"
        self.cam_K = None
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))

        self.track_pairs: Dict[int, deque] = defaultdict(lambda: deque(maxlen=120))
        self.pub_result = rospy.Publisher(self.result_topic, String, queue_size=1)

        self.load_extrinsic()
        rospy.Subscriber(self.trigger_topic, String, self.on_trigger)
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self.cam_info_cb)

        self.sub_det = message_filters.Subscriber(self.detections_topic, DetectionArray)
        self.sub_radar = message_filters.Subscriber(self.radar_topic, PointCloud2)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.sub_det, self.sub_radar], 30, 0.2)
        self.ts.registerCallback(self.data_callback)

        rospy.loginfo("[TrajectoryEval] Diagnosis node ready")

    def on_trigger(self, msg: String):
        """
        트리거 수신 시 수집 시작 및 기준점 모드 설정
        """
        mode = (msg.data or "").strip().lower()
        self.ref_mode = mode if mode in {"top", "center", "bottom"} else "bottom"

        rospy.loginfo(f"[TrajectoryEval] Start collecting: {self.collection_duration}s, ref={self.ref_mode}")
        self.track_pairs.clear()
        self.start_time = rospy.Time.now()
        self.collecting = True
        self.load_extrinsic()

    def load_extrinsic(self):
        """
        진단에 사용할 extrinsic(R, t)을 파일에서 로드
        """
        if not os.path.exists(self.extrinsic_path):
            return

        try:
            with open(self.extrinsic_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.R = np.array(data["R"], dtype=np.float64)
            self.t = np.array(data["t"], dtype=np.float64).reshape(3, 1)
        except Exception as exc:
            rospy.logwarn(f"[TrajectoryEval] Failed to load extrinsic: {exc}")

    def cam_info_cb(self, msg: CameraInfo):
        """
        CameraInfo에서 카메라 내부행렬 K 갱신
        """
        self.cam_K = np.array(msg.K, dtype=np.float64).reshape(3, 3)

    def _get_bbox_ref_point(self, bbox) -> Tuple[float, float]:
        """
        설정된 기준 모드(top/center/bottom)에 맞는 bbox 기준점 반환
        """
        x1, y1, x2, y2 = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
        if self.ref_mode == "top":
            return (x1 + x2) / 2.0, float(y1)
        if self.ref_mode == "center":
            return (x1 + x2) / 2.0, (y1 + y2) / 2.0
        return (x1 + x2) / 2.0, float(y2)

    def _read_radar(self, radar_msg: PointCloud2) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        레이더 메시지에서 XYZ와 선택적 Doppler 필드 추출
        """
        try:
            radar_raw = np.array(list(pc2.read_points(radar_msg, field_names=("x", "y", "z", "doppler"), skip_nans=True)))
            if radar_raw.shape[0] == 0:
                return np.zeros((0, 3)), None
            return radar_raw[:, :3], radar_raw[:, 3]
        except Exception:
            radar_xyz = np.array(list(pc2.read_points(radar_msg, field_names=("x", "y", "z"), skip_nans=True)))
            if radar_xyz.shape[0] == 0:
                return np.zeros((0, 3)), None
            return radar_xyz[:, :3], None

    def _cluster_radar_points(self, points: np.ndarray, velocities: Optional[np.ndarray]):
        """
        XY 거리 기반 클러스터링 후 최소 속도 조건을 만족하는 중심점 반환
        """
        n = points.shape[0]
        if n == 0:
            return []

        visited = np.zeros(n, dtype=bool)
        clusters = []
        for i in range(n):
            if visited[i]:
                continue

            dist_sq = np.sum((points[:, :2] - points[i, :2]) ** 2, axis=1)
            neighbors = np.where((dist_sq <= self.cluster_tol ** 2) & (~visited))[0]
            if len(neighbors) < 2:
                continue

            visited[neighbors] = True
            vel_med = None
            if velocities is not None and velocities.size > 0:
                vel_med = float(np.median(velocities[neighbors]))
                if abs(vel_med) < self.min_speed_mps:
                    continue

            clusters.append((np.median(points[neighbors], axis=0), vel_med))

        return clusters

    def _project_points(self, pts_3d: np.ndarray) -> np.ndarray:
        """
        레이더 3D 점을 extrinsic과 K를 이용해 영상 좌표(u, v)로 투영
        """
        if self.cam_K is None or pts_3d.size == 0:
            return np.zeros((0, 2))

        pts_cam = (self.R @ pts_3d.T) + self.t
        valid = pts_cam[2, :] > 0.5
        pts_cam = pts_cam[:, valid]
        if pts_cam.shape[1] == 0:
            return np.zeros((0, 2))

        uv_h = self.cam_K @ pts_cam
        return (uv_h[:2, :] / uv_h[2, :]).T

    def data_callback(self, det_msg: DetectionArray, radar_msg: PointCloud2):
        """
        수집 중 bbox 기준점과 레이더 투영점을 최근접 매칭해 트랙별로 누적
        """
        if not self.collecting or self.cam_K is None:
            return

        radar_xyz, radar_doppler = self._read_radar(radar_msg)
        if radar_xyz.shape[0] == 0:
            return

        radar_clusters = self._cluster_radar_points(radar_xyz, radar_doppler)
        if not radar_clusters:
            return

        centers = np.array([c[0] for c in radar_clusters], dtype=np.float64)
        proj_pts = self._project_points(centers)
        if proj_pts.shape[0] == 0:
            return

        for det in det_msg.detections:
            if det.id <= 0:
                continue

            u_ref, v_ref = self._get_bbox_ref_point(det.bbox)
            dists = np.hypot(proj_pts[:, 0] - u_ref, proj_pts[:, 1] - v_ref)
            idx = int(np.argmin(dists))
            if dists[idx] < self.match_max_dist_px:
                self.track_pairs[det.id].append(((u_ref, v_ref), (proj_pts[idx, 0], proj_pts[idx, 1])))

    def _fit_direction(self, points: np.ndarray) -> Optional[np.ndarray]:
        """
        점 집합의 주성분 방향벡터 계산
        """
        if points.shape[0] < 2:
            return None

        mean = points.mean(axis=0)
        centered = points - mean
        cov = np.cov(centered.T)
        if cov.shape != (2, 2):
            return None

        eigvals, eigvecs = np.linalg.eigh(cov)
        vec = eigvecs[:, np.argmax(eigvals)]
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            return None
        return vec / norm

    def analyze_and_report(self):
        """
        수집된 트랙 대응쌍으로 RMSE와 방향 오차를 계산해 리포트 발행
        """
        self.collecting = False
        summaries = []

        for tid, pairs in self.track_pairs.items():
            if len(pairs) < self.min_track_len:
                continue

            ref_pts = np.array([p[0] for p in pairs], dtype=np.float64)
            rad_pts = np.array([p[1] for p in pairs], dtype=np.float64)
            diff = ref_pts - rad_pts
            rmse = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))

            dir_ref = self._fit_direction(ref_pts)
            dir_rad = self._fit_direction(rad_pts)
            angle = None
            if dir_ref is not None and dir_rad is not None:
                dot = float(np.clip(np.abs(np.dot(dir_ref, dir_rad)), -1.0, 1.0))
                angle = float(np.degrees(np.arccos(dot)))

            summaries.append({"id": tid, "rmse": rmse, "angle": angle, "count": len(pairs)})

        if len(summaries) < self.min_tracks:
            msg = (
                f"⚠ [평가 실패] 유효 트랙 부족 ({len(summaries)}개)\n"
                f"- 최소 트랙 수: {self.min_tracks}\n"
                f"- 트랙 기준 길이: {self.min_track_len} 프레임"
            )
            self.pub_result.publish(msg)
            return

        rmses = np.array([s["rmse"] for s in summaries], dtype=np.float64)
        angles = np.array([s["angle"] for s in summaries if s["angle"] is not None], dtype=np.float64)

        rms_mean = float(np.mean(rmses))
        rms_median = float(np.median(rmses))
        ang_mean = float(np.mean(angles)) if angles.size > 0 else float("nan")
        worst = sorted(summaries, key=lambda x: x["rmse"], reverse=True)[:3]

        lines = [
            "📊 궤적 기반 평가 결과",
            f"- 기준점 모드: {self.ref_mode}",
            f"- 유효 트랙 수: {len(summaries)}",
            f"- 평균 RMSE(px): {rms_mean:.2f}",
            f"- 중앙값 RMSE(px): {rms_median:.2f}",
        ]
        if angles.size > 0:
            lines.append(f"- 평균 방향 각도 차(°): {ang_mean:.2f}")

        lines.append("")
        lines.append("🔎 상위 오차 트랙")
        for s in worst:
            ang_str = f"{s['angle']:.2f}°" if s["angle"] is not None else "N/A"
            lines.append(f"- ID {s['id']}: RMSE {s['rmse']:.2f}px, Δθ {ang_str}, N={s['count']}")

        self.pub_result.publish("\n".join(lines))

    def run(self):
        """
        수집 시간을 주기 감시하고 만료 시 자동 리포트 생성
        """
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.collecting:
                elapsed = (rospy.Time.now() - self.start_time).to_sec()
                if elapsed > self.collection_duration:
                    self.analyze_and_report()
            rate.sleep()


if __name__ == "__main__":
    node = TrajectoryEvaluationNode()
    node.run()
