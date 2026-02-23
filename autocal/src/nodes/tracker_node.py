#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
역할:
- 객체 검출 결과(`DetectionArray`)를 SORT 기반 칼만 필터 트래커에 입력해 프레임 간 객체 ID를 안정적으로 부여한다.
- IoU 기반 헝가리안 매칭으로 검출-트랙 연관을 수행하고, 미매칭 검출은 신규 트랙으로 생성하며 오래 갱신되지 않은 트랙은 제거한다.
- 추적 결과를 `DetectionArray` 형식으로 다시 발행해 후단 노드가 일관된 track id를 사용하도록 지원한다.

입력:
- `~in_topic` (`autocal/DetectionArray`): 추적할 검출 bbox 목록.
- `~iou_threshold`, `~min_hits`, `~max_age` (`float/int`): SORT 동작 파라미터.

출력:
- `~out_topic` (`autocal/DetectionArray`): id가 부여된 추적 결과.
"""

import numpy as np
import rospy
from autocal.msg import BoundingBox, Detection, DetectionArray
from filterpy.kalman import KalmanFilter

# ---------------------------------------------------------
# 트래커 파라미터
# ---------------------------------------------------------
DEFAULT_IOU_THRESHOLD = 0.1      # 검출-트랙 매칭 IoU 임계값
DEFAULT_MIN_HITS = 1             # 출력 시작 최소 연속 매칭 횟수
DEFAULT_MAX_AGE = 15             # 미갱신 허용 프레임 수

DEFAULT_IN_TOPIC = "/autocal/detections"   # 입력 검출 토픽
DEFAULT_OUT_TOPIC = "/autocal/tracks"      # 출력 추적 토픽


def linear_assignment(cost_matrix):
    """
    헝가리안 알고리즘으로 최소 비용 매칭 인덱스를 계산
    """
    try:
        from scipy.optimize import linear_sum_assignment

        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))
    except ImportError:
        rospy.logerr("[tracker] scipy is required for linear assignment")
        return np.empty((0, 2))


def iou_batch(bb_test, bb_gt):
    """
    두 bbox 집합 간 IoU 행렬을 계산
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h

    o = wh / (
        (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
        - wh
    )
    return o


def convert_bbox_to_z(bbox):
    """
    [x1,y1,x2,y2] bbox를 [cx,cy,scale,ratio] 상태벡터로 변환
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    상태벡터 [cx,cy,scale,ratio]를 [x1,y1,x2,y2] bbox로 변환
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
    return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        """
        초기 bbox로 트랙 상태와 칼만 필터를 생성
        """
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        매칭된 검출 bbox로 필터 상태를 측정 업데이트
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        다음 프레임 bbox를 예측하고 내부 상태를 갱신
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1][0]

    def get_state(self):
        """
        현재 필터 상태를 bbox 형식으로 반환
        """
        return convert_x_to_bbox(self.kf.x)[0]


class Sort:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.1):
        """
        SORT 파라미터와 트랙 리스트를 초기화
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        현재 프레임 검출을 입력받아 추적 결과 배열을 반환
        """
        self.frame_count += 1

        # 기존 트랙 예측
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # 검출-트랙 매칭
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold
        )

        # 매칭 트랙 업데이트
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # 미매칭 검출로 신규 트랙 생성
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)

        # 출력 및 오래된 트랙 제거
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    IoU 기준으로 검출과 트래커를 매칭하고 미매칭 인덱스를 분리
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty((0, 2))

    unmatched_detections = [d for d in range(len(detections)) if d not in matched_indices[:, 0]]
    unmatched_trackers = [t for t in range(len(trackers)) if t not in matched_indices[:, 1]]

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    matches = np.empty((0, 2), dtype=int) if len(matches) == 0 else np.concatenate(matches, axis=0)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class TrackerNode:
    def __init__(self):
        """
        트래커 파라미터, 입출력 토픽, SORT 인스턴스를 초기화
        """
        rospy.init_node("tracker_node", anonymous=True)

        self.iou_thres = float(rospy.get_param("~iou_threshold", DEFAULT_IOU_THRESHOLD))
        self.min_hits = int(rospy.get_param("~min_hits", DEFAULT_MIN_HITS))
        self.max_age = int(rospy.get_param("~max_age", DEFAULT_MAX_AGE))

        self.tracker = Sort(max_age=self.max_age, min_hits=self.min_hits, iou_threshold=self.iou_thres)

        in_topic = rospy.get_param("~in_topic", DEFAULT_IN_TOPIC)
        out_topic = rospy.get_param("~out_topic", DEFAULT_OUT_TOPIC)
        self.sub = rospy.Subscriber(in_topic, DetectionArray, self.callback, queue_size=1)
        self.pub = rospy.Publisher(out_topic, DetectionArray, queue_size=1)

        rospy.loginfo(f"[tracker] Ready. iou={self.iou_thres} max_age={self.max_age} min_hits={self.min_hits}")

    def callback(self, msg):
        """
        DetectionArray를 SORT에 적용하고 track id가 포함된 결과를 발행
        """
        # DetectionArray를 numpy로 변환
        dets_list = []
        for d in msg.detections:
            dets_list.append([d.bbox.xmin, d.bbox.ymin, d.bbox.xmax, d.bbox.ymax, d.score])

        dets = np.array(dets_list)
        if len(dets) == 0:
            dets = np.empty((0, 5))

        # SORT 업데이트
        try:
            trackers = self.tracker.update(dets)
        except Exception as exc:
            rospy.logerr(f"[tracker] update error: {exc}")
            return

        # 추적 결과 메시지 구성
        out_msg = DetectionArray()
        out_msg.header = msg.header

        # 입력 detection lane_id를 트랙 결과로 전달하기 위한 준비
        det_boxes = np.array([[d.bbox.xmin, d.bbox.ymin, d.bbox.xmax, d.bbox.ymax] for d in msg.detections], dtype=np.float64) if len(msg.detections) > 0 else np.zeros((0, 4), dtype=np.float64)

        for trk in trackers:
            d = Detection()
            d.id = int(trk[4])
            d.score = 1.0

            bbox = BoundingBox()
            bbox.xmin = int(trk[0])
            bbox.ymin = int(trk[1])
            bbox.xmax = int(trk[2])
            bbox.ymax = int(trk[3])
            d.bbox = bbox

            # 트랙 bbox와 가장 가까운 입력 detection의 lane_id 계승
            if det_boxes.shape[0] > 0:
                trk_box = np.array([bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax], dtype=np.float64)
                ious = iou_batch(trk_box.reshape(1, 4), det_boxes).reshape(-1)
                best_idx = int(np.argmax(ious))

                lane_val = msg.detections[best_idx].lane_id if best_idx < len(msg.detections) else ""
                if not lane_val:
                    # IoU가 낮거나 빈 lane_id일 때 중심점 최근접으로 재선택
                    tcx = (trk_box[0] + trk_box[2]) * 0.5
                    tcy = (trk_box[1] + trk_box[3]) * 0.5
                    det_cx = (det_boxes[:, 0] + det_boxes[:, 2]) * 0.5
                    det_cy = (det_boxes[:, 1] + det_boxes[:, 3]) * 0.5
                    d2 = (det_cx - tcx) ** 2 + (det_cy - tcy) ** 2
                    near_idx = int(np.argmin(d2))
                    lane_val = msg.detections[near_idx].lane_id if near_idx < len(msg.detections) else ""

                d.lane_id = lane_val if lane_val else "UNKNOWN"

            out_msg.detections.append(d)

        self.pub.publish(out_msg)


if __name__ == "__main__":
    try:
        TrackerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
