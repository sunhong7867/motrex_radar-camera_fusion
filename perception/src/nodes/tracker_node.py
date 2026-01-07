#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tracker_node.py

역할:
- 프레임별 DetectionArray(검출 bbox 리스트)를 입력으로 받아
  IoU 기반 매칭으로 동일 객체를 track으로 유지하면서 "안정적인 id"를 부여
- output은 동일 msg 타입(DetectionArray)이지만, Detection.id가 tracker가 생성한 track_id로 고정됨
- speed_estimator_node에서 Kalman(속도 안정화)을 id 기준으로 돌리기 위해 필수 단계

입력:
- ~in_topic (perception/DetectionArray)  [default: /perception/detections_roi]

출력:
- ~out_topic (perception/DetectionArray) [default: /perception/tracks]

파라미터:
- ~iou_threshold (default: 0.3) : track-검출 매칭 최소 IoU
- ~use_class (default: True)    : cls가 같을 때만 매칭
- ~max_age_sec (default: 0.8)   : 이 시간 이상 업데이트 없으면 track 제거
- ~min_hits (default: 2)        : 이 횟수 이상 매칭된 track만 출력(안정화)
- ~publish_unconfirmed (default: True) : min_hits 미만 track도 출력할지
- ~min_score (default: 0.0)     : 입력 detection score 필터
- ~min_bbox_area (default: 0.0) : 입력 bbox area 필터
- ~debug/enable (default: True)
- ~debug/topic (default: /perception/tracker_debug)
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import rospy
from std_msgs.msg import String

from perception.msg import DetectionArray, Detection, BoundingBox


def _param(name: str, default):
    return rospy.get_param(name, default) if rospy.has_param(name) else default


def bbox_area(bb: BoundingBox) -> float:
    w = float(bb.xmax) - float(bb.xmin)
    h = float(bb.ymax) - float(bb.ymin)
    if w <= 0 or h <= 0:
        return 0.0
    return w * h


def iou_bbox(a: BoundingBox, b: BoundingBox) -> float:
    ax1, ay1, ax2, ay2 = float(a.xmin), float(a.ymin), float(a.xmax), float(a.ymax)
    bx1, by1, bx2, by2 = float(b.xmin), float(b.ymin), float(b.xmax), float(b.ymax)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


@dataclass
class Track:
    tid: int
    cls: int
    bbox: BoundingBox
    score: float
    created_ts: float
    last_ts: float
    hits: int = 1
    misses: int = 0


class TrackerNode:
    def __init__(self):
        self.in_topic = _param("~in_topic", "/perception/detections_roi")
        self.out_topic = _param("~out_topic", "/perception/tracks")

        self.iou_threshold = float(_param("~iou_threshold", 0.30))
        self.use_class = bool(_param("~use_class", True))
        self.max_age_sec = float(_param("~max_age_sec", 0.8))
        self.min_hits = int(_param("~min_hits", 2))
        self.publish_unconfirmed = bool(_param("~publish_unconfirmed", True))

        self.min_score = float(_param("~min_score", 0.0))
        self.min_bbox_area = float(_param("~min_bbox_area", 0.0))

        self.debug_enable = bool(_param("~debug/enable", True))
        self.debug_topic = _param("~debug/topic", "/perception/tracker_debug")
        self.pub_debug = rospy.Publisher(self.debug_topic, String, queue_size=3) if self.debug_enable else None

        self.pub_out = rospy.Publisher(self.out_topic, DetectionArray, queue_size=5)
        rospy.Subscriber(self.in_topic, DetectionArray, self._on_dets, queue_size=1)

        self._tracks: Dict[int, Track] = {}
        self._next_id = 1

        rospy.loginfo(f"[tracker] in={self.in_topic} out={self.out_topic}")
        rospy.loginfo(f"[tracker] iou_th={self.iou_threshold} use_class={self.use_class} max_age={self.max_age_sec}s min_hits={self.min_hits}")

    def _now(self, msg_header_stamp) -> float:
        # ROS time stamp가 들어오면 그걸 쓰고, 없으면 wall time 사용
        try:
            if msg_header_stamp is not None and msg_header_stamp.to_sec() > 0:
                return float(msg_header_stamp.to_sec())
        except Exception:
            pass
        return time.time()

    def _filter_dets(self, dets: List[Detection]) -> List[Detection]:
        out = []
        for d in dets:
            if float(d.score) < self.min_score:
                continue
            if self.min_bbox_area > 0.0 and bbox_area(d.bbox) < self.min_bbox_area:
                continue
            out.append(d)
        return out

    def _cleanup_dead_tracks(self, now: float):
        dead = []
        for tid, trk in self._tracks.items():
            if (now - trk.last_ts) > self.max_age_sec:
                dead.append(tid)
        for tid in dead:
            self._tracks.pop(tid, None)

    def _greedy_match(
        self,
        dets: List[Detection],
        track_ids: List[int]
    ) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
        """
        returns:
        - matches: list of (det_idx, track_idx, iou)
        - unmatched_det_idx list
        - unmatched_track_idx list
        """
        if len(dets) == 0 or len(track_ids) == 0:
            return [], list(range(len(dets))), list(range(len(track_ids)))

        T = len(track_ids)
        D = len(dets)
        iou_mat = np.zeros((D, T), dtype=np.float64)

        for di, d in enumerate(dets):
            for ti, tid in enumerate(track_ids):
                trk = self._tracks[tid]
                if self.use_class and int(d.cls) != int(trk.cls):
                    iou_mat[di, ti] = 0.0
                else:
                    iou_mat[di, ti] = iou_bbox(d.bbox, trk.bbox)

        matches = []
        det_used = set()
        trk_used = set()

        while True:
            # 가장 큰 IoU 찾기
            max_idx = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
            max_val = float(iou_mat[max_idx])
            if max_val < self.iou_threshold:
                break

            di, ti = int(max_idx[0]), int(max_idx[1])
            if di in det_used or ti in trk_used:
                iou_mat[di, ti] = 0.0
                continue

            matches.append((di, ti, max_val))
            det_used.add(di)
            trk_used.add(ti)

            # greedy니까 해당 행/열 제거 효과로 0 처리
            iou_mat[di, :] = 0.0
            iou_mat[:, ti] = 0.0

        unmatched_dets = [i for i in range(D) if i not in det_used]
        unmatched_trks = [i for i in range(T) if i not in trk_used]
        return matches, unmatched_dets, unmatched_trks

    def _create_track(self, d: Detection, now: float) -> int:
        tid = self._next_id
        self._next_id += 1

        bb = BoundingBox(xmin=int(d.bbox.xmin), ymin=int(d.bbox.ymin), xmax=int(d.bbox.xmax), ymax=int(d.bbox.ymax))
        self._tracks[tid] = Track(
            tid=tid,
            cls=int(d.cls),
            bbox=bb,
            score=float(d.score),
            created_ts=now,
            last_ts=now,
            hits=1,
            misses=0
        )
        return tid

    def _on_dets(self, msg: DetectionArray):
        now = self._now(msg.header.stamp)
        self._cleanup_dead_tracks(now)

        dets = self._filter_dets(list(msg.detections))
        track_ids = list(self._tracks.keys())

        matches, unmatched_det_idx, unmatched_trk_idx = self._greedy_match(dets, track_ids)

        # 1) 매칭된 track 업데이트
        for di, ti, _ in matches:
            d = dets[di]
            tid = track_ids[ti]
            trk = self._tracks[tid]
            trk.bbox = BoundingBox(
                xmin=int(d.bbox.xmin), ymin=int(d.bbox.ymin),
                xmax=int(d.bbox.xmax), ymax=int(d.bbox.ymax)
            )
            trk.score = float(d.score)
            trk.cls = int(d.cls)
            trk.last_ts = now
            trk.hits += 1
            trk.misses = 0
            self._tracks[tid] = trk

        # 2) 매칭 안 된 track은 misses 증가 (시간 기반 제거는 cleanup에서 함)
        for ti in unmatched_trk_idx:
            tid = track_ids[ti]
            trk = self._tracks[tid]
            trk.misses += 1
            self._tracks[tid] = trk

        # 3) 매칭 안 된 detection은 새 track 생성
        new_cnt = 0
        for di in unmatched_det_idx:
            self._create_track(dets[di], now)
            new_cnt += 1

        # 4) 출력: track들을 DetectionArray로 변환
        out = DetectionArray()
        out.header = msg.header
        out.detections = []

        pub_cnt = 0
        for tid, trk in self._tracks.items():
            confirmed = (trk.hits >= self.min_hits)
            if (not self.publish_unconfirmed) and (not confirmed):
                continue

            d = Detection()
            d.id = int(tid)
            d.cls = int(trk.cls)
            d.score = float(trk.score)
            d.bbox = trk.bbox
            out.detections.append(d)
            pub_cnt += 1

        self.pub_out.publish(out)

        if self.pub_debug:
            self.pub_debug.publish(String(
                data=f"det_in={len(msg.detections)} det_used={len(dets)} tracks={len(self._tracks)} "
                     f"matched={len(matches)} new={new_cnt} pub={pub_cnt}"
            ))


def main():
    rospy.init_node("tracker_node", anonymous=False)
    TrackerNode()
    rospy.spin()


if __name__ == "__main__":
    main()
