#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tracker_node.py
역할: YOLO 검출 결과(Bounding Box)를 받아 칼만 필터(SORT)로 추적하여 ID를 부여합니다.
"""

import rospy
import numpy as np
from perception.msg import DetectionArray, Detection, BoundingBox
from filterpy.kalman import KalmanFilter

# ==============================================================================
# 1. SORT Algorithm Classes (KalmanBoxTracker, Sort)
# ==============================================================================

def linear_assignment(cost_matrix):
    try:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))
    except ImportError:
        rospy.logerr("Scipy not found. Please install: pip3 install scipy")
        return np.empty((0, 2))

def iou_batch(bb_test, bb_gt):
    """
    bb_test: (N, 4)
    bb_gt: (K, 4)
    Returns: (N, K) IoU matrix
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1]) +
              (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o

def convert_bbox_to_z(bbox):
    """
    Takes a bounding box [x1,y1,x2,y2] and returns form [x,y,s,r] where x,y is center
    s is scale/area and r is aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))

class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self, bbox):
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4) 
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  
                              [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        # [중요] 배열의 첫 번째 요소(bbox)를 반환
        return self.history[-1][0] 

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)[0]

class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.1):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        self.frame_count += 1
        
        # 1. Predict existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        
        for t, trk in enumerate(trks):
            # [수정] predict()가 bbox 좌표 [x1, y1, x2, y2]를 반환함
            pos = self.trackers[t].predict()
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
                
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
            
        # 2. Match detections to trackers
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # 3. Update matched trackers
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # 4. Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            self.trackers.append(trk)
            
        # 5. Output logic
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))
            i -= 1
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
                
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if len(trackers)==0:
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty((0,2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:,0]:
            unmatched_detections.append(d)
            
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:,1]:
            unmatched_trackers.append(t)

    # Filter out matches with low IoU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
            
    if len(matches) == 0:
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


# ==============================================================================
# 2. ROS Wrapper Node
# ==============================================================================

class TrackerNode:
    def __init__(self):
        rospy.init_node('tracker_node', anonymous=True)
        
        # Parameters
        self.iou_thres = rospy.get_param('~iou_threshold', 0.1)
        self.min_hits = rospy.get_param('~min_hits', 1)
        self.max_age = rospy.get_param('~max_age', 15) 
        
        self.tracker = Sort(max_age=self.max_age, min_hits=self.min_hits, iou_threshold=self.iou_thres)
        
        self.sub = rospy.Subscriber(rospy.get_param('~in_topic', '/perception/detections'), 
                                    DetectionArray, self.callback, queue_size=1)
        self.pub = rospy.Publisher(rospy.get_param('~out_topic', '/perception/tracks'), 
                                   DetectionArray, queue_size=1)
        
        rospy.loginfo(f"[tracker] iou_th={self.iou_thres} max_age={self.max_age} min_hits={self.min_hits}")

    def callback(self, msg):
        # 1. Convert DetectionArray -> Numpy
        dets_list = []
        for d in msg.detections:
            # [x1, y1, x2, y2, score]
            dets_list.append([d.bbox.xmin, d.bbox.ymin, d.bbox.xmax, d.bbox.ymax, d.score])
        
        dets = np.array(dets_list)
        if len(dets) == 0:
            dets = np.empty((0, 5))

        # 2. SORT Update
        try:
            trackers = self.tracker.update(dets)
        except Exception as e:
            rospy.logerr(f"Tracker Update Error: {e}")
            return

        # 3. Publish Result
        out_msg = DetectionArray()
        out_msg.header = msg.header
        
        for trk in trackers:
            # trk: [x1, y1, x2, y2, id]
            d = Detection()
            d.id = int(trk[4])
            d.score = 1.0 # Tracking success
            
            bbox = BoundingBox()
            bbox.xmin = int(trk[0])
            bbox.ymin = int(trk[1])
            bbox.xmax = int(trk[2])
            bbox.ymax = int(trk[3])
            d.bbox = bbox
            
            out_msg.detections.append(d)
            
        self.pub.publish(out_msg)

if __name__ == '__main__':
    try:
        TrackerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass