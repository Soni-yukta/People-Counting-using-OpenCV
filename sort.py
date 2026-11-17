"""
Asynchronous / Real-time Simple Online and Realtime Tracking (SORT).
Refactored to be self-contained, high-performance, and free of obsolete dependencies
(no filterpy, no lap compilation dependencies, no skimage or matplotlib module-level imports).
"""

from __future__ import print_function
import numpy as np
from scipy.optimize import linear_sum_assignment


class KalmanFilter(object):
    """
    Lightweight, self-contained 2D Constant Velocity Kalman Filter.
    Replaces external filterpy dependency with pure NumPy.
    State vector: [u, v, s, r, u_dot, v_dot, s_dot]^T (dim_x = 7)
    Measurement vector: [u, v, s, r]^T (dim_z = 4)
    where:
      u, v : bounding box center coordinates
      s    : scale (area)
      r    : aspect ratio (w / h)
    """
    def __init__(self, dim_x=7, dim_z=4):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.x = np.zeros((dim_x, 1), dtype=np.float64)
        self.P = np.eye(dim_x, dtype=np.float64)
        self.Q = np.eye(dim_x, dtype=np.float64)
        self.F = np.eye(dim_x, dtype=np.float64)
        self.H = np.zeros((dim_z, dim_x), dtype=np.float64)
        self.R = np.eye(dim_z, dtype=np.float64)
        self._I = np.eye(dim_x, dtype=np.float64)

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        if z is None:
            return
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        try:
            K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        except np.linalg.LinAlgError:
            K = np.dot(np.dot(self.P, self.H.T), np.linalg.pinv(S))
        self.x = self.x + np.dot(K, y)
        self.P = np.dot(self._I - np.dot(K, self.H), self.P)


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    Computes IoU between two arrays of bounding boxes in [x1,y1,x2,y2] format.
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
    denom = (
        (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
        - wh
    )
    return np.where(denom > 0, wh / denom, 0.0)


def convert_bbox_to_z(bbox):
    """
    Converts [x1, y1, x2, y2] to center form [x, y, s, r]^T.
    """
    b = np.asarray(bbox).flatten()
    w = max(1.0, float(b[2] - b[0]))
    h = max(1.0, float(b[3] - b[1]))
    x = float(b[0]) + w / 2.0
    y = float(b[1]) + h / 2.0
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r], dtype=np.float64).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Converts state vector [x, y, s, r] back to [x1, y1, x2, y2].
    """
    vals = np.asarray(x).flatten()
    w = np.sqrt(max(1.0, float(vals[2] * vals[3])))
    h = max(1.0, float(vals[2] / max(1e-6, w)))
    cx = float(vals[0])
    cy = float(vals[1])
    if score is None:
        return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]).reshape((1, 4))
    else:
        return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0, score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    Tracks an individual bounding box using the constant-velocity Kalman filter.
    Maintains centroid trajectory history for precise vector and line-crossing calculation.
    """
    count = 0

    @classmethod
    def reset_id_count(cls, start_id=0):
        cls.count = start_id

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float64)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ], dtype=np.float64)

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
        
        # Trajectory history: list of (cx, cy) tuples (max 50 points)
        cx = float(bbox[0] + bbox[2]) / 2.0
        cy = float(bbox[1] + bbox[3]) / 2.0
        self.centroids = [(cx, cy)]

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        
        cx = float(bbox[0] + bbox[2]) / 2.0
        cy = float(bbox[1] + bbox[3]) / 2.0
        self.centroids.append((cx, cy))
        if len(self.centroids) > 50:
            self.centroids.pop(0)

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)

    def get_last_two_centroids(self):
        """Returns (P_prev, P_curr) for motion vector and line intersection."""
        if len(self.centroids) >= 2:
            return self.centroids[-2], self.centroids[-1]
        elif len(self.centroids) == 1:
            return self.centroids[0], self.centroids[0]
        return None, None


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
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
        matched_indices = np.empty(shape=(0, 2), dtype=int)

    unmatched_detections = []
    for d in range(len(detections)):
        if len(matched_indices) == 0 or d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t in range(len(trackers)):
        if len(matched_indices) == 0 or t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # Filter out matches with low IoU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    """
    SORT Multi-Object Tracker.
    """
    def __init__(self, max_age=20, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def reset(self):
        """Resets tracker state and counter for repeatable clean runs."""
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.reset_id_count(1)

    def get_track_by_id(self, track_id):
        """Retrieves a tracker instance by ID."""
        for trk in self.trackers:
            if trk.id == track_id:
                return trk
        return None

    def update(self, dets=np.empty((0, 5))):
        """
        Updates the tracker with detections in the current frame.
        dets: numpy array of format [[x1, y1, x2, y2, score], ...]
        Returns: numpy array of format [[x1, y1, x2, y2, id], ...]
        """
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold
        )

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or trk.hits >= self.min_hits or self.frame_count <= self.min_hits
            ):
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
