"""A BoT-SORT-style multi-object tracker (MIT-licensed reimplementation, no ReID).

Follows the BoT-SORT recipe (Aharon et al., 2022), built on ByteTrack:

* Kalman filter whose state uses box width and height directly (x, y, w, h + velocities),
  with process/measurement noise proportional to the box size;
* two-stage association: high-score detections first, then the low-score ones recover
  tracks that would otherwise be lost (occlusion, blur, dim monochrome frames);
* optional camera-motion compensation (sparse optical flow + partial affine) for moving
  cameras – a laptop camera on a desk usually does not need it;
* explicit track lifecycle state machine: TENTATIVE -> CONFIRMED <-> LOST -> REMOVED.

Appearance re-identification (the "ReID" branch of BoT-SORT) is left out: it needs an extra
network, and for a single static laptop camera IoU + motion is usually enough.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

TENTATIVE, CONFIRMED, LOST, REMOVED = "TENTATIVE", "CONFIRMED", "LOST", "REMOVED"


class KalmanXYWH:
    """Constant-velocity Kalman filter over (cx, cy, w, h) as in BoT-SORT."""

    std_pos, std_vel = 1.0 / 20, 1.0 / 160

    def __init__(self):
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = 1.0
        self.H = np.eye(4, 8)

    def initiate(self, z: np.ndarray):
        mean = np.r_[z, np.zeros(4)]
        w, h = z[2], z[3]
        std = [2 * self.std_pos * w, 2 * self.std_pos * h, 2 * self.std_pos * w, 2 * self.std_pos * h,
               10 * self.std_vel * w, 10 * self.std_vel * h, 10 * self.std_vel * w, 10 * self.std_vel * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        w, h = mean[2], mean[3]
        q = np.diag(np.square([self.std_pos * w, self.std_pos * h, self.std_pos * w, self.std_pos * h,
                               self.std_vel * w, self.std_vel * h, self.std_vel * w, self.std_vel * h]))
        return self.F @ mean, self.F @ cov @ self.F.T + q

    def update(self, mean, cov, z):
        w, h = mean[2], mean[3]
        r = np.diag(np.square([self.std_pos * w, self.std_pos * h, self.std_pos * w, self.std_pos * h]))
        s = self.H @ cov @ self.H.T + r
        k = cov @ self.H.T @ np.linalg.inv(s)
        mean = mean + k @ (z - self.H @ mean)
        cov = (np.eye(8) - k @ self.H) @ cov
        return mean, cov


def xyxy_to_xywh(b) -> np.ndarray:
    x1, y1, x2, y2 = b
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)], dtype=float)


def xywh_to_xyxy(m) -> Tuple[float, float, float, float]:
    cx, cy, w, h = m[:4]
    return (float(cx - w / 2), float(cy - h / 2), float(cx + w / 2), float(cy + h / 2))


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def assign(cost: np.ndarray, thresh: float):
    """Hungarian assignment; pairs with cost > thresh are rejected."""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(cost)
    matches = [(r, c) for r, c in zip(rows, cols) if cost[r, c] <= thresh]
    mr, mc = {r for r, _ in matches}, {c for _, c in matches}
    return matches, [r for r in range(cost.shape[0]) if r not in mr], [c for c in range(cost.shape[1]) if c not in mc]


@dataclass
class Track:
    tid: int
    cls: int
    score: float
    mean: np.ndarray
    cov: np.ndarray
    state: str = TENTATIVE
    hits: int = 1
    start_frame: int = 0
    last_frame: int = 0
    first_time: float = 0.0
    last_time: float = 0.0
    history: List[str] = field(default_factory=list)   # lifecycle transitions, for audit

    @property
    def box(self):
        return xywh_to_xyxy(self.mean)

    def set_state(self, s: str, frame: int):
        if s != self.state:
            self.history.append(f"{frame}:{self.state}->{s}")
            self.state = s


class BotSortLite:
    def __init__(self, track_high_thresh: float = 0.5, track_low_thresh: float = 0.1, new_track_thresh: float = 0.6,
                 match_thresh: float = 0.8, track_buffer: int = 30, frame_rate: float = 30.0, fuse_score: bool = True,
                 class_aware: bool = True, gmc: bool = False):
        self.high, self.low, self.new_thr, self.match_thr = track_high_thresh, track_low_thresh, new_track_thresh, match_thresh
        self.max_lost = max(1, int(round(frame_rate / 30.0 * track_buffer)))
        self.fuse_score, self.class_aware, self.gmc_on = fuse_score, class_aware, gmc
        self.kf = KalmanXYWH()
        self.tracks: List[Track] = []
        self.removed: List[Track] = []
        self.frame = 0
        self._next_id = 1
        self._prev_gray = None

    # ---------------------------------------------------------------- helpers
    def _cost(self, tracks: Sequence[Track], dets, fuse: Optional[bool] = None) -> np.ndarray:
        if not tracks or not dets:
            return np.zeros((len(tracks), len(dets)))
        tb = np.array([t.box for t in tracks])
        db = np.array([d.box for d in dets])
        sim = iou_matrix(tb, db)
        if self.fuse_score if fuse is None else fuse:
            sim = sim * np.array([d.score for d in dets])[None, :]
        cost = 1.0 - sim
        if self.class_aware:
            same = np.array([[t.cls == d.cls for d in dets] for t in tracks])
            cost = np.where(same, cost, 1e6)
        return cost

    def _gmc(self, gray: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """2x3 affine of the camera motion between the previous and current grey frame."""
        if not self.gmc_on or gray is None:
            return None
        import cv2

        prev, self._prev_gray = self._prev_gray, gray
        if prev is None:
            return None
        pts = cv2.goodFeaturesToTrack(prev, maxCorners=500, qualityLevel=0.01, minDistance=1, blockSize=3)
        if pts is None or len(pts) < 10:
            return None
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None)
        good = st.reshape(-1) == 1
        if good.sum() < 10:
            return None
        A, _ = cv2.estimateAffinePartial2D(pts[good], nxt[good], method=cv2.RANSAC)
        return A

    def _apply_gmc(self, A: np.ndarray):
        R, t = A[:, :2], A[:, 2]
        R8 = np.kron(np.eye(4), R)
        for tr in self.tracks:
            m = R8 @ tr.mean
            m[:2] += t
            tr.mean = m
            tr.cov = R8 @ tr.cov @ R8.T

    def _activate(self, tr: Track, det, t: float):
        tr.mean, tr.cov = self.kf.update(tr.mean, tr.cov, xyxy_to_xywh(det.box))
        tr.score, tr.hits, tr.last_frame, tr.last_time = det.score, tr.hits + 1, self.frame, t
        tr.set_state(CONFIRMED, self.frame)

    # ---------------------------------------------------------------- main step
    def update(self, detections, t: Optional[float] = None, gray: Optional[np.ndarray] = None) -> List[Track]:
        """One frame. Returns the CONFIRMED tracks matched in this frame."""
        self.frame += 1
        t = float(self.frame) if t is None else t
        high = [d for d in detections if d.score >= self.high]
        low = [d for d in detections if self.low <= d.score < self.high]

        for tr in self.tracks:
            tr.mean, tr.cov = self.kf.predict(tr.mean, tr.cov)
        A = self._gmc(gray)
        if A is not None:
            self._apply_gmc(A)

        confirmed = [tr for tr in self.tracks if tr.state in (CONFIRMED, LOST)]
        tentative = [tr for tr in self.tracks if tr.state == TENTATIVE]

        # stage 1: confirmed + lost tracks <-> high-score detections
        m1, ut1, ud1 = assign(self._cost(confirmed, high), self.match_thr)
        for r, c in m1:
            self._activate(confirmed[r], high[c], t)
        # stage 2: still-tracked (not lost) tracks <-> low-score detections (ByteTrack recovery)
        rest = [confirmed[i] for i in ut1 if confirmed[i].state == CONFIRMED]
        m2, ut2, _ = assign(self._cost(rest, low, fuse=False), 0.5)   # plain IoU, as in ByteTrack
        for r, c in m2:
            self._activate(rest[r], low[c], t)
        matched = {id(confirmed[r]) for r, _ in m1} | {id(rest[r]) for r, _ in m2}
        for tr in confirmed:
            if id(tr) not in matched and tr.state == CONFIRMED:
                tr.set_state(LOST, self.frame)
        # stage 3: tentative tracks need a second high-score hit
        left = [high[i] for i in ud1]
        m3, ut3, ud3 = assign(self._cost(tentative, left), 0.7)
        for r, c in m3:
            self._activate(tentative[r], left[c], t)
        for i in ut3:
            tentative[i].set_state(REMOVED, self.frame)
        # new tracks from unmatched confident detections
        for i in ud3:
            d = left[i]
            if d.score < self.new_thr:
                continue
            mean, cov = self.kf.initiate(xyxy_to_xywh(d.box))
            self.tracks.append(Track(self._next_id, d.cls, d.score, mean, cov, TENTATIVE, 1, self.frame,
                                     self.frame, t, t, [f"{self.frame}:NEW->TENTATIVE"]))
            self._next_id += 1
        # lost too long -> removed
        for tr in self.tracks:
            if tr.state == LOST and self.frame - tr.last_frame > self.max_lost:
                tr.set_state(REMOVED, self.frame)
        self.removed += [tr for tr in self.tracks if tr.state == REMOVED]
        self.tracks = [tr for tr in self.tracks if tr.state != REMOVED]
        return [tr for tr in self.tracks if tr.state == CONFIRMED and tr.last_frame == self.frame]

    def active(self, include_lost: bool = False) -> List[Track]:
        ok = (CONFIRMED, LOST) if include_lost else (CONFIRMED,)
        return [tr for tr in self.tracks if tr.state in ok]
