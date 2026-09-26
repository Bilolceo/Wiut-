"""Per-track kinematics shared by the rules and the risk estimator.

Everything here reads only a track's own points up to time ``t`` (causal),
so the same helpers are safe inside RiskEstimator.step.
"""
from __future__ import annotations

import numpy as np

from src.tracks.store import Track, TrackPoint

RIDER_CLASSES = {"bicycle", "motorcycle"}


def point_at(track: Track, t: float, max_gap_sec: float = 0.25) -> TrackPoint | None:
    """The track's point observed at (or within `max_gap_sec` of) time t."""
    return track.at(t, max_gap_sec)


def velocity_mps(track: Track, t: float, lookback_sec: float = 1.0) -> np.ndarray | None:
    """World-frame velocity vector (m/s) over the last `lookback_sec` before t."""
    window = track.points_in_window(t, lookback_sec)
    if len(window) < 2 or any(p.x_m is None for p in window):
        return None
    a, b = window[0], window[-1]
    dt = b.t_sec - a.t_sec
    if dt < 0.2:
        return None
    return np.array([(b.x_m - a.x_m) / dt, (b.y_m - a.y_m) / dt])


def acceleration_mps2(track: Track, t: float, span_sec: float = 1.0) -> float | None:
    """Signed change of speed (m/s^2) between [t-2*span, t-span] and [t-span, t]; negative = braking."""
    v_now = velocity_mps(track, t, span_sec)
    v_before = velocity_mps(track, t - span_sec, span_sec)
    if v_now is None or v_before is None:
        return None
    return float((np.linalg.norm(v_now) - np.linalg.norm(v_before)) / span_sec)


def box_norm(p: TrackPoint) -> tuple[float, float, float, float]:
    """(x1, y1, x2, y2) from the bottom-centre point and box size, normalized coords."""
    return (p.x_norm - p.w_norm / 2, p.y_norm - p.h_norm, p.x_norm + p.w_norm / 2, p.y_norm)


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def is_rider(ped_point: TrackPoint, t: float, rider_tracks: list[Track]) -> bool:
    """A "person" detection sitting on a bicycle/motorcycle is a rider, not a pedestrian."""
    pb = box_norm(ped_point)
    for tr in rider_tracks:
        q = point_at(tr, t)
        if q is not None and iou(pb, box_norm(q)) > 0.1:
            return True
        if q is not None and abs(q.x_norm - ped_point.x_norm) < max(q.w_norm, 0.01) and abs(q.y_norm - ped_point.y_norm) < 0.02:
            return True
    return False


def time_to_collision(
    p1: np.ndarray, v1: np.ndarray, p2: np.ndarray, v2: np.ndarray, radius_m: float
) -> float | None:
    """Constant-velocity time until the two centres are within `radius_m`.

    0.0 if already within the radius AND still closing in, None if they never
    get that close (moving apart, side by side, or the closest approach is
    wider than the radius). Two cars already level in adjacent lanes are not
    a conflict -- that case made Part B fire constantly on the samples.
    """
    r = p2 - p1
    v = v2 - v1
    c = float(r @ r) - radius_m**2
    a = float(v @ v)
    b = 2.0 * float(r @ v)
    if c <= 0:
        return 0.0 if b < 0 else None
    if a < 1e-9 or b >= 0:  # no relative motion, or not closing in
        return None
    disc = b * b - 4 * a * c
    if disc < 0:
        return None  # passes by without entering the radius
    return (-b - disc**0.5) / (2 * a)


def drac_mps2(p1: np.ndarray, v1: np.ndarray, p2: np.ndarray, v2: np.ndarray, radius_m: float) -> float:
    """Deceleration rate to avoid crash: closing speed^2 / (2 * gap). 0 if not closing."""
    r = p2 - p1
    dist = float(np.linalg.norm(r))
    if dist < 1e-6:
        return 0.0
    closing = -float((v2 - v1) @ (r / dist))
    gap = max(dist - radius_m, 0.1)
    return closing**2 / (2 * gap) if closing > 0 else 0.0
