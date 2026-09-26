"""wrong_way: driving against the traffic direction / in the oncoming lane.

Compares each vehicle's recent heading (image-space, no homography needed)
against the direction of the lane polygon it is in (configs/scene_manual.json).

Heading is only judged while the vehicle actually moves: on C3896 all 325
raw candidates were vehicles standing in the red-light queue whose box-bottom
jitter produced random headings (median net displacement 0.001 of the frame).
A candidate must also cover real distance against the lane direction.
"""
from __future__ import annotations

import numpy as np

from src.events.base import Candidate, flags_to_segments

MIN_SPEED_MPS = 2.0  # ~7 km/h: below this the heading is box jitter
MIN_SPEED_NORM_PER_SEC = 0.02  # fallback without metric calibration
OPPOSING_COS_THRESHOLD = -0.5  # heading . lane_direction below this = against traffic (> 120 deg off)
LOOKBACK_SEC = 1.5
MIN_DURATION_SEC = 2.0
MIN_AGAINST_M = 5.0  # net travel against the lane direction over the segment
MIN_AGAINST_NORM = 0.03


def _moving(track, t: float) -> bool:
    speed = track.speed_mps_at(t, LOOKBACK_SEC)
    if speed is not None:
        return speed >= MIN_SPEED_MPS
    window = track.points_in_window(t, LOOKBACK_SEC)
    if len(window) < 2 or window[-1].t_sec <= window[0].t_sec:
        return False
    a, b = window[0], window[-1]
    return np.hypot(b.x_norm - a.x_norm, b.y_norm - a.y_norm) / (b.t_sec - a.t_sec) >= MIN_SPEED_NORM_PER_SEC


def _travel_against(points, lane_dir: np.ndarray) -> tuple[float, bool]:
    """(distance travelled against lane_dir, in metres if available) over the points."""
    a, b = points[0], points[-1]
    along_img = float(np.dot([b.x_norm - a.x_norm, b.y_norm - a.y_norm], lane_dir))
    if a.x_m is not None and b.x_m is not None:
        # lane directions are image-space; the sign is what matters, the length comes from metres
        return (np.hypot(b.x_m - a.x_m, b.y_m - a.y_m) if along_img < 0 else 0.0), True
    return -along_img, False


class WrongWayRule:
    label = "wrong_way"

    def run(self, track_store, scene, light_log: dict | None = None) -> list[Candidate]:
        candidates: list[Candidate] = []
        if not scene.lanes:
            return candidates  # nothing to compare against yet

        for track in track_store.vehicle_tracks():
            flags: list[tuple[float, bool]] = []
            lane_of: dict[float, np.ndarray] = {}
            for p in track.points:
                lane = scene.lane_at(p.x_norm, p.y_norm)
                flagged = False
                if lane is not None and np.linalg.norm(lane.direction) > 1e-6 and _moving(track, p.t_sec):
                    heading = track.image_heading_at(p.t_sec, LOOKBACK_SEC)
                    lane_dir = lane.direction / np.linalg.norm(lane.direction)
                    if heading is not None and float(np.dot(heading, lane_dir)) < OPPOSING_COS_THRESHOLD:
                        flagged = True
                        lane_of[p.t_sec] = lane_dir
                flags.append((p.t_sec, flagged))
            for start, end in flags_to_segments(flags, min_duration_sec=MIN_DURATION_SEC, merge_gap_sec=1.0):
                pts = track.points_in_window(end, end - start)
                lane_dir = next((lane_of[p.t_sec] for p in pts if p.t_sec in lane_of), None)
                if lane_dir is None:
                    continue
                dist, metric = _travel_against(pts, lane_dir)
                if dist < (MIN_AGAINST_M if metric else MIN_AGAINST_NORM):
                    continue
                candidates.append(
                    Candidate(
                        start=start, end=end, label=self.label, conf=0.6,
                        evidence={"track_id": track.track_id, "cls": track.cls, "against_m": round(dist, 1)},
                    )
                )
        return candidates
