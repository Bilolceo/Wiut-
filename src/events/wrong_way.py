"""wrong_way: driving against the traffic direction / in the oncoming lane.

Compares each vehicle's recent heading (image-space, no homography needed)
against the lane polygon it is currently in. scene.json's lanes[] direction
vectors come from build_scene.py's trajectory clustering, so they already
encode "normal" traffic flow for that lane.
"""
from __future__ import annotations

import numpy as np

from src.events.base import Candidate, flags_to_segments

MIN_SPEED_TO_JUDGE_NORM = 0.01  # ignore near-stationary vehicles (heading is noise)
OPPOSING_COS_THRESHOLD = -0.3  # heading . lane_direction below this = against traffic
LOOKBACK_SEC = 1.5


class WrongWayRule:
    label = "wrong_way"

    def run(self, track_store, scene, light_log: dict | None = None) -> list[Candidate]:
        candidates: list[Candidate] = []
        if not scene.lanes:
            return candidates  # nothing to compare against yet

        for track in track_store.vehicle_tracks():
            flags: list[tuple[float, bool]] = []
            for p in track.points:
                lane = scene.lane_at(p.x_norm, p.y_norm)
                heading = track.image_heading_at(p.t_sec, LOOKBACK_SEC)
                flagged = False
                if lane is not None and heading is not None and np.linalg.norm(lane.direction) > 1e-6:
                    lane_dir = lane.direction / np.linalg.norm(lane.direction)
                    cos_sim = float(np.dot(heading, lane_dir))
                    flagged = cos_sim < OPPOSING_COS_THRESHOLD
                flags.append((p.t_sec, flagged))
            for start, end in flags_to_segments(flags, min_duration_sec=1.0, merge_gap_sec=1.0):
                candidates.append(
                    Candidate(
                        start=start,
                        end=end,
                        label=self.label,
                        conf=0.6,
                        evidence={"track_id": track.track_id, "cls": track.cls},
                    )
                )
        return candidates
