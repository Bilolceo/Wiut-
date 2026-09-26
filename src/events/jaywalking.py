"""jaywalking: pedestrian on the carriageway outside a crossing.

A pedestrian's ground point (bottom-centre of the box) counts as "on the
carriageway" when it is inside road_mask but not on a crosswalk or a refuge
island (small margin for foot/box jitter at the zebra edge). "person"
detections riding a bicycle/motorcycle are riders, not pedestrians.
Without a road_mask the rule abstains.

Segment-level gates, from the sample videos: a jaywalker walks. Standing
"pedestrians" on the carriageway were a moped rider waiting at the stop
line whose moped went undetected, or detector noise -- both with mean box
confidence ~0.33 vs >= 0.56 for real walkers.
"""
from __future__ import annotations

import numpy as np

from src.events.base import Candidate, flags_to_segments
from src.scene.geometry import in_any_zone, point_in_polygon
from src.tracks.kinematics import RIDER_CLASSES, is_rider

MIN_DURATION_SEC = 2.0
MERGE_GAP_SEC = 1.0
MIN_MEAN_CONF = 0.5
MIN_SPEED_NORM_PER_SEC = 0.008  # median image-space speed over the segment
# Walking along the zebra edge or stepping off it onto the kerb is not
# jaywalking: allow ~1 m. Perspective-aware -- the margin scales with the
# person's own box height (~1.7 m tall; normalized x is 16/9 wider than y),
# so it means the same distance in the foreground and far away.
MIN_ZONE_MARGIN_NORM = 0.015
MARGIN_PER_BOX_HEIGHT = 0.3  # ~1 m: a 1.7 m person is ~150-220 px tall at 4K


def _walking_person(points) -> bool:
    if len(points) < 2 or float(np.mean([p.conf for p in points])) < MIN_MEAN_CONF:
        return False
    steps = [
        np.hypot(b.x_norm - a.x_norm, b.y_norm - a.y_norm) / (b.t_sec - a.t_sec)
        for a, b in zip(points, points[1:])
        if b.t_sec > a.t_sec
    ]
    return bool(steps) and float(np.median(steps)) >= MIN_SPEED_NORM_PER_SEC


class JaywalkingRule:
    label = "jaywalking"

    def run(self, track_store, scene, light_log: dict | None = None) -> list[Candidate]:
        if scene.road_mask is None:
            return []
        riders = [t for t in track_store.tracks.values() if t.cls in RIDER_CLASSES]
        safe_zones = list(scene.crosswalks) + list(scene.refuges)
        candidates: list[Candidate] = []
        for track in track_store.pedestrian_tracks():
            flags = []
            for p in track.points:
                pt = (p.x_norm, p.y_norm)
                margin = max(MIN_ZONE_MARGIN_NORM, MARGIN_PER_BOX_HEIGHT * p.h_norm)
                on_road = point_in_polygon(pt, scene.road_mask) and not in_any_zone(pt, safe_zones, margin)
                flags.append((p.t_sec, on_road and not is_rider(p, p.t_sec, riders)))
            for start, end in flags_to_segments(flags, MIN_DURATION_SEC, MERGE_GAP_SEC):
                if not _walking_person(track.points_in_window(end, end - start)):
                    continue
                candidates.append(
                    Candidate(start, end, self.label, conf=0.6, evidence={"track_id": track.track_id})
                )
        return candidates
