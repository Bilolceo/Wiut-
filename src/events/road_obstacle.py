"""road_obstacle: a static new object on the carriageway that no tracker owns.

Consumes StaticObjectMonitor blobs (track_store.extras["static_objects"],
already held static >= 3 s). A blob is dropped if its centre is off the
road_mask or it overlaps any tracked road user at that instant -- a car
waiting at the red light is static too, but YOLO already explains it.
"""
from __future__ import annotations

from src.events.base import Candidate, flags_to_segments
from src.scene.geometry import point_in_polygon
from src.tracks.kinematics import box_norm, iou

MIN_DURATION_SEC = 3.0
MERGE_GAP_SEC = 2.0
TRACK_OVERLAP_IOU = 0.05


class RoadObstacleRule:
    label = "road_obstacle"

    def run(self, track_store, scene, light_log: dict | None = None) -> list[Candidate]:
        blobs = track_store.extras.get("static_objects", [])
        if not blobs or scene.road_mask is None:
            return []
        tracks = list(track_store.tracks.values())
        flags: dict[float, bool] = {}
        for b in blobs:
            cx, cy = (b.box[0] + b.box[2]) / 2, b.box[3]
            explained = any(
                (p := tr.at(b.t_sec, 0.6)) is not None and iou(box_norm(p), b.box) > TRACK_OVERLAP_IOU for tr in tracks
            )
            hit = point_in_polygon((cx, cy), scene.road_mask) and not explained
            flags[b.t_sec] = flags.get(b.t_sec, False) or hit
        times = sorted(set(flags) | set(track_store.sample_times()))
        series = [(t, flags.get(t, False)) for t in times]
        return [
            # blobs are reported once already static for MIN_DURATION_SEC: the object arrived that much earlier
            Candidate(max(0.0, s - MIN_DURATION_SEC), e, self.label, conf=0.4, evidence={"source": "static_objects"})
            for s, e in flags_to_segments(series, 1.0, MERGE_GAP_SEC)  # already held static upstream
        ]
