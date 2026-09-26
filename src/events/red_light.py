"""red_light: crossing the stop line while the controlling traffic light is
red (CLAUDE.md's CLASSES: "crossing the stop line on red").

Distinct from `stop_line` (stopped past the line on red): here the vehicle's
frame-to-frame motion actually crosses the stop-line segment -- it keeps
going instead of coming to rest beyond it.

Requires `light_log` (per traffic_light-id chronological LightState samples,
from src.perception.pipeline.build_track_store). A stop_line whose
controlled_by id has no matching light_log entry is skipped entirely: we
never guess a light's colour from geometry alone.
"""
from __future__ import annotations

from src.events.base import Candidate, merge_intervals, nearest_in_time
from src.scene.geometry import segments_intersect

CROSSING_EVENT_PAD_SEC = 1.0  # candidate window padding around the crossing instant
MERGE_GAP_SEC = 1.0
MAX_LIGHT_GAP_SEC = 1.5  # ignore a light sample farther than this from the crossing


def _lane_direction_ok(motion: tuple[float, float], tl, scene) -> bool:
    """If the controlling light names a lane, require the vehicle to be
    moving with that lane's direction (into the intersection, not backing
    away over the line). No lane info -> nothing to gate on, so allow it."""
    for lane in scene.lanes:
        if lane.id in tl.lanes:
            d = lane.direction
            return float(motion[0] * d[0] + motion[1] * d[1]) > 0
    return True


class RedLightRule:
    label = "red_light"

    def run(self, track_store, scene, light_log: dict | None = None) -> list[Candidate]:
        if not light_log:
            return []  # no traffic-light state available -> nothing to verify a "red" against
        candidates: list[Candidate] = []
        tl_by_id = {tl.id: tl for tl in scene.traffic_lights}

        for track in track_store.vehicle_tracks():
            pts = track.points
            crossings: list[tuple[float, float]] = []
            for i in range(1, len(pts)):
                a, b = pts[i - 1], pts[i]
                for sl in scene.stop_lines:
                    if sl.controlled_by not in light_log:
                        continue
                    if not segments_intersect(
                        (a.x_norm, a.y_norm), (b.x_norm, b.y_norm), sl.line[0], sl.line[1]
                    ):
                        continue
                    tl = tl_by_id.get(sl.controlled_by)
                    motion = (b.x_norm - a.x_norm, b.y_norm - a.y_norm)
                    if tl is not None and not _lane_direction_ok(motion, tl, scene):
                        continue
                    t_cross = (a.t_sec + b.t_sec) / 2.0
                    state = nearest_in_time(light_log[sl.controlled_by], t_cross, MAX_LIGHT_GAP_SEC)
                    if state is not None and state.state == "red":
                        crossings.append((t_cross - CROSSING_EVENT_PAD_SEC, t_cross + CROSSING_EVENT_PAD_SEC))

            for start, end in merge_intervals(crossings, MERGE_GAP_SEC):
                candidates.append(
                    Candidate(
                        start=max(0.0, start),
                        end=end,
                        label=self.label,
                        conf=0.75,
                        evidence={"track_id": track.track_id, "cls": track.cls},
                    )
                )
        return candidates
