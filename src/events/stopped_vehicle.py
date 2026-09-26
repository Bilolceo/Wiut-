"""stopped_vehicle: stationary on the carriageway >= 10 s, not queued at a signal.

When `light_log` is available (from src.perception.pipeline.build_track_store),
"queued at a signal" is verified against the light's actual state: stopped
near a stop line while it is red/yellow/unknown is a legitimate queue,
stopped near one while it is verified green is NOT a queue (more likely a
breakdown or double-parked vehicle) and should still be flagged. Without
light_log we fall back to the old proximity-only heuristic.
"""
from __future__ import annotations

from src.events.base import Candidate, flags_to_segments, nearest_in_time
from src.scene.geometry import point_segment_distance

MIN_STATIONARY_SEC = 10.0
MAX_DISPLACEMENT_M = 1.0
STOP_LINE_PROXIMITY_NORM = 0.05  # normalized image-space distance treated as "at a signal"
MAX_LIGHT_GAP_SEC = 1.5


def _queued_at_signal(x_norm: float, y_norm: float, t_sec: float, scene, light_log) -> bool:
    for sl in scene.stop_lines:
        if point_segment_distance((x_norm, y_norm), sl.line) > STOP_LINE_PROXIMITY_NORM:
            continue
        if not light_log or sl.controlled_by not in light_log:
            return True  # no light state to check -> proximity-only fallback (old behaviour)
        state = nearest_in_time(light_log[sl.controlled_by], t_sec, MAX_LIGHT_GAP_SEC)
        # red/yellow/unknown near the line -> plausibly a legitimate queue;
        # only a *verified* green rules out "waiting for the signal".
        return state is None or state.state != "green"
    return False


class StoppedVehicleRule:
    label = "stopped_vehicle"

    def run(self, track_store, scene, light_log: dict | None = None) -> list[Candidate]:
        candidates: list[Candidate] = []
        for track in track_store.vehicle_tracks():
            flags: list[tuple[float, bool]] = []
            for p in track.points:
                stationary = track.is_stationary(p.t_sec, MIN_STATIONARY_SEC, MAX_DISPLACEMENT_M)
                queued = stationary and _queued_at_signal(p.x_norm, p.y_norm, p.t_sec, scene, light_log)
                flags.append((p.t_sec, stationary and not queued))
            for start, end in flags_to_segments(flags, min_duration_sec=1.0, merge_gap_sec=2.0):
                candidates.append(
                    Candidate(
                        start=start,
                        end=end,
                        label=self.label,
                        conf=0.7,
                        evidence={"track_id": track.track_id, "cls": track.cls},
                    )
                )
        return candidates
