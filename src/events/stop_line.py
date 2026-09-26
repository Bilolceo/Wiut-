"""stop_line: stopped past the stop line on red (CLAUDE.md's CLASSES).

Distinct from `red_light` (actively crossing/driving through on red): here
the vehicle comes to rest on the far side of the line instead of continuing
through the intersection.

"Far side" is decided per-track from the vehicle's own trajectory (the sign
of `signed_distance_to_line` at its first observed point is the "approach"
side; a later point with the opposite sign has crossed) -- this needs no
lane-direction metadata and works for a stop line authored in either
orientation. Requires `light_log`, like `red_light`.
"""
from __future__ import annotations

from src.events.base import Candidate, flags_to_segments, nearest_in_time
from src.scene.geometry import signed_distance_to_line

MIN_STATIONARY_SEC = 2.0
MAX_DISP_NORM = 0.03  # normalized image-space displacement treated as "still"
MAX_LIGHT_GAP_SEC = 1.5


def _stationary_norm(track, t_end: float, window_sec: float, max_disp_norm: float) -> bool:
    window = track.points_in_window(t_end, window_sec)
    if len(window) < 2 or window[0].t_sec > t_end - window_sec + 1e-6:
        return False
    x0, y0 = window[0].x_norm, window[0].y_norm
    return all(((p.x_norm - x0) ** 2 + (p.y_norm - y0) ** 2) ** 0.5 <= max_disp_norm for p in window)


class StopLineRule:
    label = "stop_line"

    def run(self, track_store, scene, light_log: dict | None = None) -> list[Candidate]:
        if not light_log:
            return []
        candidates: list[Candidate] = []

        for track in track_store.vehicle_tracks():
            pts = track.points
            if not pts:
                continue
            for sl in scene.stop_lines:
                if sl.controlled_by not in light_log:
                    continue
                baseline_sign: bool | None = None
                flags: list[tuple[float, bool]] = []
                for p in pts:
                    d = signed_distance_to_line((p.x_norm, p.y_norm), sl.line)
                    if baseline_sign is None and abs(d) > 1e-6:
                        baseline_sign = d > 0
                    beyond = baseline_sign is not None and (d > 0) != baseline_sign
                    red = False
                    if beyond and _stationary_norm(track, p.t_sec, MIN_STATIONARY_SEC, MAX_DISP_NORM):
                        state = nearest_in_time(light_log[sl.controlled_by], p.t_sec, MAX_LIGHT_GAP_SEC)
                        red = state is not None and state.state == "red"
                    flags.append((p.t_sec, red))
                for start, end in flags_to_segments(flags, min_duration_sec=MIN_STATIONARY_SEC, merge_gap_sec=2.0):
                    candidates.append(
                        Candidate(
                            start=start,
                            end=end,
                            label=self.label,
                            conf=0.7,
                            evidence={"track_id": track.track_id, "cls": track.cls, "stop_line": sl.id},
                        )
                    )
        return candidates
