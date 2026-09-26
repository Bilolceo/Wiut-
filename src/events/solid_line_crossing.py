"""solid_line_crossing: a vehicle's ground point crosses a solid lane marking.

Box-bottom jitter makes a car straddling a line flip sides frame to frame,
so a crossing only counts when the vehicle stays on the original side for
SETTLE_SEC before and on the new side for SETTLE_SEC after.
"""
from __future__ import annotations

from src.events.base import Candidate, merge_intervals
from src.scene.geometry import polyline_segments, segments_intersect, signed_distance_to_line

SETTLE_SEC = 0.5
EVENT_PAD_SEC = 1.0
MERGE_GAP_SEC = 1.0


def _side(p, seg) -> bool:
    return signed_distance_to_line((p.x_norm, p.y_norm), seg) > 0


class SolidLineCrossingRule:
    label = "solid_line_crossing"

    def run(self, track_store, scene, light_log: dict | None = None) -> list[Candidate]:
        segments = [(line.id, (a, b)) for line in scene.solid_lines for a, b in polyline_segments(line.polygon)]
        if not segments:
            return []
        candidates: list[Candidate] = []
        for track in track_store.vehicle_tracks():
            pts = track.points
            hits: list[tuple[float, float]] = []
            for i in range(1, len(pts)):
                a, b = pts[i - 1], pts[i]
                for _line_id, seg in segments:
                    if not segments_intersect((a.x_norm, a.y_norm), (b.x_norm, b.y_norm), seg[0], seg[1]):
                        continue
                    before = [p for p in pts[:i] if a.t_sec - p.t_sec <= SETTLE_SEC]
                    after = [p for p in pts[i:] if p.t_sec - b.t_sec <= SETTLE_SEC]
                    side_a = _side(a, seg)
                    if (
                        before[0].t_sec <= a.t_sec - SETTLE_SEC + 1e-6
                        and after[-1].t_sec >= b.t_sec + SETTLE_SEC - 1e-6
                        and all(_side(p, seg) == side_a for p in before)
                        and all(_side(p, seg) != side_a for p in after)
                    ):
                        t = (a.t_sec + b.t_sec) / 2
                        hits.append((t - EVENT_PAD_SEC, t + EVENT_PAD_SEC))
            for start, end in merge_intervals(hits, MERGE_GAP_SEC):
                candidates.append(Candidate(max(0.0, start), end, self.label, conf=0.5, evidence={"track_id": track.track_id}))
        return candidates
