"""illegal_u_turn: a U-turn manoeuvre (heading reverses by > 150 deg) inside a no_u_turn zone."""
from __future__ import annotations

import numpy as np

from src.events.base import Candidate, merge_intervals
from src.scene.geometry import in_any_zone

MIN_REVERSAL_DEG = 150.0
MIN_MANOEUVRE_SEC = 2.0
MAX_MANOEUVRE_SEC = 12.0
HEADING_LOOKBACK_SEC = 1.5
MERGE_GAP_SEC = 2.0


class IllegalUTurnRule:
    label = "illegal_u_turn"

    def run(self, track_store, scene, light_log: dict | None = None) -> list[Candidate]:
        if not scene.no_u_turn_zones:
            return []
        cos_limit = float(np.cos(np.radians(MIN_REVERSAL_DEG)))
        candidates: list[Candidate] = []
        for track in track_store.vehicle_tracks():
            pts = track.points
            in_zone = [in_any_zone((p.x_norm, p.y_norm), scene.no_u_turn_zones) for p in pts]
            if not any(in_zone):
                continue
            headings = [track.image_heading_at(p.t_sec, HEADING_LOOKBACK_SEC) for p in pts]
            spans: list[tuple[float, float]] = []
            for i, (pi, hi) in enumerate(zip(pts, headings)):
                if hi is None:
                    continue
                for j in range(i + 1, len(pts)):
                    dt = pts[j].t_sec - pi.t_sec
                    if dt > MAX_MANOEUVRE_SEC:
                        break
                    hj = headings[j]
                    if dt < MIN_MANOEUVRE_SEC or hj is None or float(hi @ hj) > cos_limit:
                        continue
                    if any(in_zone[i : j + 1]):
                        spans.append((pi.t_sec - HEADING_LOOKBACK_SEC, pts[j].t_sec))
                    break
            for start, end in merge_intervals(spans, MERGE_GAP_SEC):
                candidates.append(Candidate(max(0.0, start), end, self.label, conf=0.5, evidence={"track_id": track.track_id}))
        return candidates
