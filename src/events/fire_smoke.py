"""fire_smoke: visible fire or smoke (LOW-CONFIDENCE colour/texture heuristic).

Consumes FireSmokeMonitor blobs (track_store.extras["fire_smoke"]), which
are already persistence-gated (>= 2 s, docs/PLAN.md section 6). A starting
point only: headlight glare at dusk and white vehicles are the known false
positives, so the class stays disabled in configs/thresholds.yaml until an
open-vocabulary detector or the VLM verifier confirms candidates.
"""
from __future__ import annotations

from src.events.base import Candidate, flags_to_segments

MIN_DURATION_SEC = 2.0
MERGE_GAP_SEC = 2.0


class FireSmokeRule:
    label = "fire_smoke"

    def run(self, track_store, scene, light_log: dict | None = None) -> list[Candidate]:
        found = track_store.extras.get("fire_smoke", {})
        hit_times = {b.t_sec for kind in ("fire", "smoke") for b in found.get(kind, [])}
        if not hit_times:
            return []
        times = sorted(hit_times | set(track_store.sample_times()))
        series = [(t, t in hit_times) for t in times]
        return [
            Candidate(s, e, self.label, conf=0.3, evidence={"needs_verification": True})
            for s, e in flags_to_segments(series, MIN_DURATION_SEC, MERGE_GAP_SEC)
        ]
