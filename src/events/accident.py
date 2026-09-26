"""accident: collision between two road users (LOW-CONFIDENCE candidate generator).

A pair counts when (1) their ground points come within CONTACT_M in metres,
(2) at least one of them was moving fast just before and braked hard
(> 4 m/s^2, docs/PLAN.md section 6), and (3) both then stay (almost) still
for POST_STOP_SEC. Image-space box overlap is not used: in this oblique view
queued cars overlap on screen all the time.

These candidates are deliberately low-confidence: they are meant to be
confirmed by the VLM verifier (src/verify) before being emitted, and the
class stays disabled in configs/thresholds.yaml until that exists.
"""
from __future__ import annotations

from src.events.base import Candidate, merge_intervals
from src.events.pairs import points_by_time, world_distance
from src.tracks.kinematics import acceleration_mps2
from src.tracks.store import VEHICLE_CLASSES

CONTACT_M = 3.0
MIN_PRE_SPEED_MPS = 3.0
HARD_BRAKE_MPS2 = -4.0
POST_STOP_SEC = 3.0
STOPPED_MPS = 1.0
MERGE_GAP_SEC = 3.0


def _stays_stopped(track, t: float) -> bool:
    end = t + POST_STOP_SEC
    if not track.points or track.points[-1].t_sec < end:
        return False
    speed = track.speed_mps_at(end, POST_STOP_SEC)
    return speed is not None and speed < STOPPED_MPS


def _braked_from_speed(track, t: float) -> bool:
    before = track.speed_mps_at(t - 0.5, 1.0)
    if before is None or before < MIN_PRE_SPEED_MPS:
        return False
    return any(
        (a := acceleration_mps2(track, t + dt, 0.5)) is not None and a <= HARD_BRAKE_MPS2
        for dt in (-0.5, 0.0, 0.5, 1.0)
    )


class AccidentRule:
    label = "accident"

    def run(self, track_store, scene, light_log: dict | None = None) -> list[Candidate]:
        index = points_by_time(track_store, VEHICLE_CLASSES | {"pedestrian", "bicycle"})
        spans: list[tuple[float, float, int, int]] = []
        for t, members in index.items():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    (ta, pa), (tb, pb) = members[i], members[j]
                    if ta.cls not in VEHICLE_CLASSES and tb.cls not in VEHICLE_CLASSES:
                        continue
                    d = world_distance(pa, pb)
                    if d is None or d > CONTACT_M:
                        continue
                    if not (_braked_from_speed(ta, t) or _braked_from_speed(tb, t)):
                        continue
                    if _stays_stopped(ta, t) and _stays_stopped(tb, t):
                        spans.append((t, t + POST_STOP_SEC, ta.track_id, tb.track_id))
        candidates = []
        for start, end in merge_intervals([(s, e) for s, e, *_ in spans], MERGE_GAP_SEC):
            ids = sorted({i for s, e, a, b in spans if start <= s <= end for i in (a, b)})
            candidates.append(Candidate(start, end, self.label, conf=0.3, evidence={"track_ids": ids, "needs_verification": True}))
        return candidates
