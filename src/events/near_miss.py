"""near_miss: a conflict with sharp braking or swerving and no contact (LOW-CONFIDENCE).

A pair is in conflict when its constant-velocity time-to-collision drops
below TTC_MAX_SEC (docs/PLAN.md section 6: TTC < 1.0 s) and at least one
vehicle reacts evasively within REACTION_SEC: hard braking or a fast heading
change. Pairs that actually touch (within CONTACT_M) belong to `accident`.

Low confidence by design: like accident, these are VLM-verifier candidates,
and the class stays disabled in configs/thresholds.yaml until then.
"""
from __future__ import annotations

import numpy as np

from src.events.base import Candidate, merge_intervals
from src.events.pairs import points_by_time, world_distance
from src.tracks.kinematics import acceleration_mps2, time_to_collision, velocity_mps
from src.tracks.store import VEHICLE_CLASSES

TTC_MAX_SEC = 1.0
RADIUS_M = {"vehicle": 2.0, "pedestrian": 1.0}
CONTACT_M = 3.0
EVASIVE_BRAKE_MPS2 = -3.0
EVASIVE_YAW_DEG = 30.0  # heading change over REACTION_SEC
REACTION_SEC = 2.0
MIN_SPEED_MPS = 2.0
PAD_BEFORE_SEC, PAD_AFTER_SEC = 1.0, 2.0
MERGE_GAP_SEC = 2.0


def _radius(cls: str) -> float:
    return RADIUS_M["vehicle"] if cls in VEHICLE_CLASSES else RADIUS_M["pedestrian"]


def _evasive(track, t: float) -> bool:
    if track.cls not in VEHICLE_CLASSES:
        return False
    for dt in (0.0, 0.5, 1.0, 1.5, 2.0):
        a = acceleration_mps2(track, t + dt, 0.5)
        if a is not None and a <= EVASIVE_BRAKE_MPS2:
            return True
    h0, h1 = track.heading_at(t, 1.0), track.heading_at(t + REACTION_SEC, 1.0)
    return h0 is not None and h1 is not None and float(np.degrees(np.arccos(np.clip(h0 @ h1, -1, 1)))) >= EVASIVE_YAW_DEG


def _touches(ta, tb, t: float) -> bool:
    for dt in np.arange(0.0, REACTION_SEC + 0.01, 0.5):
        pa, pb = ta.at(t + dt), tb.at(t + dt)
        d = world_distance(pa, pb) if pa is not None and pb is not None else None
        if d is not None and d <= CONTACT_M:
            return True
    return False


class NearMissRule:
    label = "near_miss"

    def run(self, track_store, scene, light_log: dict | None = None) -> list[Candidate]:
        index = points_by_time(track_store, VEHICLE_CLASSES | {"pedestrian", "bicycle"})
        spans: list[tuple[float, float]] = []
        for t, members in index.items():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    (ta, pa), (tb, pb) = members[i], members[j]
                    if ta.cls not in VEHICLE_CLASSES and tb.cls not in VEHICLE_CLASSES:
                        continue
                    va, vb = velocity_mps(ta, t), velocity_mps(tb, t)
                    if va is None or vb is None or max(np.linalg.norm(va), np.linalg.norm(vb)) < MIN_SPEED_MPS:
                        continue
                    ttc = time_to_collision(
                        np.array([pa.x_m, pa.y_m]), va, np.array([pb.x_m, pb.y_m]), vb, _radius(ta.cls) + _radius(tb.cls)
                    )
                    if ttc is None or ttc <= 0.0 or ttc > TTC_MAX_SEC:
                        continue
                    if (_evasive(ta, t) or _evasive(tb, t)) and not _touches(ta, tb, t):
                        spans.append((t - PAD_BEFORE_SEC, t + PAD_AFTER_SEC))
        return [
            Candidate(max(0.0, s), e, self.label, conf=0.35, evidence={"needs_verification": True})
            for s, e in merge_intervals(spans, MERGE_GAP_SEC)
        ]
