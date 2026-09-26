"""congestion: standstill / crawling traffic across all lanes of one direction.

Needs world metres (speed_mps_at) -- a scene.json without a verified
homography will simply never trigger this rule (speed_mps_at returns None),
which is the safe failure mode.

A queue at a red light is not congestion: for lanes a traffic light controls
(traffic_lights[].lanes), a standstill only counts while that light is
verified green -- traffic that does not move on green is a jam.
"""
from __future__ import annotations

import numpy as np

from src.events.base import Candidate, flags_to_segments, nearest_in_time

CRAWL_SPEED_MPS = 1.5  # ~5.4 km/h
MIN_DURATION_SEC = 15.0  # a brief red-light stop is not congestion
DIRECTION_GROUP_COS = 0.7  # lanes within this cosine similarity share a "direction"
SAMPLE_STEP_SEC = 1.0
MAX_LIGHT_GAP_SEC = 1.5
# Two buses at a stop or one parked car are not a jam (C3896 @ 79-96 s: the
# far-road bus stop was flagged with 2 vehicles while traffic flowed freely).
MIN_SLOW_VEHICLES = 5


def _signal_allows_flow(lane_ids: set[str], t: float, scene, light_log) -> bool:
    """False while a light controlling any of these lanes is not verified green."""
    for tl in scene.traffic_lights:
        if lane_ids & set(tl.lanes):
            state = nearest_in_time((light_log or {}).get(tl.id, []), t, MAX_LIGHT_GAP_SEC)
            if state is None or state.state != "green":
                return False
    return True


def _group_lanes_by_direction(lanes) -> list[list]:
    groups: list[list] = []
    for lane in lanes:
        n = np.linalg.norm(lane.direction)
        if n < 1e-6:
            continue
        d = lane.direction / n
        placed = False
        for group in groups:
            gd = group[0].direction / np.linalg.norm(group[0].direction)
            if np.dot(d, gd) >= DIRECTION_GROUP_COS:
                group.append(lane)
                placed = True
                break
        if not placed:
            groups.append([lane])
    return groups


class CongestionRule:
    label = "congestion"

    def run(self, track_store, scene, light_log: dict | None = None) -> list[Candidate]:
        candidates: list[Candidate] = []
        groups = _group_lanes_by_direction(scene.lanes)
        if not groups:
            return candidates

        vehicle_tracks = track_store.vehicle_tracks()
        sample_times = track_store.sample_times()
        if not sample_times:
            return candidates
        t0, t1 = sample_times[0], sample_times[-1]
        n_steps = max(1, int((t1 - t0) / SAMPLE_STEP_SEC))
        eval_times = [t0 + i * SAMPLE_STEP_SEC for i in range(n_steps + 1)]

        for group in groups:
            lane_ids = {lane.id for lane in group}
            flags: list[tuple[float, bool]] = []
            for t in eval_times:
                speeds = []
                lanes_with_traffic = set()
                for track in vehicle_tracks:
                    pts = track.points_in_window(t, 0.5)
                    if not pts:
                        continue
                    p = pts[-1]
                    lane = scene.lane_at(p.x_norm, p.y_norm)
                    if lane is None or lane.id not in lane_ids:
                        continue
                    speed = track.speed_mps_at(t, lookback_sec=1.5)
                    if speed is None:
                        continue
                    speeds.append(speed)
                    lanes_with_traffic.add(lane.id)
                # require the jam to actually span multiple lanes of the group,
                # not just one stalled car in one lane
                is_jam = (
                    _signal_allows_flow(lane_ids, t, scene, light_log)
                    and len(speeds) >= MIN_SLOW_VEHICLES
                    and len(lanes_with_traffic) >= min(2, len(lane_ids))
                    and (sum(1 for s in speeds if s < CRAWL_SPEED_MPS) / len(speeds)) >= 0.8
                )
                flags.append((t, is_jam))
            for start, end in flags_to_segments(flags, min_duration_sec=MIN_DURATION_SEC, merge_gap_sec=3.0):
                candidates.append(
                    Candidate(
                        start=start,
                        end=end,
                        label=self.label,
                        conf=0.65,
                        evidence={"lane_ids": sorted(lane_ids)},
                    )
                )
        return candidates
