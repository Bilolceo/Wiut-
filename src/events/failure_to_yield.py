"""failure_to_yield: a vehicle drives through a crossing while a pedestrian is on it.

At every sampled instant, a moving vehicle whose ground point is on a
crosswalk is checked against the pedestrians on that same crosswalk. Only
pedestrians in the vehicle's path count -- ahead of it (0..PATH_AHEAD_M along
its heading) and within PATH_HALF_WIDTH_M of its line of travel: a zebra
spans several lanes, and a car crossing the far lane, or passing behind a
pedestrian, is not failing to yield. A vehicle creeping below
MIN_SPEED_MPS is yielding. The pedestrian must be on the zebra itself (no
edge margin) and walking: on C3896 people waiting on the kerb at the zebra's
end (174 s, 183 s) were the main false positive.
"""
from __future__ import annotations

import math

from src.events.base import Candidate, merge_intervals
from src.scene.geometry import point_polygon_distance
from src.tracks.kinematics import RIDER_CLASSES, is_rider, point_at

MIN_SPEED_MPS = 2.0
MIN_SPEED_NORM_PER_SEC = 0.02  # fallback without metric calibration
PATH_BEHIND_M, PATH_AHEAD_M = 1.0, 10.0
PATH_HALF_WIDTH_M = 3.0
MAX_PED_DISTANCE_NORM = 0.08  # fallback without metric calibration
ZONE_MARGIN_NORM = 0.008  # vehicle ground point vs zebra
PED_MIN_SPEED_NORM_PER_SEC = 0.01  # a pedestrian standing still at the kerb is waiting, not crossing
EVENT_PAD_SEC = 0.5
MERGE_GAP_SEC = 1.0


def _moving(track, t: float) -> bool:
    speed = track.speed_mps_at(t, 1.0)
    if speed is not None:
        return speed >= MIN_SPEED_MPS
    window = track.points_in_window(t, 1.0)
    if len(window) < 2:
        return False
    a, b = window[0], window[-1]
    dt = b.t_sec - a.t_sec
    return dt > 0 and math.hypot(b.x_norm - a.x_norm, b.y_norm - a.y_norm) / dt >= MIN_SPEED_NORM_PER_SEC


def _walking(ped, t: float) -> bool:
    window = ped.points_in_window(t, 1.0)
    if len(window) < 2 or window[-1].t_sec - window[0].t_sec < 0.3:
        return False
    a, b = window[0], window[-1]
    return math.hypot(b.x_norm - a.x_norm, b.y_norm - a.y_norm) / (b.t_sec - a.t_sec) >= PED_MIN_SPEED_NORM_PER_SEC


def _in_path(vehicle, v, p) -> bool:
    """Pedestrian point p inside the corridor ahead of vehicle point v."""
    heading = vehicle.heading_at(v.t_sec, 1.0)
    if heading is not None and v.x_m is not None and p.x_m is not None:
        rx, ry = p.x_m - v.x_m, p.y_m - v.y_m
        along = rx * heading[0] + ry * heading[1]
        across = abs(rx * heading[1] - ry * heading[0])
        return -PATH_BEHIND_M <= along <= PATH_AHEAD_M and across <= PATH_HALF_WIDTH_M
    return math.hypot(v.x_norm - p.x_norm, v.y_norm - p.y_norm) <= MAX_PED_DISTANCE_NORM


class FailureToYieldRule:
    label = "failure_to_yield"

    def run(self, track_store, scene, light_log: dict | None = None) -> list[Candidate]:
        if not scene.crosswalks:
            return []
        riders = [t for t in track_store.tracks.values() if t.cls in RIDER_CLASSES]
        pedestrians = track_store.pedestrian_tracks()
        candidates: list[Candidate] = []
        for vehicle in track_store.vehicle_tracks():
            hits: list[tuple[float, float]] = []
            ped_ids: set[int] = set()
            for v in vehicle.points:
                zone = next(
                    (z for z in scene.crosswalks if point_polygon_distance((v.x_norm, v.y_norm), z.polygon) <= ZONE_MARGIN_NORM),
                    None,
                )
                if zone is None or not _moving(vehicle, v.t_sec):
                    continue
                for ped in pedestrians:
                    p = point_at(ped, v.t_sec)
                    if (
                        p is not None
                        and point_polygon_distance((p.x_norm, p.y_norm), zone.polygon) == 0.0
                        and _walking(ped, v.t_sec)
                        and _in_path(vehicle, v, p)
                        and not is_rider(p, v.t_sec, riders)
                    ):
                        hits.append((v.t_sec - EVENT_PAD_SEC, v.t_sec + EVENT_PAD_SEC))
                        ped_ids.add(ped.track_id)
                        break
            for start, end in merge_intervals(hits, MERGE_GAP_SEC):
                candidates.append(
                    Candidate(
                        max(0.0, start), end, self.label, conf=0.6,
                        evidence={"track_id": vehicle.track_id, "pedestrian_ids": sorted(ped_ids)},
                    )
                )
        return candidates
