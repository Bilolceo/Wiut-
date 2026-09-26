"""accident, near_miss, road_obstacle, fire_smoke: synthetic tracks / frames."""
from __future__ import annotations

import numpy as np
from conftest import add_track, line

from src.events.accident import AccidentRule
from src.events.fire_smoke import FireSmokeRule
from src.events.near_miss import NearMissRule
from src.events.road_obstacle import RoadObstacleRule
from src.perception.scene_monitor import Blob, FireSmokeMonitor, StaticObjectMonitor
from src.tracks.store import TrackStore


def _store():
    return TrackStore(video_id="t", fps=10.0)


def _crash_course():
    """Car A at 10 m/s slams into car B waiting 20 m ahead, both stop."""
    a = line(0.0, 2.0, (0.10, 0.50), (0.30, 0.50)) + line(2.1, 8.0, (0.301, 0.50), (0.301, 0.50))
    b = line(0.0, 8.0, (0.32, 0.50), (0.32, 0.50))
    return a, b


# -------------------------------------------------------------------- accident
def test_accident_flags_hard_stop_into_contact(make_scene):
    store = _store()
    a, b = _crash_course()
    add_track(store, 1, "car", a)
    add_track(store, 2, "car", b)
    cands = AccidentRule().run(store, make_scene())
    assert cands and cands[0].label == "accident" and cands[0].conf < 0.5
    assert 1.5 <= cands[0].start <= 3.0


def test_accident_ignores_gentle_stop_behind_queue(make_scene):
    """Braking at ~2 m/s^2 to stop 2 m behind a waiting car: a normal queue."""
    store = _store()
    pts, x, v = [], 0.10, 10.0
    for i in range(81):
        t = i / 10
        pts.append((t, x, 0.50))
        v = max(0.0, v - 2.0 * 0.1)
        x += v * 0.1 / 100
    add_track(store, 1, "car", pts)
    add_track(store, 2, "car", line(0.0, 8.0, (x + 0.02, 0.50), (x + 0.02, 0.50)))
    assert AccidentRule().run(store, make_scene()) == []


# ------------------------------------------------------------------- near_miss
def test_near_miss_flags_conflict_with_evasive_braking(make_scene):
    store = _store()
    # car A 10 m/s towards B; brakes hard at t=2 s and stops 5 m short of B
    pts, x, v = [], 0.10, 10.0
    for i in range(61):
        t = i / 10
        pts.append((t, x, 0.50))
        if t >= 2.0:
            v = max(0.0, v - 8.0 * 0.1)
        x += v * 0.1 / 100
    b_x = pts[-1][1] + 0.05
    add_track(store, 1, "car", pts)
    add_track(store, 2, "car", line(0.0, 6.0, (b_x, 0.50), (b_x, 0.50)))
    cands = NearMissRule().run(store, make_scene())
    assert cands and cands[0].label == "near_miss"


def test_near_miss_ignores_parallel_traffic(make_scene):
    store = _store()
    add_track(store, 1, "car", line(0.0, 6.0, (0.10, 0.50), (0.70, 0.50)))
    add_track(store, 2, "car", line(0.0, 6.0, (0.10, 0.54), (0.70, 0.54)))
    assert NearMissRule().run(store, make_scene()) == []


# --------------------------------------------------------------- road_obstacle
def test_road_obstacle_flags_untracked_static_blob(make_scene):
    store = _store()
    add_track(store, 1, "car", line(0.0, 10.0, (0.10, 0.10), (0.90, 0.10)))  # traffic elsewhere
    store.extras["static_objects"] = [Blob(t, (0.50, 0.60, 0.53, 0.62), 0.001) for t in np.arange(3.0, 10.0, 0.5)]
    cands = RoadObstacleRule().run(store, make_scene())
    assert len(cands) == 1 and cands[0].start == 0.0  # arrived 3 s before it was reported


def test_road_obstacle_ignores_blob_explained_by_a_stopped_car(make_scene):
    store = _store()
    add_track(store, 1, "car", line(0.0, 10.0, (0.515, 0.62), (0.515, 0.62)), w=0.04, h=0.04)
    store.extras["static_objects"] = [Blob(t, (0.50, 0.59, 0.53, 0.62), 0.001) for t in np.arange(3.0, 10.0, 0.5)]
    assert RoadObstacleRule().run(store, make_scene()) == []


def test_static_monitor_finds_object_that_appears_and_stays():
    rng = np.random.default_rng(0)
    base = rng.integers(60, 70, size=(270, 480, 3), dtype=np.uint8)
    mon = StaticObjectMonitor(warmup_sec=2.0)
    for i in range(40):  # 20 s at 2 fps; box appears at t=6 s and stays
        frame = base.copy()
        if i * 0.5 >= 6.0:
            frame[120:150, 200:240] = 230
        mon.update(frame, i * 0.5)
    blobs = mon.result(lambda x, y: (x, y))
    assert blobs and min(b.t_sec for b in blobs) >= 6.0 + 3.0 - 1e-6
    x1, y1, x2, y2 = blobs[-1].box
    assert x1 <= 200 / 480 <= x2 and y1 <= 130 / 270 <= y2


# ------------------------------------------------------------------ fire_smoke
def test_fire_monitor_and_rule_flag_persistent_flames(make_scene):
    frame = np.full((270, 480, 3), 50, dtype=np.uint8)
    frame[100:140, 200:260] = (0, 140, 255)  # bright saturated orange, BGR
    mon = FireSmokeMonitor()
    for i in range(12):
        mon.update(frame, i * 0.5)
    store = _store()
    add_track(store, 1, "car", line(0.0, 6.0, (0.10, 0.10), (0.20, 0.10)))
    store.extras["fire_smoke"] = mon.result(lambda x, y: (x, y))
    cands = FireSmokeRule().run(store, make_scene())
    assert cands and cands[0].label == "fire_smoke"


def test_fire_monitor_ignores_small_tail_lights():
    frame = np.full((270, 480, 3), 50, dtype=np.uint8)
    frame[100:103, 200:204] = (0, 140, 255)  # a tail-light-sized blob
    mon = FireSmokeMonitor()
    for i in range(12):
        mon.update(frame, i * 0.5)
    assert mon.result(lambda x, y: (x, y))["fire"] == []
