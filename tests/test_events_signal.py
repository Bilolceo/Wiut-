"""Synthetic-data tests for the signal-dependent event rules (red_light,
stop_line) and for stopped_vehicle's light-aware queue check -- all driven by
a hand-built scene + track store + fake light_log, no YOLO/video needed."""
from __future__ import annotations

import json

import pytest

from src.events.red_light import RedLightRule
from src.events.stop_line import StopLineRule
from src.events.stopped_vehicle import StoppedVehicleRule
from src.perception.traffic_light import LightState
from src.scene.loader import load_scene
from src.tracks.store import TrackPoint, TrackStore


@pytest.fixture()
def scene(tmp_path):
    scene_json = {
        "reference_video": "C3896.MP4",
        "crosswalks": [],
        "stop_lines": [{"id": "sl1", "line": [[0.3, 0.5], [0.5, 0.5]], "controlled_by": "tl1"}],
        "traffic_lights": [{"id": "tl1", "roi": [0.0, 0.0, 0.1, 0.1], "lanes": []}],
        "no_u_turn_zones": [],
        "intersection": {"polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]},
        "road_mask": {"polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]},
        "homography": {},
        "camera_alignment": {},
        "lanes": [],
    }
    p = tmp_path / "scene_signal_test.json"
    p.write_text(json.dumps(scene_json))
    return load_scene(p)


def _red_log(times):
    return {"tl1": [(t, LightState("red", 0.9)) for t in times]}


def _green_log(times):
    return {"tl1": [(t, LightState("green", 0.9)) for t in times]}


def test_red_light_flags_crossing_on_red(scene):
    store = TrackStore(video_id="t", fps=2.0)
    store.add_point(1, "car", TrackPoint(frame_idx=0, t_sec=0.0, x_norm=0.4, y_norm=0.6))
    store.add_point(1, "car", TrackPoint(frame_idx=1, t_sec=1.0, x_norm=0.4, y_norm=0.4))
    cands = RedLightRule().run(store, scene, _red_log([0.0, 1.0]))
    assert cands and cands[0].label == "red_light"


def test_red_light_ignores_crossing_on_green(scene):
    store = TrackStore(video_id="t", fps=2.0)
    store.add_point(2, "car", TrackPoint(frame_idx=0, t_sec=0.0, x_norm=0.4, y_norm=0.6))
    store.add_point(2, "car", TrackPoint(frame_idx=1, t_sec=1.0, x_norm=0.4, y_norm=0.4))
    assert RedLightRule().run(store, scene, _green_log([0.0, 1.0])) == []


def test_red_light_abstains_without_light_log(scene):
    store = TrackStore(video_id="t", fps=2.0)
    store.add_point(3, "car", TrackPoint(frame_idx=0, t_sec=0.0, x_norm=0.4, y_norm=0.6))
    store.add_point(3, "car", TrackPoint(frame_idx=1, t_sec=1.0, x_norm=0.4, y_norm=0.4))
    assert RedLightRule().run(store, scene, None) == []


def test_stop_line_flags_stopping_beyond_line_on_red(scene):
    store = TrackStore(video_id="t", fps=1.0)
    store.add_point(4, "car", TrackPoint(frame_idx=0, t_sec=0.0, x_norm=0.4, y_norm=0.6))  # approach side
    for t in [1.0, 2.0, 3.0, 4.0, 5.0]:  # crosses, then sits still beyond the line
        store.add_point(4, "car", TrackPoint(frame_idx=int(t), t_sec=t, x_norm=0.4, y_norm=0.4))
    cands = StopLineRule().run(store, scene, _red_log([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
    assert cands and cands[0].label == "stop_line"


def test_stop_line_ignores_vehicle_that_never_crosses(scene):
    store = TrackStore(video_id="t", fps=1.0)
    for t in range(6):
        store.add_point(5, "car", TrackPoint(frame_idx=t, t_sec=float(t), x_norm=0.4, y_norm=0.6))
    assert StopLineRule().run(store, scene, _red_log(list(range(6)))) == []


def test_stop_line_ignores_stop_beyond_line_on_green(scene):
    store = TrackStore(video_id="t", fps=1.0)
    store.add_point(6, "car", TrackPoint(frame_idx=0, t_sec=0.0, x_norm=0.4, y_norm=0.6))
    for t in [1.0, 2.0, 3.0, 4.0, 5.0]:
        store.add_point(6, "car", TrackPoint(frame_idx=int(t), t_sec=t, x_norm=0.4, y_norm=0.4))
    assert StopLineRule().run(store, scene, _green_log([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])) == []


def test_stopped_vehicle_still_flags_a_stop_at_the_line_when_light_is_verified_green(scene):
    # Sitting at the stop line while the light is provably green is not a
    # legitimate queue (more likely a breakdown / double-parked vehicle).
    store = TrackStore(video_id="t", fps=1.0)
    for i in range(13):
        store.add_point(
            7, "car", TrackPoint(frame_idx=i, t_sec=float(i), x_norm=0.4, y_norm=0.5, x_m=5.0, y_m=1.0)
        )
    cands = StoppedVehicleRule().run(store, scene, _green_log(list(range(13))))
    assert cands and cands[0].label == "stopped_vehicle"


def test_stopped_vehicle_still_falls_back_to_proximity_without_light_log(scene):
    store = TrackStore(video_id="t", fps=1.0)
    for i in range(13):
        store.add_point(
            8, "car", TrackPoint(frame_idx=i, t_sec=float(i), x_norm=0.4, y_norm=0.5, x_m=5.0, y_m=1.0)
        )
    assert StoppedVehicleRule().run(store, scene, None) == []
