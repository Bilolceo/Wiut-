"""Synthetic-data tests for the "easy" event rules: stopped_vehicle, wrong_way,
congestion. No YOLO/video needed -- these exercise the rule logic in
isolation against a hand-built scene + track store, per CLAUDE.md's
"work module by module, test after each module" workflow.
"""
from __future__ import annotations

import json

import pytest

from src.scene.loader import load_scene
from src.tracks.store import TrackPoint, TrackStore
from src.events.stopped_vehicle import StoppedVehicleRule
from src.events.wrong_way import WrongWayRule
from src.events.congestion import CongestionRule


@pytest.fixture()
def scene(tmp_path):
    scene_json = {
        "reference_video": "C3896.MP4",
        "crosswalks": [],
        "stop_lines": [{"id": "sl1", "line": [[0.3, 0.4], [0.5, 0.33]], "controlled_by": "tl1"}],
        "traffic_lights": [],
        "no_u_turn_zones": [],
        "intersection": {"polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]},
        "road_mask": {"polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]},
        "homography": {
            "image_points": [[0.0, 0.0], [0.0, 0.3], [0.6, 0.3], [0.6, 0.0]],
            "world_points_m": [[0, 0], [0, 10.5], [21, 10.5], [21, 0]],
        },
        "camera_alignment": {},
        "lanes": [
            {
                "id": "lane_fwd",
                "polygon": [[0.0, 0.0], [0.6, 0.0], [0.6, 0.3], [0.0, 0.3]],
                "direction": [1.0, 0.0],
                "turn_allowed": ["straight"],
            },
        ],
    }
    p = tmp_path / "scene_test.json"
    p.write_text(json.dumps(scene_json))
    return load_scene(p)


def test_stopped_vehicle_flags_a_stationary_car(scene):
    store = TrackStore(video_id="test", fps=10.0)
    for i in range(13):
        store.add_point(
            1, "car", TrackPoint(frame_idx=i, t_sec=float(i), x_norm=0.5, y_norm=0.9, x_m=15.0, y_m=25.0)
        )
    cands = StoppedVehicleRule().run(store, scene)
    assert cands and cands[0].label == "stopped_vehicle"


def test_stopped_vehicle_ignores_queue_at_stop_line(scene):
    store = TrackStore(video_id="test", fps=10.0)
    for i in range(13):
        store.add_point(
            2, "car", TrackPoint(frame_idx=i, t_sec=float(i), x_norm=0.35, y_norm=0.38, x_m=5.0, y_m=1.0)
        )
    assert StoppedVehicleRule().run(store, scene) == []


def test_wrong_way_flags_car_against_lane_direction(scene):
    store = TrackStore(video_id="test", fps=10.0)
    for i in range(10):
        x = 0.5 - 0.03 * i  # lane direction is +x -> this is wrong-way
        store.add_point(3, "car", TrackPoint(frame_idx=i, t_sec=float(i), x_norm=x, y_norm=0.15))
    cands = WrongWayRule().run(store, scene)
    assert cands and cands[0].label == "wrong_way"


def test_wrong_way_ignores_car_with_traffic(scene):
    store = TrackStore(video_id="test", fps=10.0)
    for i in range(10):
        x = 0.1 + 0.03 * i  # matches lane direction
        store.add_point(4, "car", TrackPoint(frame_idx=i, t_sec=float(i), x_norm=x, y_norm=0.15))
    assert WrongWayRule().run(store, scene) == []


def test_congestion_flags_multi_lane_crawl(scene):
    store = TrackStore(video_id="test", fps=10.0)
    for i in range(21):
        t = float(i)
        for k in range(5):  # a jam needs MIN_SLOW_VEHICLES crawling vehicles, not two
            store.add_point(
                5 + k, "car",
                TrackPoint(frame_idx=i, t_sec=t, x_norm=0.1 + 0.1 * k, y_norm=0.15, x_m=1.0 + 7 * k + 0.05 * i, y_m=5.0),
            )
    cands = CongestionRule().run(store, scene)
    assert cands and cands[0].label == "congestion"


def test_congestion_ignores_single_car_crawl(scene):
    """One stalled car in one lane is stopped_vehicle territory, not congestion."""
    store = TrackStore(video_id="test", fps=10.0)
    for i in range(21):
        store.add_point(
            7, "car", TrackPoint(frame_idx=i, t_sec=float(i), x_norm=0.2, y_norm=0.15, x_m=1.0 + 0.05 * i, y_m=5.0)
        )
    assert CongestionRule().run(store, scene) == []
