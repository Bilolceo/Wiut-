"""Shared synthetic scene + track helpers for the rule tests (no video, no YOLO)."""
from __future__ import annotations

import json

import pytest

from src.scene.loader import load_scene
from src.tracks.store import TrackPoint, TrackStore

# 1 normalized unit = 100 m in both axes: easy mental arithmetic in tests.
BASE_SCENE = {
    "reference_video": "test.MP4",
    "crosswalks": [{"id": "cw", "polygon": [[0.40, 0.40], [0.60, 0.40], [0.60, 0.46], [0.40, 0.46]]}],
    "stop_lines": [],
    "traffic_lights": [],
    "refuges": [{"id": "island", "polygon": [[0.80, 0.80], [0.90, 0.80], [0.90, 0.90], [0.80, 0.90]]}],
    "solid_lines": [{"id": "solid", "line": [[0.0, 0.50], [1.0, 0.50]]}],
    "no_u_turn_zones": [{"id": "nou", "polygon": [[0.0, 0.0], [0.3, 0.0], [0.3, 0.3], [0.0, 0.3]]}],
    "intersection": None,
    "road_mask": {"polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]},
    "homography": {
        "image_points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        "world_points_m": [[0, 0], [100, 0], [100, 100], [0, 100]],
    },
    "camera_alignment": {},
    "lanes": [],
}


@pytest.fixture()
def make_scene(tmp_path):
    def _make(**overrides):
        raw = {**BASE_SCENE, **overrides}
        p = tmp_path / "scene.json"
        p.write_text(json.dumps(raw))
        return load_scene(p)

    return _make


def add_track(store: TrackStore, tid: int, cls: str, points, fps: float = 10.0, conf: float = 0.8, w=0.02, h=0.05):
    """points: [(t_sec, x_norm, y_norm), ...]; metres follow the 100 m/unit test homography."""
    for t, x, y in points:
        store.add_point(
            tid, cls,
            TrackPoint(frame_idx=round(t * fps), t_sec=t, x_norm=x, y_norm=y, x_m=x * 100, y_m=y * 100,
                       w_norm=w, h_norm=h, conf=conf),
        )


def line(t0: float, t1: float, p0, p1, fps: float = 10.0):
    """Constant-velocity samples from p0 at t0 to p1 at t1."""
    n = round((t1 - t0) * fps)
    return [(t0 + i / fps, p0[0] + (p1[0] - p0[0]) * i / n, p0[1] + (p1[1] - p0[1]) * i / n) for i in range(n + 1)]
