"""perception.pipeline.build_track_store, exercised end-to-end with a fake
detector (no ultralytics / GPU needed) against a tiny synthetic video."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.perception.detector import Detection
from src.perception.pipeline import build_track_store
from src.scene.loader import Scene


class FakePerception:
    """Returns one deterministic detection per frame -- a stand-in for
    Perception.track_frame so this test never imports ultralytics."""

    def __init__(self):
        self.calls = 0

    def track_frame(self, frame_bgr):
        self.calls += 1
        return [Detection(track_id=1, cls="car", cx_norm=0.5, cy_norm=0.5, w_norm=0.1, h_norm=0.1, conf=0.9)]


@pytest.fixture()
def tiny_video(tmp_path):
    path = tmp_path / "tiny.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, 10.0, (100, 40))
    for i in range(10):
        frame = np.full((40, 100, 3), 10, dtype=np.uint8)  # dark housing
        # the ROI is (0.4, 0.4, 0.6, 0.6) -> px (40,16)-(60,24); a small lit
        # lamp inside it, dark housing around -- mimics a real signal head
        # where the lamp is a fraction of the ROI box, not the whole thing.
        frame[18:22, 47:53] = (0, 0, 255)
        vw.write(frame)
    vw.release()
    return path


@pytest.fixture()
def identity_scene():
    return Scene(
        reference_video="tiny.mp4",
        crosswalks=[],
        stop_lines=[],
        traffic_lights=[_tl("tl1", (0.4, 0.4, 0.6, 0.6))],
        lanes=[],
        no_u_turn_zones=[],
        intersection=None,
        road_mask=None,
        homography={},  # no world-metres calibration -> x_m/y_m stay None
        camera_alignment={},  # -> identity alignment, ref size = video's own
        raw={},
    )


def _tl(id_, roi):
    from src.scene.loader import TrafficLight

    return TrafficLight(id=id_, roi=roi, lanes=[])


def test_build_track_store_populates_one_track_per_detection(tiny_video, identity_scene):
    fake = FakePerception()
    store, light_log, meta = build_track_store(tiny_video, identity_scene, fake, target_fps=10.0)

    assert fake.calls == 10
    assert 1 in store.tracks
    track = store.tracks[1]
    assert len(track) == 10
    assert track.points[0].x_norm == pytest.approx(0.5, abs=1e-6)
    assert track.points[0].y_norm == pytest.approx(0.5, abs=1e-6)
    # no homography configured -> world coords must stay unset, never guessed
    assert track.points[0].x_m is None


def test_build_track_store_classifies_traffic_light_every_frame(tiny_video, identity_scene):
    fake = FakePerception()
    _, light_log, _ = build_track_store(tiny_video, identity_scene, fake, target_fps=10.0)

    assert "tl1" in light_log
    states = light_log["tl1"]
    assert len(states) == 10
    # every frame has a solid red patch inside the ROI -> majority-smoothed to red
    assert all(state.state == "red" for _, state in states)
