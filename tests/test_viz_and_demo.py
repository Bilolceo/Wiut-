"""Annotated rendering and the demo API (the demo tests skip without the demo dependencies)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from conftest import add_track, line

from src.events.base import Candidate
from src.tracks.store import TrackStore
from src.viz.render import Annotator, involved_tracks
from src.viz.series import downsample_max


def test_involved_tracks_only_for_emitted_events():
    cands = [Candidate(1, 3, "jaywalking", 0.6, {"track_id": 7}), Candidate(10, 12, "near_miss", 0.35, {"track_ids": [1, 2]})]
    out = involved_tracks(cands, [[1.0, 3.0, "jaywalking"]])
    assert set(out) == {7} and out[7] == [(1, 3, "jaywalking")]


def test_annotator_draws_involved_road_user_in_red(make_scene):
    scene = make_scene()
    store = TrackStore(video_id="t", fps=10.0)
    add_track(store, 1, "car", line(0.0, 2.0, (0.5, 0.5), (0.5, 0.5)), w=0.1, h=0.1)
    ann = Annotator(scene, store, {}, [[0.0, 2.0, "red_light"]], [Candidate(0, 2, "red_light", 0.8, {"track_id": 1})], [], (200, 100))
    img = ann.draw(np.zeros((100, 200, 3), np.uint8), frame_idx=10, t=1.0)
    red = (img[..., 2] > 200) & (img[..., 1] < 80) & (img[..., 0] < 80)
    assert red[30:60, 80:120].any()


def test_downsample_keeps_the_peak():
    curve = [[i / 30, 0.9 if i == 17 else 0.05] for i in range(90)]
    out = downsample_max(curve, 2.0)
    assert len(out) == 6 and max(v for _, v in out) == 0.9


@pytest.fixture()
def client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    monkeypatch.setenv("DEMO_WORK_DIR", str(tmp_path))
    monkeypatch.setenv("DEMO_MAX_SEC", "2")
    import importlib

    import demo.server as server
    importlib.reload(server)
    from fastapi.testclient import TestClient

    return TestClient(server.app)


def _clip(path, seconds: float) -> str:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 36))
    for _ in range(int(seconds * 10)):
        w.write(np.zeros((36, 64, 3), np.uint8))
    w.release()
    return str(path)


def test_demo_rejects_non_video_and_too_long(client, tmp_path):
    bad = tmp_path / "x.mp4"
    bad.write_text("not a video")
    r = client.post("/api/jobs", files={"video": ("x.mp4", bad.open("rb"), "video/mp4")})
    assert r.status_code == 422 and "readable" in r.json()["detail"]
    long_clip = _clip(tmp_path / "long.mp4", 5.0)
    r = client.post("/api/jobs", files={"video": ("long.mp4", open(long_clip, "rb"), "video/mp4")})
    assert r.status_code == 422 and "accepts up to" in r.json()["detail"]


def test_demo_does_not_serve_uploads_or_unknown_jobs(client):
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get("/api/jobs/nope/files/input.mp4").status_code == 404
    assert client.get("/api/health").json()["ok"] is True
