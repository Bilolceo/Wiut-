"""io.decode tests against a tiny synthetic video (no sample data needed)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.io.decode import read_meta, sample_frames


@pytest.fixture()
def tiny_video(tmp_path):
    path = tmp_path / "tiny.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, 10.0, (64, 48))
    for i in range(30):  # 3.0s at 10 fps
        frame = np.full((48, 64, 3), (i * 8) % 255, dtype=np.uint8)
        vw.write(frame)
    vw.release()
    return path


def test_read_meta(tiny_video):
    meta = read_meta(tiny_video)
    assert meta.width == 64
    assert meta.height == 48
    assert meta.fps == pytest.approx(10.0, abs=0.5)
    assert meta.n_frames == pytest.approx(30, abs=1)


def test_sample_frames_respects_target_fps(tiny_video):
    # native 10 fps, target 5 fps -> every 2nd frame -> ~15 frames
    frames = list(sample_frames(tiny_video, target_fps=5.0))
    assert 12 <= len(frames) <= 16
    # timestamps must be monotonically increasing
    ts = [f.t_sec for f in frames]
    assert ts == sorted(ts)
    assert frames[0].bgr.shape == (48, 64, 3)


def test_sample_frames_never_exceeds_native_fps(tiny_video):
    # asking for MORE than native fps should just return every frame, not crash
    frames = list(sample_frames(tiny_video, target_fps=100.0))
    assert len(frames) == pytest.approx(30, abs=1)
