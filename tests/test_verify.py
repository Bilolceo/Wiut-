"""VLM verifier with a fake backend: the 4.3 GB model is never loaded here."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.events.base import Candidate
from src.verify.vlm import ClipSpec, Verifier, evidence_crop, read_clip, verify_candidate


class FakeVLM:
    def __init__(self, p: float) -> None:
        self.p, self.questions = p, []

    def p_yes(self, images, question):
        assert images and all(im.ndim == 3 for im in images)
        self.questions.append(question)
        return self.p


@pytest.fixture()
def clip_path(tmp_path):
    path = str(tmp_path / "clip.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 180))
    for i in range(60):
        frame = np.full((180, 320, 3), i * 4, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_verify_candidate_accepts_and_rejects_by_p_yes():
    frames = [np.zeros((10, 10, 3), np.uint8)]
    assert verify_candidate(frames, Candidate(0, 1, "accident", 0.3), FakeVLM(0.8)) == (True, 0.8)
    assert verify_candidate(frames, Candidate(0, 1, "accident", 0.3), FakeVLM(0.2)) == (False, 0.2)


def test_verify_candidate_rejects_unknown_label_and_empty_clip():
    frames = [np.zeros((10, 10, 3), np.uint8)]
    assert verify_candidate(frames, Candidate(0, 1, "jaywalking", 0.9), FakeVLM(0.99)) == (False, 0.0)
    assert verify_candidate([], Candidate(0, 1, "accident", 0.3), FakeVLM(0.99)) == (False, 0.0)


def test_read_clip_returns_resized_frames(clip_path):
    frames = read_clip(clip_path, 2.0, 3.0, ClipSpec(n_frames=4, width=160))
    assert len(frames) == 4 and all(f.shape == (90, 160, 3) for f in frames)
    cropped = read_clip(clip_path, 2.0, 3.0, ClipSpec(n_frames=2, width=100), crop=(0.5, 0.5, 1.0, 1.0))
    assert len(cropped) == 2 and cropped[0].shape[1] == 100


def test_verifier_respects_the_call_budget_and_passes_other_classes(clip_path):
    vlm = FakeVLM(0.9)
    cands = [Candidate(1.0, 2.0, "near_miss", 0.35), Candidate(2.0, 3.0, "accident", 0.3),
             Candidate(3.0, 4.0, "near_miss", 0.35), Candidate(0.5, 1.5, "red_light", 0.75)]
    out = Verifier(vlm, {"accident", "near_miss"}, max_calls=2).filter(cands, clip_path)
    assert len(vlm.questions) == 2
    assert [c.label for c in out].count("red_light") == 1  # not a verified class: untouched
    verified = [c for c in out if c.label != "red_light"]
    assert len(verified) == 2 and all(c.conf == 0.9 and c.evidence["vlm_p_yes"] == 0.9 for c in verified)


def test_verifier_drops_rejected_candidates(clip_path):
    out = Verifier(FakeVLM(0.1), {"accident"}).filter([Candidate(1.0, 2.0, "accident", 0.3)], clip_path)
    assert out == []


def test_evidence_crop_stays_inside_the_frame():
    from conftest import add_track, line
    from src.tracks.store import TrackStore

    store = TrackStore(video_id="t", fps=10.0)
    add_track(store, 1, "car", line(0.0, 2.0, (0.95, 0.95), (0.99, 0.99)))
    crop = evidence_crop(Candidate(0.0, 2.0, "accident", 0.3, {"track_ids": [1]}), store, lambda x, y: (x, y), ClipSpec())
    x1, y1, x2, y2 = crop
    assert 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0 and x2 - x1 >= 0.35


def test_detect_verify_drops_verified_classes_when_weights_are_missing(monkeypatch):
    import src.detect as detect

    monkeypatch.setattr(detect, "get_vlm_backend", lambda model_dir: None)
    cands = [Candidate(1.0, 2.0, "accident", 0.3), Candidate(1.0, 2.0, "red_light", 0.75)]
    cfg = {"verifier": {"enabled": True, "model_dir": "nowhere", "labels": ["accident"]}}
    out = detect.verify(cands, "x.mp4", None, None, None, cfg, deadline=0.0)
    assert [c.label for c in out] == ["red_light"]


def test_verifier_budget_goes_to_high_priority_classes_first(clip_path):
    """Many road_obstacle candidates (rule conf 0.4) must not starve accident (0.3) of VLM calls."""
    vlm = FakeVLM(0.9)
    cands = [Candidate(float(i), i + 1.0, "road_obstacle", 0.4) for i in range(3)] + [Candidate(4.0, 5.0, "accident", 0.3)]
    out = Verifier(vlm, ["accident", "near_miss", "road_obstacle"], max_calls=1).filter(cands, clip_path)
    assert [c.label for c in out] == ["accident"]
