"""Part B: TTC maths and the causal RiskEstimator with a fake detector."""
from __future__ import annotations

import numpy as np

from src.perception.detector import Detection
from src.risk.estimator import CausalRiskEstimator
from src.tracks.kinematics import time_to_collision

FPS = 30.0


def test_ttc_head_on():
    ttc = time_to_collision(np.array([0.0, 0.0]), np.array([10.0, 0.0]), np.array([50.0, 0.0]), np.array([-10.0, 0.0]), 4.0)
    assert abs(ttc - (50.0 - 4.0) / 20.0) < 1e-6


def test_ttc_none_when_moving_apart_or_passing_wide():
    assert time_to_collision(np.array([0.0, 0.0]), np.array([-5.0, 0.0]), np.array([10.0, 0.0]), np.array([5.0, 0.0]), 4.0) is None
    assert time_to_collision(np.array([0.0, 0.0]), np.array([10.0, 0.0]), np.array([30.0, 20.0]), np.array([-10.0, 0.0]), 4.0) is None


class FakePerception:
    """Two cars on the near approach; `closing` decides whether they converge."""

    def __init__(self, closing: bool) -> None:
        self.closing, self.calls = closing, 0

    def track_frame(self, frame):
        t = self.calls * 3 / FPS  # called on every 3rd frame
        self.calls += 1
        s = 0.06 * t  # normalized units per second (~3 m/s per car near the stop line)
        if self.closing:
            xa, xb = 0.20 + s, 0.44 - s
        else:
            xa, xb = 0.24 - s, 0.40 + s
        return [
            Detection(1, "car", xa, 0.455, 0.04, 0.04, 0.9),
            Detection(2, "car", xb, 0.435, 0.04, 0.04, 0.9),
        ]


def _run(closing: bool, seconds: float = 3.5) -> list[float]:
    est = CausalRiskEstimator(perception_factory=lambda: FakePerception(closing))
    est.reset({"video_id": "synthetic.mp4", "fps": FPS, "width": 64, "height": 36, "n_frames": int(seconds * FPS)})
    frame = np.zeros((36, 64, 3), dtype=np.uint8)
    return [est.step(frame, i / FPS) for i in range(int(seconds * FPS))]


def test_risk_rises_for_converging_cars():
    scores = _run(closing=True)
    assert max(scores) > 0.5
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_risk_stays_low_for_diverging_cars():
    scores = _run(closing=False)
    assert max(scores) < 0.2


def test_risk_is_causal():
    """The score at frame k must not depend on frames after k."""
    full = _run(closing=True, seconds=3.5)
    short = _run(closing=True, seconds=2.0)
    assert full[: len(short)] == short


def test_reset_clears_state():
    est = CausalRiskEstimator(perception_factory=lambda: FakePerception(True))
    est.reset({"video_id": "a.mp4", "fps": FPS, "width": 64, "height": 36, "n_frames": 90})
    frame = np.zeros((36, 64, 3), dtype=np.uint8)
    for i in range(90):
        est.step(frame, i / FPS)
    est.reset({"video_id": "b.mp4", "fps": FPS, "width": 64, "height": 36, "n_frames": 90})
    assert est.score == 0.0 and not est.history.tracks and est.perception is None


def test_ttc_ignores_pairs_already_close_but_not_closing():
    """Two cars level in adjacent lanes (2 m apart, same speed): not a conflict. This made Part B fire constantly."""
    p1, p2 = np.array([0.0, 0.0]), np.array([0.0, 2.0])
    v = np.array([8.0, 0.0])
    assert time_to_collision(p1, v, p2, v, 4.0) is None
    assert time_to_collision(p1, v, p2, v - np.array([0.0, 1.0]), 4.0) == 0.0  # overlapping and closing


def test_worst_conflict_ignores_jitter_level_closing():
    from src.config import load_yaml
    from src.risk.estimator import worst_conflict

    cfg = load_yaml("risk.yaml")
    states = [("car", np.array([0.0, 0.0]), np.array([5.0, 0.0])), ("car", np.array([6.0, 0.0]), np.array([4.5, 0.0]))]
    assert worst_conflict(states, cfg) == (None, 0.0)  # closing at 0.5 m/s: noise, not a conflict


def test_risk_formula_is_one_half_at_ttc_half_and_low_without_conflict():
    """Regression: the old formula could never exceed ~0.35 from TTC alone."""
    from src.config import load_yaml
    from src.risk.estimator import risk_from_conflict

    cfg = load_yaml("risk.yaml")
    assert abs(risk_from_conflict(cfg["ttc_half_sec"], 0.0, cfg) - 0.5) < 1e-9
    assert risk_from_conflict(0.0, 0.0, cfg) > 0.9
    assert risk_from_conflict(None, 0.0, cfg) < 0.06


def test_single_step_spike_does_not_raise_the_score():
    from src.risk import estimator as E

    est = CausalRiskEstimator(perception_factory=lambda: FakePerception(False))
    est.reset({"video_id": "x.mp4", "fps": FPS, "width": 64, "height": 36, "n_frames": 30})
    calls = iter([(0.1, 0.0)] + [(None, 0.0)] * 20)  # one step with TTC 0.1 s, then nothing
    orig = E.worst_conflict
    E.worst_conflict = lambda states, cfg: next(calls)
    try:
        scores = [est.step(np.zeros((36, 64, 3), np.uint8), i / FPS) for i in range(60)]
    finally:
        E.worst_conflict = orig
    assert max(scores) < 0.2


def test_calibrated_homography_is_metric():
    """scripts/calibrate_camera.py output: stop line ~14 m (4 lanes), same zebra equally wide left/right of the island."""
    from src.detect import get_scene

    s = get_scene()
    d = lambda a, b: float(np.linalg.norm(np.subtract(s.to_metres(*a), s.to_metres(*b))))
    assert 12.5 < d((0.150, 0.487), (0.485, 0.423)) < 15.5
    left, right = d((0.40, 0.5185), (0.40, 0.5570)), d((0.80, 0.4567), (0.80, 0.4847))
    assert abs(left - right) / left < 0.15
