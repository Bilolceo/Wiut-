"""Post-processing contract, traffic-light layout check, camera alignment, and solution.py safety."""
from __future__ import annotations

import cv2
import numpy as np

import solution
from src.events.base import Candidate
from src.perception.traffic_light import classify_roi
from src.post.postprocess import finalize
from src.scene.align import estimate_alignment

TH = {
    "merge_gap_sec": 1.0,
    "classes": {
        "jaywalking": {"enabled": True, "min_conf": 0.5, "min_duration_sec": 1.0},
        "red_light": {"enabled": True, "min_conf": 0.5, "min_duration_sec": 0.5},
        "accident": {"enabled": False, "min_conf": 0.3, "min_duration_sec": 1.0},
    },
}


# ---------------------------------------------------------------- postprocess
def test_finalize_merges_same_class_and_never_overlaps():
    cands = [Candidate(1.0, 4.0, "jaywalking", 0.6), Candidate(3.0, 6.0, "jaywalking", 0.6), Candidate(6.5, 9.0, "jaywalking", 0.6)]
    events = finalize(cands, 60.0, TH)
    assert events == [[1.0, 9.0, "jaywalking"]]


def test_finalize_drops_disabled_weak_short_and_clips_to_duration():
    cands = [
        Candidate(1.0, 5.0, "accident", 0.9),  # disabled class
        Candidate(1.0, 5.0, "red_light", 0.4),  # below min_conf
        Candidate(10.0, 10.2, "red_light", 0.9),  # shorter than min_duration
        Candidate(-1.0, 3.0, "jaywalking", 0.9),  # starts before 0
        Candidate(58.0, 70.0, "red_light", 0.9),  # ends after duration
    ]
    events = finalize(cands, 60.0, TH)
    assert events == [[0.0, 3.0, "jaywalking"], [58.0, 60.0, "red_light"]]
    for s, e, _ in events:
        assert 0.0 <= s < e <= 60.0


# ------------------------------------------------------------- traffic light
def test_vertical_layout_rejects_red_blob_in_bottom_half():
    frame = np.full((60, 20, 3), 10, dtype=np.uint8)
    frame[45:52, 7:13] = (0, 0, 255)  # red where the green lamp sits
    assert classify_roi(frame, (0, 0, 1, 1), layout="vertical").state == "unknown"
    frame2 = np.full((60, 20, 3), 10, dtype=np.uint8)
    frame2[8:15, 7:13] = (0, 0, 255)
    assert classify_roi(frame2, (0, 0, 1, 1), layout="vertical").state == "red"


def test_tiny_daylight_lamp_is_still_read():
    """On the samples the lit lamp is ~0.2-1 % of the ROI (the old 2 % floor missed it)."""
    frame = np.full((100, 50, 3), 120, dtype=np.uint8)  # grey housing / visors in daylight
    frame[20:23, 20:24] = (0, 0, 230)  # 12 px lit
    assert classify_roi(frame, (0, 0, 1, 1), layout="vertical").state == "red"


# ------------------------------------------------------------------ alignment
def test_alignment_recovers_a_known_shift():
    rng = np.random.default_rng(0)
    ref = cv2.GaussianBlur(rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8), (5, 5), 0)
    shifted = np.roll(ref, shift=(12, -30), axis=(0, 1))  # this video = reference moved by (-30, +12)
    H = estimate_alignment(shifted, ref, (1920, 1080))
    assert H is not None
    assert abs(H[0, 2] - 30) < 2 and abs(H[1, 2] + 12) < 2


# ---------------------------------------------------------------- solution.py
def test_detect_events_never_raises_on_bad_input():
    assert solution.detect_events("/nonexistent/video.mp4") == []


def test_solution_exposes_the_harness_interface():
    assert len(solution.CLASSES) == 14 and len(set(solution.CLASSES)) == 14
    assert callable(solution.detect_events) and hasattr(solution.RiskEstimator, "step")
