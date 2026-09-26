"""perception.traffic_light tests against synthetic solid-colour patches."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.perception.traffic_light import LightState, classify_roi, smooth_states


def _frame_with_patch(bgr_color, size=(60, 20), frame_size=(100, 40)):
    """A dark frame with one small solid-colour rectangle centred in it,
    mimicking a lit lamp against a dark signal housing."""
    w, h = frame_size
    frame = np.full((h, w, 3), 10, dtype=np.uint8)  # near-black housing
    pw, ph = size
    x0, y0 = (w - pw) // 2, (h - ph) // 2
    frame[y0 : y0 + ph, x0 : x0 + pw] = bgr_color
    return frame


@pytest.mark.parametrize(
    "name,bgr",
    [("red", (0, 0, 255)), ("yellow", (0, 255, 255)), ("green", (0, 200, 0))],
)
def test_classify_roi_identifies_lit_colour(name, bgr):
    frame = _frame_with_patch(bgr)
    result = classify_roi(frame, (0.0, 0.0, 1.0, 1.0))
    assert result.state == name
    assert result.confidence > 0.0


def test_classify_roi_reports_unknown_when_dark():
    frame = np.full((40, 100, 3), 10, dtype=np.uint8)  # nothing lit
    result = classify_roi(frame, (0.0, 0.0, 1.0, 1.0))
    assert result.state == "unknown"
    assert result.confidence == 0.0


def test_classify_roi_reports_unknown_for_degenerate_roi():
    frame = np.zeros((40, 100, 3), dtype=np.uint8)
    result = classify_roi(frame, (0.5, 0.5, 0.5, 0.5))  # zero-area ROI
    assert result.state == "unknown"


def test_smooth_states_majority_votes_out_a_single_misread():
    states = [
        LightState("red", 0.9),
        LightState("red", 0.9),
        LightState("green", 0.3),  # one bad frame (glare, motion blur, ...)
        LightState("red", 0.9),
        LightState("red", 0.9),
    ]
    smoothed = smooth_states(states, window=3)
    assert smoothed[2].state == "red"
