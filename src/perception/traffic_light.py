"""Traffic-light colour state via HSV thresholding on a small ROI.

Deliberately not a learned model: the ROI is tiny (a few dozen px even at
4K), the signal head is a bright saturated LED cluster against a dark
housing, and classic HSV thresholding is the standard, fast, deterministic
approach for this -- keeps the fp16 GPU budget for the VLM verifier instead.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Hue ranges on OpenCV's 0-179 H scale. Red wraps around 0/180, so it needs
# two ranges.
_HUE_RANGES: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "red": [((0, 90, 90), (10, 255, 255)), ((170, 90, 90), (180, 255, 255))],
    "yellow": [((15, 90, 90), (35, 255, 255))],
    "green": [((40, 60, 90), (90, 255, 255))],
}
# A real lit lamp is a small bright blob inside the ROI, not the whole ROI
# (which also contains the dark housing) -- guards against classifying a
# stray reflection or the whole box as "lit".
MIN_LIT_FRACTION = 0.02
MAX_LIT_FRACTION = 0.6


@dataclass
class LightState:
    state: str  # "red" | "yellow" | "green" | "unknown"
    confidence: float  # in [0, 1]; 0.0 for "unknown"


def classify_roi(frame_bgr: np.ndarray, roi_norm: tuple[float, float, float, float]) -> LightState:
    """roi_norm: (x1, y1, x2, y2) normalized to this frame's own width/height."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = roi_norm
    px1, px2 = sorted((int(x1 * w), int(x2 * w)))
    py1, py2 = sorted((int(y1 * h), int(y2 * h)))
    px1, py1 = max(0, px1), max(0, py1)
    px2, py2 = min(w, px2), min(h, py2)
    if px2 - px1 < 2 or py2 - py1 < 2:
        return LightState("unknown", 0.0)

    patch = frame_bgr[py1:py2, px1:px2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    total = patch.shape[0] * patch.shape[1]

    scores: dict[str, float] = {}
    for name, ranges in _HUE_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
        scores[name] = int(cv2.countNonZero(mask)) / total

    best = max(scores, key=scores.get)
    frac = scores[best]
    if frac < MIN_LIT_FRACTION or frac > MAX_LIT_FRACTION:
        # too dim (nothing lit) or too saturated (probably not a lamp, e.g.
        # a red car or green foliage drifted into the ROI) -- report unknown
        # rather than guess.
        return LightState("unknown", 0.0)
    confidence = min(1.0, frac / (2 * MIN_LIT_FRACTION))
    return LightState(best, confidence)


def smooth_states(states: list[LightState], window: int = 3) -> list[LightState]:
    """Majority-vote smoothing over a sliding window -- a single misread
    frame (glare, motion blur) should not flip the reported colour."""
    if window <= 1 or len(states) <= window:
        return states
    out: list[LightState] = []
    half = window // 2
    for i in range(len(states)):
        lo, hi = max(0, i - half), min(len(states), i + half + 1)
        window_states = [s.state for s in states[lo:hi] if s.state != "unknown"]
        if not window_states:
            out.append(LightState("unknown", 0.0))
            continue
        counts = {s: window_states.count(s) for s in set(window_states)}
        best = max(counts, key=counts.get)
        conf = counts[best] / len(window_states)
        out.append(LightState(best, conf))
    return out
