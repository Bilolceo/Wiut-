"""Traffic-light colour state via HSV thresholding on a small ROI.

Deliberately not a learned model: the ROI is tiny (a few dozen px even at
4K), the signal head is a bright saturated LED cluster against a dark
housing, and classic HSV thresholding is the standard, fast, deterministic
approach for this -- keeps the fp16 GPU budget for the VLM verifier instead.

Calibrated on the sample videos (docs/PLAN.md section 4 EDA note): in
daylight the lit lamp of the median signal covers only ~0.2-1 % of its ROI
(the rest is housing and sun visors) and a few dozen pixels at 4K, so the
decision uses an absolute lit-pixel count plus dominance over the other
colours, not a fraction of the ROI.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Hue ranges on OpenCV's 0-179 H scale. Red wraps around 0/180, so it needs
# two ranges.
_HUE_RANGES: dict[str, list[tuple[int, int]]] = {
    "red": [(0, 10), (170, 180)],
    "yellow": [(15, 35)],
    "green": [(40, 95)],
}
MIN_SAT = 80
MIN_VAL = 80
MIN_LIT_PIXELS = 4  # absolute floor, in native pixels of the ROI
MIN_LIT_FRACTION = 0.0005  # relative floor, so a huge ROI needs a proportionally bigger lamp
MAX_LIT_FRACTION = 0.6  # more than this is not a lamp (red car / foliage filling the ROI)
DOMINANCE = 2.0  # winning colour must have >= this many times the runner-up's pixels
FULL_CONF_PIXELS = 30  # lit-pixel count at which confidence saturates


@dataclass
class LightState:
    state: str  # "red" | "yellow" | "green" | "unknown"
    confidence: float  # in [0, 1]; 0.0 for "unknown"


UNKNOWN = LightState("unknown", 0.0)


def _position_ok(color: str, centroid_y: float) -> bool:
    """Vertical signal head: red lamp in the top half, green in the bottom half."""
    if color == "red":
        return centroid_y < 0.5
    if color == "green":
        return centroid_y > 0.5
    return True


def classify_roi(
    frame_bgr: np.ndarray,
    roi_norm: tuple[float, float, float, float],
    layout: str | None = None,
) -> LightState:
    """roi_norm: (x1, y1, x2, y2) normalized to this frame's own width/height.

    layout="vertical" additionally requires the lit colour to sit where that
    lamp is on a vertical head (red top, green bottom), which rejects a red
    car or a reflection passing through the ROI.
    """
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = roi_norm
    px1, px2 = sorted((int(x1 * w), int(x2 * w)))
    py1, py2 = sorted((int(y1 * h), int(y2 * h)))
    px1, py1 = max(0, px1), max(0, py1)
    px2, py2 = min(w, px2), min(h, py2)
    if px2 - px1 < 2 or py2 - py1 < 2:
        return UNKNOWN

    hsv = cv2.cvtColor(frame_bgr[py1:py2, px1:px2], cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lit = (sat >= MIN_SAT) & (val >= MIN_VAL)
    total = hue.size

    counts: dict[str, int] = {}
    centroids: dict[str, float] = {}
    for name, ranges in _HUE_RANGES.items():
        in_hue = np.zeros(hue.shape, dtype=bool)
        for lo, hi in ranges:
            in_hue |= (hue >= lo) & (hue <= hi)
        mask = lit & in_hue
        counts[name] = int(mask.sum())
        if counts[name]:
            centroids[name] = float(np.nonzero(mask)[0].mean()) / hue.shape[0]

    best = max(counts, key=counts.get)
    n = counts[best]
    runner_up = max(c for k, c in counts.items() if k != best)
    if n < max(MIN_LIT_PIXELS, MIN_LIT_FRACTION * total) or n > MAX_LIT_FRACTION * total:
        return UNKNOWN
    if n < DOMINANCE * runner_up:
        return UNKNOWN  # two colours lit at once: glare or a transition frame -- abstain
    if layout == "vertical" and not _position_ok(best, centroids[best]):
        return UNKNOWN
    return LightState(best, min(1.0, n / FULL_CONF_PIXELS))


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
            out.append(UNKNOWN)
            continue
        counts = {s: window_states.count(s) for s in set(window_states)}
        best = max(counts, key=counts.get)
        out.append(LightState(best, counts[best] / len(window_states)))
    return out
