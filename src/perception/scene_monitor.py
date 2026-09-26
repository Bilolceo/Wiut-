"""Frame-level monitors for the classes that are not trajectory-based.

Both run inside build_track_store's decode loop on a small downscaled copy of
each processed frame, and hand their per-video result to the rules through
TrackStore.extras:

- StaticObjectMonitor (road_obstacle): dual-background subtraction. A slow
  MOG2 (minutes of memory) still sees a newly arrived object as foreground,
  while a fast MOG2 (seconds) has already absorbed it once it stops moving;
  "slow FG and not fast FG" held for >= min_static_sec is a static new object.
- FireSmokeMonitor (fire_smoke): colour/texture heuristics, persistence-gated.

Blobs are reported as bounding boxes normalized to THIS video's frame; the
pipeline maps them into the reference frame with `to_ref`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

PROC_WIDTH = 480


@dataclass
class Blob:
    t_sec: float
    box: tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized
    area: float  # fraction of the frame


def _small(frame_bgr: np.ndarray) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    return cv2.resize(frame_bgr, (PROC_WIDTH, round(h * PROC_WIDTH / w)), interpolation=cv2.INTER_AREA)


def _persistent_blobs(
    mask: np.ndarray, persist: np.ndarray, dt: float, min_sec: float, min_area: float, max_area: float, t: float
) -> list[Blob]:
    """Update the per-pixel persistence clock and return connected blobs held for >= min_sec.

    The clock starts at ~0 on the first frame a pixel is set (not at dt), so
    "held for min_sec" means min_sec of wall time since it appeared.
    """
    persist[:] = np.where(mask, np.where(persist > 0, persist + dt, 1e-6), 0.0)
    held = (persist >= min_sec).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(held, connectivity=8)
    h, w = mask.shape
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        frac = area / (h * w)
        if min_area <= frac <= max_area:
            out.append(Blob(t, (x / w, y / h, (x + bw) / w, (y + bh) / h), float(frac)))
    return out


def _map_blobs(blobs: list[Blob], to_ref: Callable[[float, float], tuple[float, float]]) -> list[Blob]:
    mapped = []
    for b in blobs:
        x1, y1 = to_ref(b.box[0], b.box[1])
        x2, y2 = to_ref(b.box[2], b.box[3])
        mapped.append(Blob(b.t_sec, (x1, y1, x2, y2), b.area))
    return mapped


@dataclass
class StaticObjectMonitor:
    name: str = "static_objects"
    every_sec: float = 0.5
    min_static_sec: float = 3.0
    warmup_sec: float = 5.0
    min_area: float = 0.0003
    max_area: float = 0.02
    global_change_frac: float = 0.4  # more FG than this = lighting jump / exposure change: reset
    _last_t: float = field(default=-1e9, init=False)
    _t0: float | None = field(default=None, init=False)
    _slow: cv2.BackgroundSubtractorMOG2 | None = field(default=None, init=False)
    _fast: cv2.BackgroundSubtractorMOG2 | None = field(default=None, init=False)
    _persist: np.ndarray | None = field(default=None, init=False)
    _blobs: list[Blob] = field(default_factory=list, init=False)

    def _reset(self, t_sec: float) -> None:
        self._t0 = t_sec
        self._slow = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
        self._fast = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=25, detectShadows=True)
        self._persist = None

    def update(self, frame_bgr: np.ndarray, t_sec: float) -> None:
        if t_sec - self._last_t < self.every_sec - 1e-6:
            return
        self._last_t = t_sec
        if self._slow is None:
            self._reset(t_sec)
        small = _small(frame_bgr)
        warm = t_sec - self._t0 < self.warmup_sec
        slow_rate = 0.2 if warm else 1.0 / (60.0 / self.every_sec)  # ~1 min memory after warm-up
        fg_slow = self._slow.apply(small, learningRate=slow_rate) > 200  # 127 = shadow
        fg_fast = self._fast.apply(small, learningRate=1.0 / (2.0 / self.every_sec)) > 200  # ~2 s memory
        if warm:
            return
        if fg_slow.mean() > self.global_change_frac:
            self._reset(t_sec)
            return
        static = cv2.morphologyEx((fg_slow & ~fg_fast).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        if self._persist is None:
            self._persist = np.zeros(static.shape, dtype=np.float32)
        self._blobs += _persistent_blobs(
            static, self._persist, self.every_sec, self.min_static_sec, self.min_area, self.max_area, t_sec
        )

    def result(self, to_ref: Callable[[float, float], tuple[float, float]]) -> list[Blob]:
        return _map_blobs(self._blobs, to_ref)


@dataclass
class FireSmokeMonitor:
    name: str = "fire_smoke"
    every_sec: float = 0.5
    min_persist_sec: float = 2.0
    fire_min_area: float = 0.0015  # brake lights / LED displays are far smaller
    smoke_min_area: float = 0.01
    _last_t: float = field(default=-1e9, init=False)
    _bg: cv2.BackgroundSubtractorMOG2 | None = field(default=None, init=False)
    _fire_persist: np.ndarray | None = field(default=None, init=False)
    _smoke_persist: np.ndarray | None = field(default=None, init=False)
    _fire: list[Blob] = field(default_factory=list, init=False)
    _smoke: list[Blob] = field(default_factory=list, init=False)

    def update(self, frame_bgr: np.ndarray, t_sec: float) -> None:
        if t_sec - self._last_t < self.every_sec - 1e-6:
            return
        self._last_t = t_sec
        small = _small(frame_bgr)
        if self._bg is None:
            self._bg = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=25, detectShadows=False)
            self._fire_persist = np.zeros(small.shape[:2], dtype=np.float32)
            self._smoke_persist = np.zeros(small.shape[:2], dtype=np.float32)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        fire = (hue <= 30) & (sat >= 150) & (val >= 220)
        moving = self._bg.apply(small) > 0
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        smooth = np.abs(cv2.Laplacian(grey, cv2.CV_32F)) < 8  # smoke is a low-texture haze
        smoke = moving & (sat < 40) & (val >= 90) & (val <= 230) & smooth
        self._fire += _persistent_blobs(fire, self._fire_persist, self.every_sec, self.min_persist_sec, self.fire_min_area, 0.5, t_sec)
        self._smoke += _persistent_blobs(smoke, self._smoke_persist, self.every_sec, self.min_persist_sec, self.smoke_min_area, 0.5, t_sec)

    def result(self, to_ref: Callable[[float, float], tuple[float, float]]) -> dict[str, list[Blob]]:
        return {"fire": _map_blobs(self._fire, to_ref), "smoke": _map_blobs(self._smoke, to_ref)}
