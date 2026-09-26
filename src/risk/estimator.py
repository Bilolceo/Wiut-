"""Part B: causal accident-risk estimator, P(accident starts within 5 s).

Causality is structural: step() only ever sees the frame it is handed and
the state it built from earlier frames. It never opens the video file and
never touches Part A output (CLAUDE.md hard rule).

Per processed frame (every `stride_frames`): YOLO + ByteTrack -> ground
points mapped onto the reference frame (runtime alignment) and into metres
(scene homography) -> per-track velocity from the last `history_sec` ->
pairwise TTC / DRAC for nearby road users -> hand-set sigmoid -> EMA.
"""
from __future__ import annotations

import logging
import math
from collections import deque

import numpy as np

from src.config import load_yaml, repo_path
from src.detect import get_reference_frame, get_scene
from src.scene.align import estimate_alignment
from src.tracks.kinematics import drac_mps2, time_to_collision
from src.tracks.store import VEHICLE_CLASSES

log = logging.getLogger(__name__)


def in_box(xy: tuple[float, float], box: list[float]) -> bool:
    return box[0] <= xy[0] <= box[1] and box[2] <= xy[1] <= box[3]


def risk_from_conflict(min_ttc: float | None, max_drac: float, cfg: dict) -> float:
    """Map the worst current conflict to [0, 1]; no conflict -> sigmoid(bias) (small)."""
    z = cfg["bias"] + cfg["drac_weight"] * min(max_drac / cfg["drac_ref_mps2"], cfg["drac_cap"])
    if min_ttc is not None:
        z += cfg["ttc_slope"] * (cfg["ttc_mid_sec"] - min_ttc)
    return 1.0 / (1.0 + math.exp(-z))


class TrackHistory:
    """World positions of live tracks over the last `history_sec` (past only)."""

    def __init__(self, history_sec: float, stale_sec: float) -> None:
        self.history_sec, self.stale_sec = history_sec, stale_sec
        self.tracks: dict[int, tuple[str, deque]] = {}

    def update(self, t: float, observations: list[tuple[int, str, float, float]]) -> None:
        for tid, cls, x, y in observations:
            _, hist = self.tracks.setdefault(tid, (cls, deque()))
            hist.append((t, x, y))
            while hist and t - hist[0][0] > self.history_sec:
                hist.popleft()
        for tid in [k for k, (_, h) in self.tracks.items() if t - h[-1][0] > self.stale_sec]:
            del self.tracks[tid]

    def states(self, t: float) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """(cls, position, velocity) for tracks observed at t with enough history for a velocity."""
        out = []
        for cls, hist in self.tracks.values():
            if hist[-1][0] < t - 1e-6 or len(hist) < 2:
                continue
            (t0, x0, y0), (t1, x1, y1) = hist[0], hist[-1]
            if t1 - t0 < 0.2:
                continue
            out.append((cls, np.array([x1, y1]), np.array([(x1 - x0) / (t1 - t0), (y1 - y0) / (t1 - t0)])))
        return out


def worst_conflict(states: list[tuple[str, np.ndarray, np.ndarray]], cfg: dict) -> tuple[float | None, float]:
    """(min TTC over pairs, max DRAC over pairs) among nearby, moving road users."""
    min_ttc, max_drac = None, 0.0
    radius = cfg["radius_m"]
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            (ca, pa, va), (cb, pb, vb) = states[i], states[j]
            if ca not in VEHICLE_CLASSES and cb not in VEHICLE_CLASSES:
                continue
            if np.linalg.norm(pa - pb) > cfg["max_pair_distance_m"]:
                continue
            if max(np.linalg.norm(va), np.linalg.norm(vb)) < cfg["min_speed_mps"]:
                continue
            d = pb - pa
            closing = -float((vb - va) @ d) / max(float(np.linalg.norm(d)), 1e-6)
            if closing < cfg["min_closing_mps"]:  # detector jitter gives ~0.3-0.5 m/s of fake relative motion
                continue
            r = radius["vehicle" if ca in VEHICLE_CLASSES else "pedestrian"] + radius["vehicle" if cb in VEHICLE_CLASSES else "pedestrian"]
            ttc = time_to_collision(pa, va, pb, vb, r)
            if ttc is not None and (min_ttc is None or ttc < min_ttc):
                min_ttc = ttc
            max_drac = max(max_drac, drac_mps2(pa, va, pb, vb, r))
    return min_ttc, max_drac


class CausalRiskEstimator:
    def __init__(self, perception_factory=None) -> None:
        self.cfg = load_yaml("risk.yaml")
        self.scene = get_scene()
        self._perception_factory = perception_factory or self._default_perception
        self.perception = None
        self.reset({"video_id": "", "fps": 30.0, "width": 3840, "height": 2160, "n_frames": 0})

    def _default_perception(self):
        from src.perception.detector import Perception  # deferred heavy import

        p = self.cfg["perception"]
        return Perception(
            model_path=str(repo_path(p["model"])), conf=p["conf"], tracker=p["tracker"],
            device=p.get("device", "auto"), imgsz=p["imgsz"],
        )

    def reset(self, meta: dict) -> None:
        self.meta = meta
        self.frame_idx = -1
        self.score = 0.0
        self.history = TrackHistory(self.cfg["history_sec"], self.cfg["stale_sec"])
        video_id = str(meta.get("video_id", "")).rsplit(".", 1)[0]
        self.video_size = (int(meta.get("width") or 3840), int(meta.get("height") or 2160))
        self.ref_w, self.ref_h = self.scene.reference_size(self.video_size)
        self.H_align = self.scene.alignment_for(video_id, self.video_size)
        self.next_align_t = 0.0
        self.perception = None  # fresh tracker per video: ids must not leak across videos

    def _maybe_realign(self, frame: np.ndarray, t: float) -> None:
        if t < self.next_align_t:
            return
        self.next_align_t = t + self.cfg["realign_every_sec"]
        ref = get_reference_frame()
        if ref is not None:
            H = estimate_alignment(frame, ref, (int(self.ref_w), int(self.ref_h)))
            if H is not None:
                self.H_align = H

    def _observations(self, frame: np.ndarray) -> list[tuple[int, str, float, float]]:
        h, w = frame.shape[:2]
        out = []
        for det in self.perception.track_frame(frame):
            p = self.H_align @ np.array([det.cx_norm * w, det.cy_norm * h, 1.0])
            world = self.scene.to_metres(p[0] / p[2] / self.ref_w, p[1] / p[2] / self.ref_h)
            if world is not None and in_box(world, self.cfg["valid_world_box_m"]):
                out.append((det.track_id, det.cls, world[0], world[1]))
        return out

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        self.frame_idx += 1
        if self.frame_idx % self.cfg["stride_frames"]:
            return self.score
        if self.perception is None:
            self.perception = self._perception_factory()
        self._maybe_realign(frame, t_sec)
        self.history.update(t_sec, self._observations(frame))
        min_ttc, max_drac = worst_conflict(self.history.states(t_sec), self.cfg)
        raw = risk_from_conflict(min_ttc, max_drac, self.cfg)
        a = self.cfg["ema_alpha"]
        self.score = float(np.clip(a * raw + (1 - a) * self.score, 0.0, 1.0))
        return self.score
