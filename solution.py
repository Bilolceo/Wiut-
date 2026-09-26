"""
solution.py -- thin entry point called by the organizers' harness (run_submission.py).

    detect_events(video_path)  -> [[start_sec, end_sec, label], ...]    # Part A
    RiskEstimator().reset(meta); .step(frame, t_sec) -> float           # Part B

All logic lives under src/ (see docs/PLAN.md): src/detect.py for Part A,
src/risk/estimator.py for Part B. Neither function ever raises: failures are
logged and degrade to [] / the last score.
"""
from __future__ import annotations

import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format="[solution] %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("solution")

# Official class ids (14). Which ones are actually emitted is controlled by
# configs/thresholds.yaml (macro-F1: never emit a class we cannot back up).
CLASSES: list[str] = [
    "accident",
    "near_miss",
    "red_light",
    "wrong_way",
    "illegal_u_turn",
    "stopped_vehicle",
    "jaywalking",
    "failure_to_yield",
    "illegal_turn",
    "solid_line_crossing",
    "stop_line",
    "congestion",
    "road_obstacle",
    "fire_smoke",
]

# Anticipation horizon used by the metric (seconds).
RISK_HORIZON_SEC = 5.0


def detect_events(video_path: str) -> list[list]:
    """Part A -- traffic event detection. See src/detect.py."""
    from src.detect import detect_events as _detect  # deferred: heavy imports only when called

    return _detect(video_path)


class RiskEstimator:
    """Part B -- causal accident anticipation. See src/risk/estimator.py.

    step() uses only the frames it has been given, never opens the video
    file and never uses Part A results.
    """

    def __init__(self) -> None:
        self._impl = None
        self.last_score = 0.0
        try:
            from src.risk.estimator import CausalRiskEstimator

            self._impl = CausalRiskEstimator()
        except Exception:
            log.exception("RiskEstimator init failed; returning 0.0 for every frame")

    def reset(self, meta: dict) -> None:
        self.meta = meta
        self.last_score = 0.0
        if self._impl is not None:
            try:
                self._impl.reset(meta)
            except Exception:
                log.exception("RiskEstimator.reset failed; disabling Part B for this video")
                self._impl = None

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        if self._impl is None:
            return self.last_score
        try:
            self.last_score = self._impl.step(frame, t_sec)
        except Exception:
            log.exception("RiskEstimator.step failed at t=%.2f; keeping last score", t_sec)
            self._impl = None
        return self.last_score
