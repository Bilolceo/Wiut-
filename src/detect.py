"""Part A: detect_events(video_path) -> [[start_sec, end_sec, label], ...].

decode (10 fps) -> YOLO + BoT-SORT + traffic-light state + frame monitors
-> track store (reference-frame coords, metres) -> every event rule
-> post-processing (enabled classes, thresholds, non-overlapping merge).

Never raises: any failure is logged and returns [] (CLAUDE.md hard rule),
so one bad video cannot take the whole submission down.
"""
from __future__ import annotations

import logging
import random
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from src.config import load_yaml, repo_path
from src.events.base import Candidate
from src.events.registry import ALL_RULES
from src.io.decode import read_meta
from src.perception.pipeline import build_track_store
from src.perception.scene_monitor import FireSmokeMonitor, StaticObjectMonitor
from src.post.postprocess import finalize
from src.scene.align import estimate_alignment
from src.scene.loader import Scene, load_scene

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_scene() -> Scene:
    """configs/scene.json, loaded once per process."""
    return load_scene(repo_path("configs/scene.json"))


@lru_cache(maxsize=1)
def get_reference_frame() -> np.ndarray | None:
    frame = cv2.imread(str(repo_path("configs/reference_frame.jpg")))
    if frame is None:
        log.warning("configs/reference_frame.jpg missing: falling back to static per-video alignment")
    return frame


def make_aligner(ref_native_size: tuple[int, int]):
    """Runtime aligner onto the reference frame, or None if no reference frame is available."""
    ref = get_reference_frame()
    if ref is None:
        return None
    return lambda frame_bgr: estimate_alignment(frame_bgr, ref, ref_native_size)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def _make_perception(cfg: dict):
    from src.perception.detector import Perception  # deferred: pulls in ultralytics/torch

    p = cfg["perception"]
    return Perception(
        model_path=str(repo_path(p["model"])),
        conf=p["conf"],
        tracker=str(repo_path(p["tracker"])),
        device=p.get("device", "auto"),
        imgsz=p["imgsz"],
    )


def _make_analyzers(cfg: dict) -> tuple:
    enabled = cfg.get("analyzers", {})
    analyzers = []
    if enabled.get("static_objects"):
        analyzers.append(StaticObjectMonitor())
    if enabled.get("fire_smoke"):
        analyzers.append(FireSmokeMonitor())
    return tuple(analyzers)


@lru_cache(maxsize=1)
def get_vlm_backend(model_dir: str):
    """Qwen3-VL loaded once per process (~4.3 GB); None if the weights are not there."""
    if not Path(model_dir).is_dir():
        log.warning("VLM weights not found at %s: verifier disabled, its classes will not be emitted", model_dir)
        return None
    from src.verify.vlm import QwenVLBackend  # deferred: pulls in transformers

    return QwenVLBackend(model_dir)


def verify(candidates: list[Candidate], video_path: str, store, scene: Scene, meta, cfg: dict, deadline: float) -> list[Candidate]:
    """Run the VLM verifier on the classes that need it; a failure drops only those classes, never the video."""
    vcfg = cfg.get("verifier", {})
    labels = list(vcfg.get("labels", []))  # priority order
    if not vcfg.get("enabled") or not any(c.label in labels for c in candidates):
        return candidates
    try:
        backend = get_vlm_backend(str(repo_path(vcfg["model_dir"])))
        if backend is None:
            return [c for c in candidates if c.label not in labels]
        from src.verify.vlm import Verifier

        alignments = store.extras.get("alignments") or []
        H = np.array(alignments[-1][1]) if alignments else scene.alignment_for(store.video_id, (meta.width, meta.height))
        H_inv = np.linalg.inv(H)
        ref_w, ref_h = scene.reference_size((meta.width, meta.height))

        def to_video(x_norm: float, y_norm: float) -> tuple[float, float]:
            q = H_inv @ np.array([x_norm * ref_w, y_norm * ref_h, 1.0])
            return q[0] / q[2] / meta.width, q[1] / q[2] / meta.height

        verifier = Verifier(backend, labels, vcfg.get("max_calls", 30), vcfg.get("threshold", 0.5))  # labels keep config order
        return verifier.filter(candidates, video_path, store, to_video, deadline)
    except Exception:
        log.exception("VLM verification failed on %s; dropping %s", video_path, sorted(labels))
        return [c for c in candidates if c.label not in labels]


def run_rules(track_store, scene: Scene, light_log: dict, enabled: set[str]) -> list[Candidate]:
    """Run every enabled rule; a failing rule is logged and skipped, never fatal."""
    candidates: list[Candidate] = []
    for rule_cls in ALL_RULES:
        if rule_cls.label not in enabled:
            continue
        try:
            candidates += rule_cls().run(track_store, scene, light_log)
        except Exception:
            log.exception("rule %s failed on %s", rule_cls.label, track_store.video_id)
    return candidates


def detect_events_impl(video_path: str, perception=None) -> tuple[list[list], dict]:
    """detect_events plus a debug report (per-stage timings, candidate counts).

    `perception` can be injected (tests / reuse); by default a fresh model +
    tracker is built for every video (tracker state must not leak across videos).
    """
    t0 = time.perf_counter()
    cfg = load_yaml("runtime.yaml")
    thresholds = load_yaml("thresholds.yaml")
    _seed(cfg.get("seed", 1234))
    scene = get_scene()
    meta = read_meta(video_path)
    deadline = t0 + cfg["budget"]["part_a_max_factor"] * meta.duration_sec

    perception = perception or _make_perception(cfg)
    t_model = time.perf_counter()
    store, light_log, meta = build_track_store(
        video_path,
        scene,
        perception,
        target_fps=cfg["target_fps"],
        analyzers=_make_analyzers(cfg),
        deadline=deadline,
        aligner=make_aligner(tuple(int(v) for v in scene.reference_size((meta.width, meta.height)))),
        realign_every_sec=cfg.get("realign_every_sec", 10.0),
    )
    t_tracks = time.perf_counter()

    enabled = {name for name, c in thresholds["classes"].items() if c.get("enabled")}
    candidates = run_rules(store, scene, light_log, enabled)
    t_rules = time.perf_counter()
    verify_deadline = t0 + cfg.get("verifier", {}).get("deadline_factor", 1.3) * meta.duration_sec
    candidates = verify(candidates, video_path, store, scene, meta, cfg, verify_deadline)
    events = finalize(candidates, meta.duration_sec, thresholds)
    t_end = time.perf_counter()

    report = {
        "video_id": Path(video_path).stem,
        "duration_sec": round(meta.duration_sec, 2),
        "n_tracks": len(store.tracks),
        "truncated_at_sec": store.extras.get("truncated_at_sec"),
        "sec_model_load": round(t_model - t0, 2),
        "sec_tracking": round(t_tracks - t_model, 2),
        "sec_rules": round(t_rules - t_tracks, 2),
        "sec_verify_post": round(t_end - t_rules, 2),
        "runtime_factor": round((t_end - t0) / max(meta.duration_sec, 1e-6), 3),
        "candidates": candidates,
        "track_store": store,
        "light_log": light_log,
    }
    return events, report


def detect_events(video_path: str) -> list[list]:
    try:
        events, report = detect_events_impl(video_path)
        log.info(
            "%s: %d events, runtime %.2fx (tracking %.1fs)",
            report["video_id"], len(events), report["runtime_factor"], report["sec_tracking"],
        )
        return events
    except Exception:
        log.exception("detect_events failed on %s", video_path)
        return []
