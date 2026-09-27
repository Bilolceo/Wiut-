"""Wires io -> perception -> tracks into one call: build_track_store().

`perception` is injected (duck-typed: `.track_frame(frame_bgr) -> list[Detection]`)
so this module -- and its tests -- never have to import ultralytics: the
Perception class in detector.py is the real implementation, a fake with the
same method is enough for unit tests.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from src.io.decode import VideoMeta, read_meta, sample_frames
from src.perception.traffic_light import LightState, classify_roi, smooth_states
from src.scene.loader import Scene
from src.tracks.store import TrackPoint, TrackStore


def _warp_point(H: np.ndarray, x: float, y: float) -> tuple[float, float]:
    p = H @ np.array([x, y, 1.0])
    if abs(p[2]) < 1e-9:
        return x, y
    return float(p[0] / p[2]), float(p[1] / p[2])


def _warp_roi_to_video(
    roi_norm: tuple[float, float, float, float],
    H_inv: np.ndarray,
    ref_w: float,
    ref_h: float,
    vid_w: float,
    vid_h: float,
) -> tuple[float, float, float, float]:
    """A traffic-light ROI is authored once, in the reference frame's
    normalized coords. To read it correctly in a *different* recording
    (small camera-mount shift between clips -- see docs/PLAN.md section 4
    EDA note), warp its corners back into this video's own pixel space with
    the inverse of the alignment homography before sampling colour there."""
    x1, y1, x2, y2 = roi_norm
    corners = [(x1 * ref_w, y1 * ref_h), (x2 * ref_w, y1 * ref_h), (x2 * ref_w, y2 * ref_h), (x1 * ref_w, y2 * ref_h)]
    warped = [_warp_point(H_inv, x, y) for x, y in corners]
    xs = [p[0] for p in warped]
    ys = [p[1] for p in warped]
    return (min(xs) / vid_w, min(ys) / vid_h, max(xs) / vid_w, max(ys) / vid_h)


def build_track_store(
    video_path: str | Path,
    scene: Scene,
    perception,
    target_fps: float = 10.0,
    analyzers: tuple = (),
    deadline: float | None = None,
    aligner=None,
    realign_every_sec: float = 10.0,
    progress=None,
) -> tuple[TrackStore, dict[str, list[tuple[float, LightState]]], VideoMeta]:
    """Run `perception` over sampled frames of `video_path`, aligning every
    detection into the scene's reference-frame coordinate space, and
    classify every configured traffic light on every sampled frame.

    Returns (track_store, light_log, meta). light_log maps traffic_light id
    -> chronological [(t_sec, LightState), ...] (already majority-smoothed).

    `analyzers` (duck-typed: .name, .update(frame_bgr, t_sec), .result(to_ref))
    see every sampled frame too; their results land in track_store.extras.

    `deadline` (time.perf_counter() value): stop decoding once passed and
    return what was seen -- partial events beat a video scored empty for
    exceeding the harness time budget. store.extras["truncated_at_sec"] marks it.

    `aligner(frame_bgr) -> 3x3 | None` estimates this video's alignment onto
    the reference frame at runtime (unknown test videos); it runs on the first
    frame and every `realign_every_sec`, because the mount also drifts slowly
    within a clip (~15 px at 4K over 10 s on C3896). Without it, the static
    per-video alignment from scene.json is used.

    `progress(fraction)` is called about once per second of video (demo UI).
    """
    video_path = Path(video_path)
    meta = read_meta(video_path)
    video_id = video_path.stem

    # this video's native px -> reference native px (static fallback until the aligner runs)
    H_align = scene.alignment_for(video_id, (meta.width, meta.height))
    H_inv = np.linalg.inv(H_align)
    ref_w, ref_h = scene.reference_size((meta.width, meta.height))

    store = TrackStore(video_id=video_id, fps=meta.fps, duration_sec=meta.duration_sec)
    raw_light_states: dict[str, list[tuple[float, LightState]]] = {tl.id: [] for tl in scene.traffic_lights}

    layouts = {tl.id: tl.layout for tl in scene.traffic_lights}

    def local_rois(H_inv: np.ndarray) -> dict[str, tuple[float, float, float, float]]:
        return {
            tl.id: _warp_roi_to_video(tl.roi, H_inv, ref_w, ref_h, meta.width, meta.height)
            for tl in scene.traffic_lights
        }

    video_local_rois = local_rois(H_inv)
    alignments: list[tuple[float, list]] = []
    next_align_t = 0.0
    last_progress_sec = -1

    for frame in sample_frames(video_path, target_fps=target_fps):
        if deadline is not None and time.perf_counter() > deadline:
            store.extras["truncated_at_sec"] = frame.t_sec
            break
        if progress is not None and int(frame.t_sec) != last_progress_sec:
            last_progress_sec = int(frame.t_sec)
            progress(min(1.0, frame.t_sec / max(meta.duration_sec, 1e-6)))
        if aligner is not None and frame.t_sec >= next_align_t:
            next_align_t = frame.t_sec + realign_every_sec
            H_new = aligner(frame.bgr)
            if H_new is not None:
                H_align, H_inv = H_new, np.linalg.inv(H_new)
                video_local_rois = local_rois(H_inv)
                alignments.append((frame.t_sec, H_align.round(4).tolist()))
        for det in perception.track_frame(frame.bgr):
            x_native, y_native = det.cx_norm * meta.width, det.cy_norm * meta.height
            xr, yr = _warp_point(H_align, x_native, y_native)
            x_ref_norm, y_ref_norm = xr / ref_w, yr / ref_h
            world = scene.to_metres(x_ref_norm, y_ref_norm)
            x_m, y_m = world if world is not None else (None, None)
            store.add_point(
                det.track_id,
                det.cls,
                TrackPoint(
                    frame_idx=frame.frame_idx,
                    t_sec=frame.t_sec,
                    x_norm=x_ref_norm,
                    y_norm=y_ref_norm,
                    x_m=x_m,
                    y_m=y_m,
                    w_norm=det.w_norm,
                    h_norm=det.h_norm,
                    conf=det.conf,
                ),
            )

        for analyzer in analyzers:
            analyzer.update(frame.bgr, frame.t_sec)

        for tl_id, roi in video_local_rois.items():
            raw_light_states[tl_id].append((frame.t_sec, classify_roi(frame.bgr, roi, layouts[tl_id])))

    def to_ref(x_norm: float, y_norm: float) -> tuple[float, float]:
        xr, yr = _warp_point(H_align, x_norm * meta.width, y_norm * meta.height)
        return xr / ref_w, yr / ref_h

    for analyzer in analyzers:
        store.extras[analyzer.name] = analyzer.result(to_ref)
    store.extras["alignments"] = alignments

    light_log = {
        tl_id: list(zip((t for t, _ in entries), smooth_states([s for _, s in entries])))
        for tl_id, entries in raw_light_states.items()
    }
    return store, light_log, meta
