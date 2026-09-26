"""Wires io -> perception -> tracks into one call: build_track_store().

`perception` is injected (duck-typed: `.track_frame(frame_bgr) -> list[Detection]`)
so this module -- and its tests -- never have to import ultralytics: the
Perception class in detector.py is the real implementation, a fake with the
same method is enough for unit tests.
"""
from __future__ import annotations

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
    target_fps: float = 12.5,
) -> tuple[TrackStore, dict[str, list[tuple[float, LightState]]], VideoMeta]:
    """Run `perception` over sampled frames of `video_path`, aligning every
    detection into the scene's reference-frame coordinate space, and
    classify every configured traffic light on every sampled frame.

    Returns (track_store, light_log, meta). light_log maps traffic_light id
    -> chronological [(t_sec, LightState), ...] (already majority-smoothed).
    """
    video_path = Path(video_path)
    meta = read_meta(video_path)
    video_id = video_path.stem

    H_align = scene.alignment_for(video_id)  # this video's native px -> reference native px
    H_inv = np.linalg.inv(H_align)

    # All sample videos share one physical camera at fixed native resolution
    # (see docs/PLAN.md section 4); camera_alignment doesn't carry the
    # reference frame's own size, so this video's own native size is the
    # correct stand-in unless a future scene.json says otherwise.
    ref_w = float(scene.camera_alignment.get("reference_frame_size", [meta.width, meta.height])[0])
    ref_h = float(scene.camera_alignment.get("reference_frame_size", [meta.width, meta.height])[1])

    store = TrackStore(video_id=video_id, fps=meta.fps)
    raw_light_states: dict[str, list[tuple[float, LightState]]] = {tl.id: [] for tl in scene.traffic_lights}

    # ROI, warped once per video (camera doesn't move within a clip).
    video_local_rois = {
        tl.id: _warp_roi_to_video(tl.roi, H_inv, ref_w, ref_h, meta.width, meta.height)
        for tl in scene.traffic_lights
    }

    for frame in sample_frames(video_path, target_fps=target_fps):
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

        for tl_id, roi in video_local_rois.items():
            raw_light_states[tl_id].append((frame.t_sec, classify_roi(frame.bgr, roi)))

    light_log = {
        tl_id: list(zip((t for t, _ in entries), smooth_states([s for _, s in entries])))
        for tl_id, entries in raw_light_states.items()
    }
    return store, light_log, meta
