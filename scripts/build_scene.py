#!/usr/bin/env python3
"""Build configs/scene.json from the fixed-camera sample videos.

Two parts, per docs/PLAN.md section 4:

1. Automatic: run YOLO + a tracker on sampled frames of every sample video,
   collect vehicle track centroids, align each video onto the reference
   video's pixel frame (ORB + homography -- the camera does not move within
   a clip but the mount shifts a little between recordings), build a 16x16
   flow field, and cluster track start/end zones + mean direction into
   candidate "movements" (lanes[]).
2. Manual: crosswalks, stop_lines, traffic_lights, no_u_turn_zones,
   intersection, road_mask, homography come from configs/scene_manual.json
   (hand-drawn once on the reference frame; refine with tools/scene_editor).

Output: configs/scene.json = manual elements + data-driven lanes[].
Debug artifacts (for the website EDA section): docs/scene_debug/*.png

Usage:
    python scripts/build_scene.py \
        --videos data/samples/C3896.MP4 data/samples/C3897.MP4 \
                 data/samples/C3902.MP4 data/samples/C3905.MP4 \
        --ref-video data/samples/C3896.MP4 \
        --manual configs/scene_manual.json \
        --out configs/scene.json \
        --stride-fps 2 --resize-width 960 --max-frames 0

Runs on CPU (slow) or GPU (fast) -- ultralytics picks the device
automatically. Everything here is a *dev-time* tool: it is never imported
by solution.py / run_submission.py and is not bound by the eval machine's
no-internet / 5 GB weights constraint (it downloads its own small
COCO-pretrained yolo11n.pt on first run).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
from collections import defaultdict
from pathlib import Path

import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scene.align import estimate_alignment  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[build_scene] %(message)s")
log = logging.getLogger(__name__)

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)

# COCO class ids we treat as "vehicle-like" for lane clustering.
VEHICLE_CLASS_IDS = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

GRID = 16  # flow-field grid resolution


# --------------------------------------------------------------------------- #
# Camera alignment (ORB + homography onto the reference frame)
# --------------------------------------------------------------------------- #
def read_frame(video_path: str, t_sec: float = 5.0) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_sec * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame at {t_sec}s from {video_path}")
    return frame


def estimate_homography(moving_frame: np.ndarray, ref_frame: np.ndarray) -> np.ndarray:
    """H mapping moving-frame native px -> reference native px (identity if alignment fails).

    Same estimator the runtime uses (src/scene/align.py: CLAHE + ORB + 4-DOF
    similarity with a corner-shift sanity check), so offline and online
    alignments agree.
    """
    H = estimate_alignment(moving_frame, ref_frame, (ref_frame.shape[1], ref_frame.shape[0]))
    if H is None:
        log.warning("alignment failed; using identity")
        return np.eye(3)
    return H


def warp_point(H: np.ndarray, x: float, y: float) -> tuple[float, float]:
    p = H @ np.array([x, y, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


# --------------------------------------------------------------------------- #
# Detection + tracking
# --------------------------------------------------------------------------- #
def track_video(
    video_path: str,
    model,
    stride_frames: int,
    resize_width: int,
    max_frames: int,
) -> dict[int, list[tuple[float, float, int]]]:
    """Return {track_id: [(x_px, y_px, frame_idx), ...]} in *this video's*
    native (resized) pixel coordinates."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = resize_width / native_w
    resize_dim = (resize_width, int(round(native_h * scale)))

    tracks: dict[int, list[tuple[float, float, int]]] = defaultdict(list)
    frame_idx = 0
    processed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride_frames == 0:
            small = cv2.resize(frame, resize_dim)
            result = model.track(
                small,
                persist=True,
                classes=list(VEHICLE_CLASS_IDS.keys()),
                tracker="bytetrack.yaml",
                verbose=False,
            )[0]
            if result.boxes is not None and result.boxes.id is not None:
                xyxy = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), tid in zip(xyxy, ids):
                    cx, cy = (x1 + x2) / 2.0, y2  # bottom-centre: closer to ground contact point
                    tracks[int(tid)].append((float(cx), float(cy), frame_idx))
            processed += 1
            if max_frames and processed >= max_frames:
                break
        frame_idx += 1
    cap.release()
    log.info("%s: %d sampled frames, %d tracks", video_path, processed, len(tracks))
    return tracks, resize_dim


# --------------------------------------------------------------------------- #
# Flow field + movement clustering
# --------------------------------------------------------------------------- #
def accumulate_flow(
    flow_sum: np.ndarray, flow_count: np.ndarray, tracks_norm: dict[int, list[tuple[float, float, int]]]
) -> None:
    for pts in tracks_norm.values():
        pts = sorted(pts, key=lambda p: p[2])
        for (x0, y0, _), (x1, y1, _) in zip(pts, pts[1:]):
            if not (0.0 <= x0 < 1.0 and 0.0 <= y0 < 1.0):
                continue  # warped outside the reference frame
            gx, gy = int(x0 * GRID), int(y0 * GRID)
            flow_sum[gy, gx, 0] += x1 - x0
            flow_sum[gy, gx, 1] += y1 - y0
            flow_count[gy, gx] += 1


def cluster_movements(all_tracks_norm: list[dict[int, list]], min_track_len: int = 4) -> list[dict]:
    """Cheap direction-angle + start/end-zone clustering (no sklearn dep).

    Each qualifying track contributes one (start, end, direction) sample.
    Samples are grouped by 30-degree direction bins, then within a bin by
    proximity of (start, end) zones (grid cell). This is intentionally
    simple -- it gives good-enough "movement corridor" candidates; a human
    still confirms turn_allowed and prunes noise.
    """
    samples = []
    for tracks in all_tracks_norm:
        for pts in tracks.values():
            pts = sorted(pts, key=lambda p: p[2])
            if len(pts) < min_track_len:
                continue
            x0, y0, _ = pts[0]
            x1, y1, _ = pts[-1]
            dx, dy = x1 - x0, y1 - y0
            dist = math.hypot(dx, dy)
            if dist < 0.05:  # essentially stationary -- not a through-movement
                continue
            angle = math.degrees(math.atan2(dy, dx)) % 360
            samples.append(
                {
                    "start": (x0, y0),
                    "end": (x1, y1),
                    "dir": (dx / dist, dy / dist),
                    "angle_bin": int(angle // 30),
                    "points": [(px, py) for px, py, _ in pts],
                }
            )

    buckets: dict[tuple, list] = defaultdict(list)
    for s in samples:
        sx_bin = int(s["start"][0] * 6)
        sy_bin = int(s["start"][1] * 6)
        buckets[(s["angle_bin"], sx_bin, sy_bin)].append(s)

    movements = []
    for i, (_key, group) in enumerate(sorted(buckets.items(), key=lambda kv: -len(kv[1]))):
        if len(group) < 3:
            continue  # too few tracks to trust as a real movement
        all_pts = np.array([p for s in group for p in s["points"]])
        mean_dir = np.mean([s["dir"] for s in group], axis=0)
        mean_dir = (mean_dir / (np.linalg.norm(mean_dir) + 1e-9)).tolist()
        hull_pts = cv2_convex_hull(all_pts)
        movements.append(
            {
                "id": f"lane_auto_{i}",
                "polygon": hull_pts,
                "direction": mean_dir,
                "turn_allowed": ["unclassified"],
                "n_tracks": len(group),
                "note": "auto-clustered from tracked trajectories; confirm turn_allowed by eye",
            }
        )
    return movements


def cv2_convex_hull(points: np.ndarray) -> list[list[float]]:
    pts32 = points.astype(np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(pts32).reshape(-1, 2)
    return [[float(x), float(y)] for x, y in hull]


# --------------------------------------------------------------------------- #
# Debug visualisation
# --------------------------------------------------------------------------- #
def draw_debug(ref_frame: np.ndarray, flow_sum, flow_count, movements: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = ref_frame.shape[:2]

    flow_img = ref_frame.copy()
    for gy in range(GRID):
        for gx in range(GRID):
            n = flow_count[gy, gx]
            if n < 2:
                continue
            fx, fy = flow_sum[gy, gx] / n
            cx, cy = int((gx + 0.5) / GRID * w), int((gy + 0.5) / GRID * h)
            ex, ey = int(cx + fx * w * 4), int(cy + fy * h * 4)
            cv2.arrowedLine(flow_img, (cx, cy), (ex, ey), (0, 255, 255), 2, tipLength=0.4)
    cv2.imwrite(str(out_dir / "flow_field.png"), flow_img)

    mov_img = ref_frame.copy()
    palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    for i, m in enumerate(movements):
        color = palette[i % len(palette)]
        poly = np.array([[int(x * w), int(y * h)] for x, y in m["polygon"]], dtype=np.int32)
        cv2.polylines(mov_img, [poly], True, color, 2)
        cx, cy = poly.mean(axis=0).astype(int)
        cv2.putText(
            mov_img, f"{m['id']} (n={m['n_tracks']})", (cx - 20, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
        )
    cv2.imwrite(str(out_dir / "movements_overlay.png"), mov_img)
    log.info("wrote debug images to %s", out_dir)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--ref-video", required=True)
    ap.add_argument("--manual", default="configs/scene_manual.json")
    ap.add_argument("--out", default="configs/scene.json")
    ap.add_argument("--debug-dir", default="docs/scene_debug")
    ap.add_argument("--model", default="yolo11n.pt", help="ultralytics checkpoint; small model is enough for lane clustering")
    ap.add_argument("--stride-fps", type=float, default=2.0)
    ap.add_argument("--resize-width", type=int, default=960)
    ap.add_argument("--max-frames", type=int, default=0, help="cap sampled frames per video, 0 = no cap (use a small number for a smoke test)")
    args = ap.parse_args()

    from ultralytics import YOLO  # deferred: only needed for this dev tool

    model = YOLO(args.model)

    ref_frame_native = read_frame(args.ref_video)
    ref_h, ref_w = ref_frame_native.shape[:2]

    flow_sum = np.zeros((GRID, GRID, 2), dtype=np.float64)
    flow_count = np.zeros((GRID, GRID), dtype=np.int64)
    all_tracks_norm = []
    alignments = {}

    for video in args.videos:
        cap = cv2.VideoCapture(video)
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        stride_frames = max(1, round(native_fps / args.stride_fps))

        moving_frame_native = read_frame(video)
        H = estimate_homography(moving_frame_native, ref_frame_native)
        alignments[Path(video).stem] = H.tolist()

        tracks_px, resize_dim = track_video(video, model, stride_frames, args.resize_width, args.max_frames)
        rw, rh = resize_dim
        scale_to_native = ref_w / args.resize_width  # same resize_width used for every video

        tracks_norm = {}
        for tid, pts in tracks_px.items():
            warped = []
            for x, y, fidx in pts:
                # this-video resized px -> this-video native px -> ref native px (via H) -> normalized
                x_native, y_native = x * (ref_w / rw), y * (ref_h / rh)
                xr, yr = warp_point(H, x_native, y_native)
                warped.append((xr / ref_w, yr / ref_h, fidx))
            tracks_norm[tid] = warped
        all_tracks_norm.append(tracks_norm)
        accumulate_flow(flow_sum, flow_count, tracks_norm)

    movements = cluster_movements(all_tracks_norm)
    log.info("found %d candidate movements (lanes)", len(movements))

    manual_path = Path(args.manual)
    manual = json.loads(manual_path.read_text()) if manual_path.exists() else {}
    scene = dict(manual)
    # Hand-drawn lanes win: auto clusters from 2 fps tracks are noisy (ID switches
    # make long jumps), so they are kept for the website/EDA only.
    scene["lanes"] = manual.get("lanes") or movements
    scene["auto_movements"] = movements
    scene["camera_alignment"] = {
        "note": "3x3 homography mapping each video's native pixel coords onto the reference video's native pixel coords (identity if ORB alignment failed).",
        "per_video": alignments,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(scene, indent=2, ensure_ascii=False))
    log.info("wrote %s", out_path)

    draw_debug(ref_frame_native, flow_sum, flow_count, movements, Path(args.debug_dir))


if __name__ == "__main__":
    main()
