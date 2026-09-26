"""Overlay scene geometry on a video frame, to check it by eye.

Geometry is authored in the reference frame's normalized coords; for any other
video it is mapped into that video through the inverse camera alignment, so the
overlay shows exactly what the rules will see. A 5 m ground grid drawn through
the inverse homography makes a wrong metric calibration obvious.

Usage:
    python scripts/draw_scene.py --scene configs/scene.json \
        --videos data/samples/*.MP4 --out-dir docs/scene_debug
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scene.loader import Scene, load_scene  # noqa: E402

COLORS = {
    "crosswalk": (0, 255, 255),
    "stop_line": (0, 0, 255),
    "solid_line": (255, 255, 255),
    "light": (0, 255, 0),
    "refuge": (255, 0, 255),
    "no_u_turn": (0, 128, 255),
    "intersection": (255, 128, 0),
    "road_mask": (200, 200, 200),
    "lane": (255, 255, 0),
    "grid": (80, 220, 80),
}


def read_frame(video: str, t_sec: float) -> np.ndarray:
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_sec * (cap.get(cv2.CAP_PROP_FPS) or 30.0)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read {video} at {t_sec}s")
    return frame


def draw(scene: Scene, frame: np.ndarray, video_stem: str, out_w: int = 1920) -> np.ndarray:
    h, w = frame.shape[:2]
    to_video = np.linalg.inv(scene.alignment_for(video_stem))  # reference px -> this video's px
    scale = out_w / w
    img = cv2.resize(frame, (out_w, int(h * scale)))

    def px(points) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2) * [w, h]
        pts = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), to_video).reshape(-1, 2)
        return (pts * scale).round().astype(np.int32)

    def label(text: str, at: np.ndarray, color) -> None:
        x, y = int(at[0]), int(at[1])
        cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    if scene.road_mask is not None:
        cv2.polylines(img, [px(scene.road_mask)], True, COLORS["road_mask"], 1)
    if scene.intersection is not None:
        cv2.polylines(img, [px(scene.intersection)], True, COLORS["intersection"], 1)
    for lane in scene.lanes:
        p = px(lane.polygon)
        cv2.polylines(img, [p], True, COLORS["lane"], 1)
        c = p.mean(axis=0)
        d = lane.direction / (np.linalg.norm(lane.direction) + 1e-9)
        cv2.arrowedLine(img, tuple(c.astype(int)), tuple((c + d * 60).astype(int)), COLORS["lane"], 2, tipLength=0.3)
    for kind, zones in (("crosswalk", scene.crosswalks), ("refuge", scene.refuges), ("no_u_turn", scene.no_u_turn_zones)):
        for z in zones:
            p = px(z.polygon)
            cv2.polylines(img, [p], True, COLORS[kind], 2)
            label(z.id, p[0], COLORS[kind])
    for sl in scene.solid_lines:
        cv2.polylines(img, [px(sl.polygon)], False, COLORS["solid_line"], 2)
    for sl in scene.stop_lines:
        p = px(sl.line)
        cv2.line(img, tuple(p[0]), tuple(p[1]), COLORS["stop_line"], 3)
        label(f"{sl.id} <- {sl.controlled_by}", p[0], COLORS["stop_line"])
    for tl in scene.traffic_lights:
        x1, y1, x2, y2 = tl.roi
        p = px([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        cv2.polylines(img, [p], True, COLORS["light"], 2)
        label(f"{tl.id} ({tl.kind})", p[0], COLORS["light"])

    H = scene.homography_matrix()
    if H is not None:
        to_image = np.linalg.inv(H)  # world metres -> reference normalized
        for a in range(-30, 61, 5):  # 5 m ground grid: lines along x (across lanes) and along y (along travel)
            for line in ([(a, b) for b in np.linspace(-35, 60, 96)], [(b, a) for b in np.linspace(-30, 60, 91)]):
                world = np.array(line, dtype=np.float64).reshape(-1, 1, 2)
                ref = cv2.perspectiveTransform(world, to_image).reshape(-1, 2)
                ok = (ref > -0.2).all(axis=1) & (ref < 1.2).all(axis=1)
                if ok.sum() > 1:
                    cv2.polylines(img, [px(ref[ok])], False, COLORS["grid"], 1)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="configs/scene.json")
    ap.add_argument("--alignment-from", help="take camera_alignment from this scene file instead")
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--t-sec", type=float, default=5.0)
    ap.add_argument("--out-dir", default="docs/scene_debug")
    ap.add_argument("--prefix", default="scene")
    args = ap.parse_args()

    scene = load_scene(args.scene)
    if args.alignment_from:
        scene.camera_alignment = load_scene(args.alignment_from).camera_alignment
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for video in args.videos:
        stem = Path(video).stem
        out = out_dir / f"{args.prefix}_{stem}.jpg"
        cv2.imwrite(str(out), draw(scene, read_frame(video, args.t_sec), stem), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
