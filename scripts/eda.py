"""Quick EDA for sample videos: metadata, brightness over time, and preview frames.

Usage:
    python scripts/eda.py --videos data/samples --out docs/frames
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def video_meta(cap: cv2.VideoCapture) -> dict:
    """Return resolution, fps, frame count and duration of an opened video."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": fps,
        "n_frames": n,
        "duration": n / fps,
    }


def read_frame(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
    """Seek to frame `idx` and return it (BGR) or None."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return frame if ok else None


def resize_width(frame: np.ndarray, width: int) -> np.ndarray:
    """Resize keeping aspect ratio so that the output has the given width."""
    h, w = frame.shape[:2]
    return cv2.resize(frame, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)


def brightness_profile(cap: cv2.VideoCapture, meta: dict, step_sec: float) -> list[tuple[float, float]]:
    """Mean grey level (0-255) sampled every `step_sec` seconds."""
    out = []
    t = 0.0
    while t < meta["duration"]:
        frame = read_frame(cap, int(t * meta["fps"]))
        if frame is None:
            break
        out.append((t, float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())))
        t += step_sec
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos", default="data/samples")
    ap.add_argument("--out", default="docs/frames")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--brightness-step", type=float, default=10.0)
    args = ap.parse_args()

    src = Path(args.videos)
    videos = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() == ".mp4")
    for path in videos:
        cap = cv2.VideoCapture(str(path))
        meta = video_meta(cap)
        d = meta["duration"]
        print(f"\n== {path.name}: {meta['width']}x{meta['height']} @ {meta['fps']:.2f} fps, "
              f"{meta['n_frames']} frames, {d:.1f} s")

        prof = brightness_profile(cap, meta, args.brightness_step)
        vals = [b for _, b in prof]
        print(f"   brightness mean {np.mean(vals):.0f}, min {min(vals):.0f}, max {max(vals):.0f}")
        print("   " + " ".join(f"{t:.0f}s:{b:.0f}" for t, b in prof))

        out_dir = Path(args.out) / path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        times = {"start": min(5.0, d * 0.05), "third1": d / 3, "third2": 2 * d / 3, "end": max(0.0, d - 2.0)}
        for name, t in times.items():
            frame = read_frame(cap, int(t * meta["fps"]))
            if frame is not None:
                cv2.imwrite(str(out_dir / f"{name}_{t:.1f}s.png"), resize_width(frame, args.width))
        cap.release()
        print(f"   frames -> {out_dir}")


if __name__ == "__main__":
    main()
