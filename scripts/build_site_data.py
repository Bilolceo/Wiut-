"""Build the website's data bundle: site/data/site.json + site/assets/*.jpg.

Inputs: predictions_samples.json (run_submission.py output), configs/*, the
sample videos (for the EDA brightness curves and preview frames) and the
scene overlays from scripts/draw_scene.py. Everything the page shows comes
from here, so the site always matches what the code produced.

Usage:
    python scripts/draw_scene.py --scene configs/scene.json --videos data/samples/*.MP4 --prefix scene
    python scripts/build_site_data.py --predictions predictions_samples.json --videos data/samples/*.MP4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_yaml  # noqa: E402
from src.scene.align import estimate_alignment  # noqa: E402

SITE = Path("site")
RISK_HZ = 2.0  # risk curves are downsampled to this rate, keeping each bucket's max (alarms stay visible)
BRIGHTNESS_EVERY_SEC = 5.0


def downsample_max(curve: list[list[float]], hz: float) -> list[list[float]]:
    out, bucket, t0 = [], [], None
    for t, v in curve:
        if t0 is None:
            t0 = t
        if t - t0 >= 1.0 / hz and bucket:
            out.append([round(t0, 2), round(max(bucket), 3)])
            bucket, t0 = [], t
        bucket.append(v)
    if bucket:
        out.append([round(t0, 2), round(max(bucket), 3)])
    return out


def video_stats(path: str) -> dict:
    """Resolution, fps, duration, brightness over time, and camera shift vs the reference frame."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    brightness, first = [], None
    for t in np.arange(0.0, n / fps, BRIGHTNESS_EVERY_SEC):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            break
        first = frame if first is None else first
        brightness.append([round(float(t), 1), round(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()), 1)])
    cap.release()
    A = estimate_alignment(first, cv2.imread("configs/reference_frame.jpg"), (w, h))
    shift = [round(float(A[0, 2]), 1), round(float(A[1, 2]), 1)] if A is not None else None
    return {"width": w, "height": h, "fps": round(fps, 2), "duration_sec": round(n / fps, 1), "brightness": brightness, "shift_px": shift}


def save_jpeg(src: Path, dst: Path, width: int = 1280) -> str | None:
    img = cv2.imread(str(src))
    if img is None:
        return None
    img = cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 78])
    return str(dst.relative_to(SITE))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", default="predictions_samples.json")
    ap.add_argument("--videos", nargs="+", required=True)
    args = ap.parse_args()

    preds = json.loads(Path(args.predictions).read_text())
    thresholds = load_yaml("thresholds.yaml")
    videos = []
    for path in sorted(args.videos):
        vid = Path(path).name
        entry, log = preds["videos"].get(vid, {}), preds["log"].get(vid, {})
        stats = video_stats(path)
        risk = entry.get("risk", [])
        videos.append({
            "id": Path(path).stem,
            **stats,
            "events": entry.get("events", []),
            "risk": downsample_max(risk, RISK_HZ),
            "risk_summary": {
                "mean": round(float(np.mean([v for _, v in risk])), 3) if risk else None,
                "frac_over_half": round(float(np.mean([v > 0.5 for _, v in risk])), 4) if risk else None,
            },
            "runtime": {k: log.get(k) for k in ("part_a_sec", "part_b_sec", "total_sec", "budget_sec")},
            "preview": save_jpeg(Path(f"docs/scene_debug/scene_{Path(path).stem}.jpg"), SITE / f"assets/scene_{Path(path).stem}.jpg"),
        })
    extra = {name: save_jpeg(Path(f"docs/scene_debug/{name}.jpg"), SITE / f"assets/{name}.jpg") for name in ("calib_C3896",)}
    bundle = {
        "generated_from": args.predictions,
        "classes": {k: v.get("enabled", False) for k, v in thresholds["classes"].items()},
        "videos": videos,
        "images": extra,
    }
    (SITE / "data").mkdir(parents=True, exist_ok=True)
    (SITE / "data/site.json").write_text(json.dumps(bundle, separators=(",", ":")))
    print(f"wrote site/data/site.json ({(SITE / 'data/site.json').stat().st_size // 1024} KB), {len(videos)} videos")


if __name__ == "__main__":
    main()
