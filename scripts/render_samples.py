"""Annotated renders of every sample video for the website.

For each video: a full annotated MP4 (site/assets/renders/<id>.mp4), one
example still per emitted class (site/assets/examples/) and the stills of the
curated failure cases (docs/failure_cases.json). Writes site/data/renders.json.

Events come from predictions_samples.json (the official harness output), so the
page shows exactly what was submitted; tracks come from a per-video cache of
the pipeline's track store (data/cache/<id>.tracks.pkl), built on first use.

Usage:
    python scripts/render_samples.py --videos data/samples/*.MP4 --predictions predictions_samples.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_yaml  # noqa: E402
from src.detect import detect_events_impl, get_scene, run_rules  # noqa: E402
from src.io.decode import read_meta  # noqa: E402
from src.viz.render import Annotator, render_video, snapshot  # noqa: E402

SITE = Path("site")


def load_tracks(video: str, cache_dir: Path) -> dict:
    cache = cache_dir / f"{Path(video).stem}.tracks.pkl"
    if not cache.exists():
        # verifier off: it does not change tracks, and tracks are all we need here
        _, rep = detect_events_impl(video, overrides={"verifier": {"enabled": False}})
        cache.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump({k: rep[k] for k in ("track_store", "light_log", "duration_sec")}, cache.open("wb"))
    return pickle.load(cache.open("rb"))


def save_still(img, name: str) -> str:
    out = SITE / "assets/examples" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return str(out.relative_to(SITE))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--predictions", default="predictions_samples.json")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--width", type=int, default=854)
    ap.add_argument("--skip-video", action="store_true", help="stills only (fast)")
    args = ap.parse_args()

    preds = json.loads(Path(args.predictions).read_text())["videos"]
    failures = json.loads(Path("docs/failure_cases.json").read_text())
    thresholds = load_yaml("thresholds.yaml")
    enabled = {k for k, v in thresholds["classes"].items() if v.get("enabled")}
    scene = get_scene()
    out = {"videos": {}}

    for video in sorted(args.videos):
        vid = Path(video).stem
        meta = read_meta(video)
        data = load_tracks(video, Path(args.cache_dir))
        store, light_log = data["track_store"], data["light_log"]
        for tr in store.tracks.values():  # metres from the current calibration
            for p in tr.points:
                p.x_m, p.y_m = scene.to_metres(p.x_norm, p.y_norm) or (None, None)
        events = preds[f"{vid}.MP4"]["events"]
        risk = preds[f"{vid}.MP4"]["risk"]
        candidates = run_rules(store, scene, light_log, enabled)  # only used to highlight who was involved
        ann = Annotator(scene, store, light_log, events, candidates, risk, (meta.width, meta.height))

        examples = []
        known_bad = [(f["label"], f["t"]) for f in failures if f["video"] == vid]
        for label in sorted({e[2] for e in events}):
            # a typical event (median duration) that is not one of the curated failure cases
            pool = [ev for ev in events if ev[2] == label and not any(
                lab == label and ev[0] - 1 <= t <= ev[1] + 1 for lab, t in known_bad)]
            if not pool:
                continue
            s, e, _ = sorted(pool, key=lambda ev: ev[1] - ev[0])[len(pool) // 2]
            img = snapshot(video, ann, (s + e) / 2)
            if img is not None:
                examples.append({"label": label, "start": s, "end": e, "img": save_still(img, f"{vid}_{label}.jpg")})
        fails = []
        for f in (f for f in failures if f["video"] == vid):
            img = snapshot(video, ann, f["t"])
            if img is not None:
                fails.append({**f, "img": save_still(img, f"{vid}_failure_{f['label']}_{int(f['t'])}.jpg")})

        existing = SITE / f"assets/renders/{vid}.mp4"
        render = str(existing.relative_to(SITE)) if existing.exists() else None
        if not args.skip_video:
            path = SITE / f"assets/renders/{vid}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            render_video(video, ann, path, width=args.width)
            render = str(path.relative_to(SITE))
        out["videos"][vid] = {"render": render, "examples": examples, "failures": fails}
        print(f"{vid}: {len(examples)} class examples, {len(fails)} failure cases, render={render}")

    (SITE / "data").mkdir(parents=True, exist_ok=True)
    (SITE / "data/renders.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
