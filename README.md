# WIUT Hackathon 2026 — Computer Vision track

Traffic events from a fixed CCTV road camera (Tashkent intersection, 4K @ 29.97 fps):

- **Part A** `detect_events(video)` → `[[start_sec, end_sec, label], ...]` over 14 event classes
- **Part B** `RiskEstimator.step(frame, t)` → causal P(accident starts within 5 s)

Full design and decisions: [`docs/PLAN.md`](docs/PLAN.md). Organizers' starter-kit notes: [`docs/STARTER_KIT.md`](docs/STARTER_KIT.md).

## Install and run

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # pinned ==; Linux wheel of torch includes CUDA (T4)
./weights/download.sh                      # verifies sha256; yolo11n.pt is also committed (5.6 MB)
python run_submission.py --videos data/samples --out predictions.json
python evaluate.py --pred predictions.json --validate-only
pytest -q                                  # 71 unit tests, no video / GPU needed
```

The eval machine needs no internet: weights are local and `YOLO_OFFLINE=1` is set before ultralytics loads.
Minimal Linux images need `libgl1 libglib2.0-0` (ultralytics pulls in `opencv-python`); see `Dockerfile`.

## Architecture

```
video ─► decode every 3rd frame (≈10 fps, grab/retrieve, no seeking)
        ├─► runtime camera alignment (ORB + 4-DOF similarity vs configs/reference_frame.jpg, every 10 s)
        ├─► YOLO11n @960 + BoT-SORT (no GMC: fixed camera)     ─► track store (reference coords + metres)
        ├─► traffic-light state (HSV, lamp-position check)        ─► light log
        └─► static-object monitor (dual MOG2)                    ─► frame extras
   ─► 14 event rules (src/events/, one module per class) ─► post-processing (enabled classes,
      min conf / duration, same-class merge, clip) ─► events
```

| Component | Learned or rule-based | Where |
|---|---|---|
| Road-user detection + tracking | learned (YOLO11n, COCO) + BoT-SORT | `src/perception/detector.py` |
| Camera alignment across recordings | classical (ORB, RANSAC similarity) | `src/scene/align.py` |
| Scene geometry (zebras, stop line, islands, lanes) | hand-drawn once on the reference frame | `configs/scene_manual.json` |
| Traffic-light colour | classical (HSV + lamp position) | `src/perception/traffic_light.py` |
| Event classes | rules on trajectories, geometry and light state | `src/events/*.py` |
| Part B risk | TTC / DRAC from causal tracks → hand-set sigmoid → EMA | `src/risk/estimator.py` |

**Which classes are emitted** is decided in `configs/thresholds.yaml`. The metric is macro-F1 and a predicted
class absent from the test set scores 0, so a class is enabled only after its rule was checked by eye on the
sample videos. Enabled: `red_light`, `stop_line`, `jaywalking`, `failure_to_yield`, `stopped_vehicle`.
The other nine rules are implemented and unit-tested but stay off until validated (see `docs/PLAN.md`).

All thresholds live in `configs/*.yaml` or as named module constants with the sample-video evidence next to them.

## Scene calibration

```bash
python scripts/build_scene.py --videos data/samples/*.MP4 --ref-video data/samples/C3896.MP4   # alignments + auto movements
python scripts/draw_scene.py --scene configs/scene.json --videos data/samples/*.MP4             # overlays to check by eye
python scripts/eda.py --videos data/samples                                                     # resolution, fps, brightness, frames
```

## Data, models and licences

| Item | Source | Licence |
|---|---|---|
| Sample videos | WIUT Hackathon 2026 organizers | not redistributed (git-ignored) |
| YOLO11n weights (COCO-pretrained) | [ultralytics/assets v8.3.0](https://github.com/ultralytics/assets/releases/tag/v8.3.0) | AGPL-3.0 |
| COCO (detector pre-training) | cocodataset.org | CC BY 4.0 |
| ultralytics (YOLO, BoT-SORT, ByteTrack) | github.com/ultralytics/ultralytics | AGPL-3.0 |
| OpenCV | opencv.org | Apache-2.0 |

No hosted API is used at inference. No training was done yet; all models are used as released.

## Reproducibility

Seed 1234 (`configs/runtime.yaml`) for `random`, NumPy and torch; `cudnn.deterministic=True`; fixed frame
sampling by index; post-processing sorts and rounds outputs to ms.

## Team

| Member | Role / what they did |
|---|---|
| _to fill_ | |
