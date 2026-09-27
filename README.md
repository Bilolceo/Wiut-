# WIUT Hackathon 2026 — Computer Vision track

Traffic events from a fixed CCTV road camera (Tashkent intersection, 4K @ 29.97 fps):

- **Part A** `detect_events(video)` → `[[start_sec, end_sec, label], ...]` over 14 event classes
- **Part B** `RiskEstimator.step(frame, t)` → causal P(accident starts within 5 s)

Full design and decisions: [`docs/PLAN.md`](docs/PLAN.md). Organizers' starter-kit notes: [`docs/STARTER_KIT.md`](docs/STARTER_KIT.md).

## Install and run

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # pinned ==; Linux wheel of torch includes CUDA (T4)
./weights/download.sh                      # yolo11n.pt (sha256, also committed) + Qwen3-VL-2B (pinned revision); total 4.27 GB <= 5 GB
python run_submission.py --videos data/samples --out predictions.json
python evaluate.py --pred predictions.json --validate-only
pytest -q                                  # 91 unit tests, no video / GPU / VLM weights needed
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
   ─► 14 event rules (src/events/, one module per class)
   ─► VLM verifier (Qwen3-VL-2B, ≤30 calls/video) for accident / near_miss / fire_smoke / road_obstacle
   ─► post-processing (enabled classes, min conf / duration, same-class merge, clip) ─► events
```

| Component | Learned or rule-based | Where |
|---|---|---|
| Road-user detection + tracking | learned (YOLO11n, COCO) + BoT-SORT | `src/perception/detector.py` |
| Camera alignment across recordings | classical (ORB, RANSAC similarity) | `src/scene/align.py` |
| Scene geometry (zebras, stop line, islands, lanes) | hand-drawn once on the reference frame | `configs/scene_manual.json` |
| Traffic-light colour | classical (HSV + lamp position) | `src/perception/traffic_light.py` |
| Event classes | rules on trajectories, geometry and light state | `src/events/*.py` |
| Metric calibration | fitted pinhole camera (pedestrian heights, lane vanishing point, stop line) | `scripts/calibrate_camera.py` |
| Candidate verification | learned (Qwen3-VL-2B-Instruct, P(yes) from next-token logits) | `src/verify/vlm.py` |
| Part B risk | TTC / DRAC from causal tracks → sigmoid → persistence → EMA | `src/risk/estimator.py` |

**Which classes are emitted** is decided in `configs/thresholds.yaml`. The metric is macro-F1 and a predicted
class absent from the test set scores 0, so a class is enabled only after its rule was checked by eye on the
sample videos. Enabled (10): `red_light`, `stop_line`, `jaywalking`, `failure_to_yield`, `stopped_vehicle`,
`congestion`, `wrong_way`, plus `accident`, `near_miss`, `road_obstacle` which are only emitted after the VLM
verifier confirms them (it rejected all 98 false candidates on the accident-free samples). Off until validated:
`illegal_turn` (lane turn permissions unknown), `illegal_u_turn`, `solid_line_crossing`, `fire_smoke`.

All thresholds live in `configs/*.yaml` or as named module constants with the sample-video evidence next to them.

## Scene calibration

```bash
python scripts/build_scene.py --videos data/samples/*.MP4 --ref-video data/samples/C3896.MP4   # alignments + auto movements
python scripts/draw_scene.py --scene configs/scene.json --videos data/samples/*.MP4             # overlays to check by eye
python scripts/eda.py --videos data/samples                                                     # resolution, fps, brightness, frames
python scripts/calibrate_camera.py --videos data/samples/*.MP4 --write                          # metric homography
python scripts/build_site_data.py --predictions predictions_samples.json --videos data/samples/*.MP4   # website data
```

## Data, models and licences

| Item | Source | Licence |
|---|---|---|
| Sample videos | WIUT Hackathon 2026 organizers | not redistributed (git-ignored) |
| YOLO11n weights (COCO-pretrained) | [ultralytics/assets v8.3.0](https://github.com/ultralytics/assets/releases/tag/v8.3.0) | AGPL-3.0 |
| Qwen3-VL-2B-Instruct (verifier) | [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) @ `89644892` | Apache-2.0 |
| COCO (detector pre-training) | cocodataset.org | CC BY 4.0 |
| ultralytics (YOLO, BoT-SORT, ByteTrack) | github.com/ultralytics/ultralytics | AGPL-3.0 |
| OpenCV | opencv.org | Apache-2.0 |
| transformers | github.com/huggingface/transformers | Apache-2.0 |

No hosted API is used at inference. No training was done yet; all models are used as released.

## Live demo and annotated renders

Live: **https://wiut-production.up.railway.app** (website + demo API, CPU).

```bash
pip install -r demo/requirements.txt
uvicorn demo.server:app --port 7860          # API + website on http://localhost:7860
python scripts/render_samples.py --videos data/samples/*.MP4   # site/assets/renders/*.mp4 + example/failure stills
```

`demo/server.py` (FastAPI) accepts an MP4 up to 2 minutes / 600 MB, queues it, and runs the submission code on
CPU: Part A (VLM verifier off unless the visitor ticks "full analysis"), the causal Part B estimator, then an
annotated playback rendered by `src/viz/render.py` (the same renderer as the sample videos). Measured on an
Apple M4 forced to CPU (`WIUT_DEVICE=cpu`): a 40 s 1080p clip takes 61 s end to end. `demo/Dockerfile` targets a
CPU host such as a Hugging Face Space (Docker SDK, port 7860).

Failure cases shown on the site are curated in `docs/failure_cases.json`; each was checked on the frame.

## Website

`site/` is a static page (no build step) deployed by `.github/workflows/pages.yml` to GitHub Pages
(one-time: Settings → Pages → Source: GitHub Actions). Local preview: `python -m http.server -d site`.

## Reproducibility

Seed 1234 (`configs/runtime.yaml`) for `random`, NumPy and torch; `cudnn.deterministic=True`; fixed frame
sampling by index; post-processing sorts and rounds outputs to ms.

## Team

| Member | Role / what they did |
|---|---|
| Bilol Pardabaev | Developer: CV pipeline, event rules, Part B risk model, live demo backend |
| Shirin Axmadova | Web design: website layout and presentation |
| Rahmim Berdibaev | Pitch: project presentation |
