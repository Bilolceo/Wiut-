# CLAUDE.md — WIUT Hackathon 2026, Computer Vision track

Full plan: `docs/PLAN.md` (read it before any non-trivial change).

## Task (short)
Fixed CCTV road camera. Implement in `solution.py`:
- `detect_events(video_path) -> [[start_sec, end_sec, label], ...]` (Part A, mandatory)
- `class RiskEstimator: reset(meta); step(frame_bgr, t_sec) -> float in [0,1]` = P(accident starts within 5 s) (Part B, causal)
- 14 labels exactly: accident, near_miss, red_light, wrong_way, illegal_u_turn, stopped_vehicle,
  jaywalking, failure_to_yield, illegal_turn, solid_line_crossing, stop_line, congestion, road_obstacle, fire_smoke

Score: Elim = 0.6*(0.7*A + 0.3*B) + 0.25*Website + 0.15*Code.
A = macro F1 over classes, temporal IoU 0.3/0.5/0.7. Predicting a class absent from test adds a 0 → only emit confident classes.

## Hard constraints (never break)
- Eval machine: 1x T4 16 GB (fp16; no bf16, no FlashAttention-2), 8 CPU, 32 GB RAM, Python >= 3.10, NO internet.
- Weights total <= 5 GB (`weights/download.sh`). Time per video (A+B) <= 3x duration; our target <= 1.5x.
- Open weights only. No OpenAI/Anthropic/Gemini/any hosted API at inference.
- `run_submission.py` and `evaluate.py` from the starter kit stay UNCHANGED.
- `RiskEstimator.step` uses only frames it received; never opens the file, never uses Part A output.
- Deterministic: fixed seeds, cudnn.deterministic=True, fixed frame sampling, greedy VLM decoding.
- Same-class segments must not overlap; 0 <= start < end <= duration. Never crash: catch, log, return [] / last score.

## Architecture (see docs/PLAN.md sections 3–9; current state)
decode (every 3rd frame ≈ 10 fps of 29.97 fps 4K; grab/retrieve, no seeking)
→ runtime camera alignment (src/scene/align.py vs configs/reference_frame.jpg, re-estimated every 10 s: the mount drifts)
→ YOLO11n @960 + BoT-SORT without GMC (configs/tracker_botsort.yaml) + traffic-light HSV with lamp-position check
  + dual-MOG2 static-object monitor → track store (reference-frame normalized coords + metres via homography)
→ 14 event rules (src/events/, registry.py) → post-processing (src/post/postprocess.py).
VLM verifier (Qwen3-VL-2B) is NOT implemented yet; accident/near_miss/fire_smoke are low-conf candidates, disabled.
Part B: src/risk/estimator.py — YOLO11n @640 every 3rd frame + ByteTrack → TTC/DRAC → hand sigmoid → EMA.
Enabled classes live in configs/thresholds.yaml (macro-F1: enable a class only after checking it by eye on video).
Scene geometry: configs/scene_manual.json (hand-drawn on C3896 @ 5 s) → configs/scene.json (+ alignments, auto_movements).

## Lessons from the sample videos (keep these in mind)
- Traffic-light lamp is ~0.2–1 % of its ROI in daylight: count lit pixels, never use a fraction-of-ROI floor.
- Left pole signal is a PEDESTRIAN head; the vehicle signal for the queue approach is the median head.
- Camera shifts up to ~140 px between recordings and ~15 px within one: always align at runtime.
- Full 8-DOF homography for alignment blows up with few matches (dusk vs day): use 4-DOF similarity.
- Check every rule change on real frames (contact sheets), not only on synthetic tests.

## Layout
solution.py (thin wrapper → src/), run_submission.py, evaluate.py, requirements.txt (pinned ==), Dockerfile,
weights/download.sh, configs/, src/{io,scene,perception,tracks,events,verify,post,risk,viz}/, scripts/, tools/scene_editor/,
labels/dev_gt.json, tests/, notebooks/ (EDA only), predictions_samples.json, README.md

## Conventions
- Python 3.10+, type hints, docstrings, ruff + black. Small modules, no dead code.
- Every event rule implements `EventRule.run(track_store, scene) -> list[Candidate]`.
- Default backend PyTorch fp16; TensorRT only behind a flag with automatic fallback.
- `BudgetGuard`: skip VLM + refine once 60% of the time budget is used.

## Commands
- pip install -r requirements.txt
- python run_submission.py --videos data/samples --out predictions.json
- python evaluate.py --pred predictions.json --validate-only
- python evaluate.py --pred predictions.json --gt labels/dev_gt.json
- pytest -q
- python scripts/draw_scene.py --scene configs/scene.json --videos data/samples/*.MP4   # geometry overlays

## Workflow rules for Claude Code
- Work module by module; after each module: run tests + validate-only, then stop and report.
- Never change run_submission.py / evaluate.py. Never add hosted-API calls to src/.
- Before adding a dependency, check it installs offline-safe and fits the 5 GB weights budget.
