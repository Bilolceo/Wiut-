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

## Architecture (see docs/PLAN.md sections 3–9)
decode (every 2nd frame ≈ 15 fps; source is 29.97 fps) → perception (YOLO11m + BoT-SORT, traffic-light HSV, MOG2 fwd/bwd, open-vocab fire/smoke)
→ track store (metres via homography) → event rules (one module per class) → VLM verifier
(Qwen3-VL-2B-Instruct fp16, only on candidate windows, <= 30 calls/video) → post-processing
(hysteresis, merge < 1.5 s, min duration, native 29.97 fps boundary refine, per-class thresholds).
Part B: YOLO11s every 3rd frame + ByteTrack → TTC/PET/DRAC features → LightGBM/sigmoid calibrator → EMA.
Scene geometry lives in `configs/scene.json`; all thresholds in `configs/*.yaml` (no magic numbers in code).

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

## Workflow rules for Claude Code
- Work module by module; after each module: run tests + validate-only, then stop and report.
- Never change run_submission.py / evaluate.py. Never add hosted-API calls to src/.
- Before adding a dependency, check it installs offline-safe and fits the 5 GB weights budget.
