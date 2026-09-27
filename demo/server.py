"""Live demo backend: upload a clip, get events, risk curve and an annotated playback.

Runs the same code as the submission (src/detect.py for Part A, the causal
RiskEstimator for Part B) on CPU. The VLM verifier is off by default (a 2B VLM
on CPU takes tens of seconds per call) and can be turned on per job with
`full=true`. One job runs at a time; the rest wait in a queue.

Also serves the static website from ../site, so a single container (e.g. a
Hugging Face Space) hosts both.

    uvicorn demo.server:app --host 0.0.0.0 --port 7860
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.detect import detect_events_impl, get_scene
from src.io.decode import read_meta
from src.risk.estimator import CausalRiskEstimator
from src.viz.render import Annotator, render_video, snapshot
from src.viz.series import downsample_max

log = logging.getLogger("demo")
ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("DEMO_WORK_DIR", "/tmp/wiut-demo"))
MAX_BYTES = int(os.environ.get("DEMO_MAX_MB", "600")) * 1024 * 1024
MAX_SEC = float(os.environ.get("DEMO_MAX_SEC", "120"))
KEEP_JOBS = 20
RENDER_WIDTH = 640


@dataclass
class Job:
    id: str
    full: bool
    status: str = "queued"  # queued | running | done | error
    stage: str = "waiting in queue"
    progress: float = 0.0
    error: str | None = None
    result: dict | None = None
    created: float = field(default_factory=time.time)


JOBS: dict[str, Job] = {}
LOCK = threading.Lock()
EXECUTOR = ThreadPoolExecutor(max_workers=1)

app = FastAPI(title="Traffic event detection demo")
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("DEMO_CORS", "*").split(","), allow_methods=["*"], allow_headers=["*"])


def _set(job: Job, **kw) -> None:
    with LOCK:
        for k, v in kw.items():
            setattr(job, k, v)


def _risk_curve(path: Path, job: Job) -> list[list[float]]:
    """Stream every frame through the causal RiskEstimator exactly like run_submission.py."""
    meta = read_meta(path)
    est = CausalRiskEstimator()
    est.reset({"video_id": path.name, "fps": meta.fps, "width": meta.width, "height": meta.height, "n_frames": meta.n_frames})
    cap, curve, idx = cv2.VideoCapture(str(path)), [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / meta.fps
        curve.append([t, est.step(frame, t)])
        if idx % 30 == 0:
            _set(job, progress=0.6 + 0.25 * min(1.0, t / max(meta.duration_sec, 1e-6)))
        idx += 1
    cap.release()
    return curve


def _run(job: Job, path: Path) -> None:
    out_dir = path.parent
    try:
        t0 = time.perf_counter()
        _set(job, status="running", stage="detecting and tracking road users")
        events, rep = detect_events_impl(
            str(path), overrides={"verifier": {"enabled": job.full}}, progress=lambda f: _set(job, progress=0.6 * f)
        )
        t_a = time.perf_counter()
        _set(job, stage="computing accident risk (Part B)", progress=0.6)
        risk = _risk_curve(path, job)
        t_b = time.perf_counter()

        _set(job, stage="rendering annotated playback", progress=0.85)
        meta = read_meta(path)
        ann = Annotator(get_scene(), rep["track_store"], rep["light_log"], events, rep["candidates"], risk, (meta.width, meta.height))
        render_video(str(path), ann, out_dir / "annotated.mp4", width=RENDER_WIDTH,
                     progress=lambda t: _set(job, progress=0.85 + 0.15 * min(1.0, t / max(meta.duration_sec, 1e-6))))
        examples = []
        for label in sorted({e[2] for e in events}):
            s, e, _ = next(ev for ev in events if ev[2] == label)
            img = snapshot(str(path), ann, (s + e) / 2, width=960)
            if img is not None:
                cv2.imwrite(str(out_dir / f"{label}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                examples.append({"label": label, "start": s, "end": e, "img": f"/api/jobs/{job.id}/files/{label}.jpg"})
        result = {
            "duration_sec": round(meta.duration_sec, 2), "width": meta.width, "height": meta.height, "fps": round(meta.fps, 2),
            "events": events, "risk": downsample_max(risk, 2.0), "examples": examples,
            "render": f"/api/jobs/{job.id}/files/annotated.mp4", "vlm": job.full,
            "timing_sec": {"part_a": round(t_a - t0, 1), "part_b": round(t_b - t_a, 1), "render": round(time.perf_counter() - t_b, 1)},
        }
        _set(job, status="done", stage="done", progress=1.0, result=result)
    except Exception as exc:  # the demo must never crash the server
        log.exception("job %s failed", job.id)
        _set(job, status="error", stage="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        path.unlink(missing_ok=True)  # keep outputs, drop the upload


def _cleanup() -> None:
    with LOCK:
        old = sorted(JOBS.values(), key=lambda j: j.created)[:-KEEP_JOBS]
        for job in old:
            if job.status in ("done", "error"):
                shutil.rmtree(WORK / job.id, ignore_errors=True)
                JOBS.pop(job.id, None)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "queued": sum(j.status == "queued" for j in JOBS.values()), "max_sec": MAX_SEC, "max_mb": MAX_BYTES // 2**20}


@app.post("/api/jobs")
async def create_job(video: UploadFile = File(...), full: bool = Form(False)) -> dict:
    job = Job(id=uuid.uuid4().hex[:12], full=full)
    job_dir = WORK / job.id
    job_dir.mkdir(parents=True, exist_ok=True)
    path, size = job_dir / "input.mp4", 0
    with path.open("wb") as fh:
        while chunk := await video.read(1 << 20):
            size += len(chunk)
            if size > MAX_BYTES:
                shutil.rmtree(job_dir, ignore_errors=True)
                raise HTTPException(413, f"File larger than {MAX_BYTES // 2**20} MB")
            fh.write(chunk)
    try:
        meta = read_meta(path)
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(422, "Not a readable video file")
    if meta.duration_sec > MAX_SEC + 0.5:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(422, f"Video is {meta.duration_sec:.0f} s; the demo accepts up to {MAX_SEC:.0f} s")
    _cleanup()
    with LOCK:
        JOBS[job.id] = job
    EXECUTOR.submit(_run, job, path)
    return {"id": job.id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")
    with LOCK:
        body = asdict(job)
    body["queue_position"] = sum(j.status == "queued" and j.created < job.created for j in JOBS.values())
    return body


@app.get("/api/jobs/{job_id}/files/{name}")
def get_file(job_id: str, name: str):
    if job_id not in JOBS or "/" in name or ".." in name or name == "input.mp4":
        raise HTTPException(404, "Not found")
    path = WORK / job_id / name
    if not path.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(path)


app.mount("/", StaticFiles(directory=ROOT / "site", html=True), name="site")
