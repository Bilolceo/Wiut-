"""VLM verifier: confirm low-confidence candidates with Qwen3-VL-2B-Instruct.

accident / near_miss / fire_smoke / road_obstacle rules have high recall and
poor precision (on the accident-free samples they still propose ~100
candidates). The verifier shows the VLM a short clip around each candidate
and reads P("yes") from the next-token logits of a yes/no question -- one
forward pass, no sampling, deterministic, and a calibrated-ish confidence
instead of a parsed string.

Budget (docs/PLAN.md section 7): <= max_calls per video, strongest candidates
first; candidates beyond the budget stay unverified and are dropped by
post-processing (their rule confidence is below min_conf).

The backend is injected (duck-typed `.p_yes(images, question) -> float`) so
tests run with a fake and never load the 4.3 GB model.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.events.base import Candidate

log = logging.getLogger(__name__)

QUESTIONS = {
    "accident": "These frames are from a fixed traffic camera, in time order. Do two road users (vehicles, "
                "or a vehicle and a person) actually collide or make contact in this clip?",
    "near_miss": "These frames are from a fixed traffic camera, in time order. Does a vehicle brake hard or "
                 "swerve suddenly to avoid hitting another vehicle or a person, without contact?",
    "fire_smoke": "These frames are from a fixed traffic camera. Is there visible fire or smoke coming from a "
                  "vehicle or from something on the road?",
    "road_obstacle": "These frames are from a fixed traffic camera. Is there debris, a fallen object or an "
                     "animal lying on the road surface (not a vehicle, not a person, not a road marking)?",
}
SUFFIX = " Answer with one word: yes or no."


@dataclass
class ClipSpec:
    n_frames: int = 6
    pad_sec: float = 1.0
    width: int = 640
    min_crop_frac: float = 0.35  # crop at least this share of the frame around the evidence


def read_clip(video_path: str, start: float, end: float, spec: ClipSpec, crop: tuple | None = None) -> list[np.ndarray]:
    """`n_frames` evenly spaced BGR frames over [start - pad, end + pad], optionally cropped (x1, y1, x2, y2 normalized)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    times = np.linspace(max(0.0, start - spec.pad_sec), end + spec.pad_sec, spec.n_frames)
    frames = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(n_total - 1, int(round(t * fps))))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        if crop is not None:
            x1, y1, x2, y2 = crop
            frame = frame[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]
        s = spec.width / frame.shape[1]
        frames.append(cv2.resize(frame, (spec.width, max(1, round(frame.shape[0] * s))), interpolation=cv2.INTER_AREA))
    cap.release()
    return frames


def evidence_crop(candidate: Candidate, track_store, to_video, spec: ClipSpec) -> tuple | None:
    """Box around the candidate's tracks (reference coords -> this video's coords), padded to a readable size."""
    ids = candidate.evidence.get("track_ids") or [candidate.evidence.get("track_id")]
    pts = [
        to_video(p.x_norm, p.y_norm)
        for tid in ids if tid in track_store.tracks
        for p in track_store.tracks[tid].points_in_window(candidate.end, candidate.end - candidate.start)
    ]
    if not pts:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half = max(spec.min_crop_frac, max(xs) - min(xs) + 0.1, max(ys) - min(ys) + 0.1) / 2
    x1, y1 = min(max(0.0, cx - half), 1 - 2 * half), min(max(0.0, cy - half), 1 - 2 * half)
    return (max(0.0, x1), max(0.0, y1), min(1.0, x1 + 2 * half), min(1.0, y1 + 2 * half))


def verify_candidate(frames: list[np.ndarray], candidate: Candidate, backend, threshold: float = 0.5) -> tuple[bool, float]:
    """(accept, confidence) for one candidate. Unknown labels or an empty clip are rejected."""
    question = QUESTIONS.get(candidate.label)
    if question is None or not frames:
        return False, 0.0
    p = float(backend.p_yes(frames, question + SUFFIX))
    if not np.isfinite(p):
        # fp16 overflow inside the VLM (T4 has no bf16) shows up as NaN logits: never accept on garbage
        log.error("VLM returned non-finite P(yes) for %s %.1f-%.1fs; rejecting", candidate.label, candidate.start, candidate.end)
        return False, 0.0
    return p >= threshold, p


class Verifier:
    def __init__(self, backend, labels, max_calls: int = 30, threshold: float = 0.5, spec: ClipSpec | None = None):
        """`labels` in priority order: the budget goes to earlier labels first (accident before road_obstacle)."""
        self.labels = list(labels)
        self.backend, self.max_calls, self.threshold = backend, max_calls, threshold
        self.spec = spec or ClipSpec()
        self.calls = 0

    def filter(self, candidates: list[Candidate], video_path: str, track_store=None, to_video=None, deadline=None) -> list[Candidate]:
        """Pass through labels that need no verification; verify the rest within budget; drop rejected ones."""
        import time

        keep = [c for c in candidates if c.label not in self.labels]
        rank = {label: i for i, label in enumerate(self.labels)}
        todo = sorted((c for c in candidates if c.label in rank), key=lambda c: (rank[c.label], -c.conf, c.start))
        for c in todo:
            if self.calls >= self.max_calls or (deadline is not None and time.perf_counter() > deadline):
                break
            crop = evidence_crop(c, track_store, to_video, self.spec) if track_store is not None and to_video else None
            frames = read_clip(video_path, c.start, c.end, self.spec, crop)
            self.calls += 1
            accept, p = verify_candidate(frames, c, self.backend, self.threshold)
            log.info("verify %s %.1f-%.1fs: p_yes=%.2f -> %s", c.label, c.start, c.end, p, "accept" if accept else "reject")
            if accept:
                c.conf = p
                c.evidence["vlm_p_yes"] = round(p, 3)
                keep.append(c)
        return keep


class QwenVLBackend:
    """Qwen3-VL-2B-Instruct, SDPA attention; fp16 on CUDA (T4 has no bf16), fp16 on MPS, fp32 on CPU."""

    def __init__(self, model_dir: str | Path, device: str | None = None) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        from src.perception.detector import resolve_device

        self.device = resolve_device(device)
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir, dtype=dtype, attn_implementation="sdpa"
        ).to(self.device).eval()
        tok = self.processor.tokenizer
        self.yes_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes", " yes", " Yes")})
        self.no_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No", " no", " No")})

    def p_yes(self, images: list[np.ndarray], question: str) -> float:
        import torch
        from PIL import Image

        content = [{"type": "image", "image": Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))} for im in images]
        content.append({"type": "text", "text": question})
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1].float()
        yes = torch.logsumexp(logits[self.yes_ids], 0)
        no = torch.logsumexp(logits[self.no_ids], 0)
        return float(torch.sigmoid(yes - no))
