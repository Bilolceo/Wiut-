"""Annotated renders of a processed video (sample-video pages and the live demo).

Draws, on the frames the pipeline itself sampled: scene geometry (mapped
through the runtime camera alignment), every tracked road user, the road
users involved in an emitted event (vehicle red, pedestrian yellow), the
traffic-light state, the active events and the Part B risk.
Output is H.264 MP4 (browser-playable) via the ffmpeg binary bundled with
imageio-ffmpeg.
"""
from __future__ import annotations

import bisect
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.events.base import nearest_in_time
from src.io.decode import sample_frames
from src.tracks.store import VEHICLE_CLASSES

INVOLVED_VEHICLE = (40, 40, 230)  # BGR red
INVOLVED_PERSON = (0, 215, 255)  # BGR yellow
OTHER = (200, 200, 200)
LIGHT_BGR = {"red": (40, 40, 230), "yellow": (0, 200, 255), "green": (60, 190, 60), "unknown": (140, 140, 140)}
EVENT_BGR = {  # same hues as the website palette, BGR
    "red_light": (214, 120, 42), "stop_line": (52, 104, 235), "jaywalking": (122, 175, 27),
    "failure_to_yield": (0, 161, 237), "stopped_vehicle": (164, 123, 232), "congestion": (0, 131, 0),
    "wrong_way": (167, 58, 74), "accident": (72, 73, 227), "near_miss": (72, 73, 227), "road_obstacle": (137, 135, 129),
}


def ffmpeg_writer(path: str | Path, width: int, height: int, fps: float) -> subprocess.Popen:
    import imageio_ffmpeg

    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps:.3f}", "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def involved_tracks(candidates, events) -> dict[int, list[tuple[float, float, str]]]:
    """track_id -> [(start, end, label)] for candidates that made it into the emitted events."""
    emitted = defaultdict(list)
    for s, e, label in events:
        emitted[label].append((s, e))
    out: dict[int, list] = defaultdict(list)
    for c in candidates:
        if not any(s <= c.end and c.start <= e for s, e in emitted.get(c.label, [])):
            continue
        ids = [c.evidence.get("track_id")] + list(c.evidence.get("track_ids", [])) + list(c.evidence.get("pedestrian_ids", []))
        for tid in ids:
            if tid is not None:
                out[tid].append((c.start, c.end, c.label))
    return out


class Annotator:
    def __init__(self, scene, store, light_log: dict, events: list, candidates: list, risk: list, video_size: tuple[int, int]):
        self.scene, self.store, self.events = scene, store, events
        self.vw, self.vh = video_size
        self.ref_w, self.ref_h = scene.reference_size(video_size)
        self.light = next(iter(light_log.values()), []) if light_log else []
        self.risk_t = [t for t, _ in risk]
        self.risk_v = [v for _, v in risk]
        self.by_frame: dict[int, list] = defaultdict(list)
        for tr in store.tracks.values():
            for p in tr.points:
                self.by_frame[p.frame_idx].append((tr, p))
        self.involved = involved_tracks(candidates, events)
        al = store.extras.get("alignments") or [(0.0, scene.alignment_for(store.video_id, video_size).tolist())]
        self.align_t = [t for t, _ in al]
        self.align_inv = [np.linalg.inv(np.array(H)) for _, H in al]

    def _to_px(self, pts_ref_norm: np.ndarray, t: float, scale: float) -> np.ndarray:
        H_inv = self.align_inv[max(0, bisect.bisect_right(self.align_t, t) - 1)]
        pts = np.asarray(pts_ref_norm, dtype=np.float64).reshape(-1, 2) * [self.ref_w, self.ref_h]
        return (cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H_inv).reshape(-1, 2) * scale).round().astype(np.int32)

    def draw(self, img: np.ndarray, frame_idx: int, t: float) -> np.ndarray:
        scale = img.shape[1] / self.vw
        for cw in self.scene.crosswalks:
            cv2.polylines(img, [self._to_px(cw.polygon, t, scale)], True, (0, 200, 230), 1, cv2.LINE_AA)
        for sl in self.scene.stop_lines:
            a, b = self._to_px(sl.line, t, scale)
            cv2.line(img, tuple(a), tuple(b), (60, 60, 220), 2, cv2.LINE_AA)

        active = [(s, e, lab) for s, e, lab in self.events if s <= t <= e]
        for tr, p in self.by_frame.get(frame_idx, []):
            (cx, cy), = self._to_px([[p.x_norm, p.y_norm]], t, scale)
            w, h = p.w_norm * img.shape[1], p.h_norm * img.shape[0]
            box = (int(cx - w / 2), int(cy - h), int(cx + w / 2), int(cy))
            roles = [lab for s, e, lab in self.involved.get(tr.track_id, []) if s <= t <= e]
            if roles:
                color = INVOLVED_VEHICLE if tr.cls in VEHICLE_CLASSES else INVOLVED_PERSON
                cv2.rectangle(img, box[:2], box[2:], color, 2, cv2.LINE_AA)
                _label(img, roles[0].replace("_", " "), (box[0], box[1] - 4), color)
            else:
                cv2.rectangle(img, box[:2], box[2:], OTHER, 1, cv2.LINE_AA)

        # top banner: time, light, risk, active events
        cv2.rectangle(img, (0, 0), (img.shape[1], 26), (20, 20, 20), -1)
        _text(img, f"{int(t // 60)}:{t % 60:05.2f}", (8, 18))
        state = nearest_in_time(self.light, t, 1.5)
        state = state.state if state is not None else "unknown"
        cv2.circle(img, (100, 13), 7, LIGHT_BGR[state], -1, cv2.LINE_AA)
        _text(img, f"signal {state}", (112, 18))
        if self.risk_t:
            i = min(len(self.risk_v) - 1, bisect.bisect_left(self.risk_t, t))
            _text(img, f"risk {self.risk_v[i]:.2f}", (220, 18))
        x = 310
        for _, _, lab in active:
            x = _chip(img, lab.replace("_", " "), x, EVENT_BGR.get(lab, (180, 180, 180)))
        return img


def _text(img, s, org, color=(235, 235, 235), scale=0.5):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _label(img, s, org, color):
    (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    x, y = max(0, org[0]), max(th + 4, org[1])
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 6, y + 2), color, -1)
    cv2.putText(img, s, (x + 3, y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (15, 15, 15), 1, cv2.LINE_AA)


def _chip(img, s, x, color) -> int:
    (tw, _), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(img, (x, 5), (x + tw + 10, 21), color, -1)
    cv2.putText(img, s, (x + 5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return x + tw + 16


def render_video(video_path: str, annotator: Annotator, out_path: str | Path, width: int = 960, fps: float = 10.0,
                 t_range: tuple[float, float] | None = None, progress=None) -> None:
    """Annotated H.264 MP4 of the whole video (or t_range) at the pipeline's own sampling rate."""
    writer, size = None, None
    for frame in sample_frames(video_path, target_fps=fps):
        if t_range and frame.t_sec < t_range[0]:
            continue
        if t_range and frame.t_sec > t_range[1]:
            break
        if size is None:
            size = (width, round(frame.bgr.shape[0] * width / frame.bgr.shape[1]) // 2 * 2)
            writer = ffmpeg_writer(out_path, *size, fps)
        img = annotator.draw(cv2.resize(frame.bgr, size, interpolation=cv2.INTER_AREA), frame.frame_idx, frame.t_sec)
        writer.stdin.write(img.tobytes())
        if progress is not None and frame.frame_idx % 30 == 0:
            progress(frame.t_sec)
    if writer is not None:
        writer.stdin.close()
        writer.wait()


def snapshot(video_path: str, annotator: Annotator, t: float, stride_fps: float = 10.0, width: int = 1280) -> np.ndarray | None:
    """Annotated still at the pipeline-sampled frame nearest to t (so its tracks exist)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(fps / stride_fps))
    idx = int(round(t * fps / stride)) * stride
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    img = cv2.resize(frame, (width, round(frame.shape[0] * width / frame.shape[1])), interpolation=cv2.INTER_AREA)
    return annotator.draw(img, idx, idx / fps)
