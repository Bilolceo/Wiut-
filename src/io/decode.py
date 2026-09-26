"""Video decoding and frame sampling (CPU / NVDEC backends)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

TARGET_FPS = 10.0  # every 3rd frame of the 29.97 fps sample videos (docs/PLAN.md section 9)


@dataclass
class Frame:
    frame_idx: int  # index in the ORIGINAL (native-fps) frame sequence
    t_sec: float
    bgr: np.ndarray


@dataclass
class VideoMeta:
    fps: float
    width: int
    height: int
    n_frames: int
    duration_sec: float

    def as_dict(self, video_id: str) -> dict:
        return {"video_id": video_id, "fps": self.fps, "width": self.width, "height": self.height, "n_frames": self.n_frames}


def read_meta(video_path: str | Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = n_frames / fps if fps > 0 else 0.0
    return VideoMeta(fps=fps, width=width, height=height, n_frames=n_frames, duration_sec=duration)


def sample_frames(video_path: str | Path, target_fps: float = TARGET_FPS) -> Iterator[Frame]:
    """Yield frames at (approximately) `target_fps`.

    Decodes sequentially with no seeking (grab() every frame, retrieve()
    only the sampled ones) so behaviour is identical across whatever backend
    OpenCV picks -- seeking (CAP_PROP_POS_FRAMES) is not frame-exact on all
    codecs and would break determinism (CLAUDE.md: "fixed frame sampling").
    Skipping retrieve() saves the 4K colour conversion on unsampled frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(native_fps / target_fps))
    frame_idx = 0
    try:
        while True:
            if not cap.grab():
                break
            if frame_idx % stride == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                yield Frame(frame_idx=frame_idx, t_sec=frame_idx / native_fps, bgr=frame)
            frame_idx += 1
    finally:
        cap.release()
