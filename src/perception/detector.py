"""Detection + tracking: YOLO11m + BoT-SORT, one persistent tracker per video.

Kept as a thin, swappable wrapper (duck-typed `track_frame`) so
src/perception/pipeline.py and tests can inject a fake in place of the real
model -- CLAUDE.md's "no dead code / small modules" plus keeping the heavy
ultralytics import out of anything that doesn't need it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# COCO class id -> our vehicle/pedestrian vocabulary. Only these are tracked;
# everything else (traffic light box, stop sign, ...) is ignored here and
# handled by dedicated perception (traffic_light.py) or scene geometry.
COCO_CLASS_NAMES: dict[int, str] = {
    0: "pedestrian",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class Detection:
    track_id: int
    cls: str
    cx_norm: float  # bottom-centre x, normalized to THIS frame's own width
    cy_norm: float  # bottom-centre y, normalized to THIS frame's own height
    w_norm: float
    h_norm: float
    conf: float


def resolve_device(device: str | None) -> str:
    """"auto"/None -> cuda if available, else mps (Apple dev machines), else cpu."""
    if device not in (None, "auto"):
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Perception:
    """Wraps one ultralytics YOLO model with persistent BoT-SORT tracking.

    One instance per video: BoT-SORT keeps internal track-id state across
    calls (`persist=True`), so instances must not be shared between videos.
    """

    def __init__(
        self,
        model_path: str = "weights/yolo11n.pt",
        conf: float = 0.3,
        tracker: str = "botsort.yaml",
        device: str | None = None,
        imgsz: int = 960,
    ) -> None:
        os.environ.setdefault("YOLO_OFFLINE", "1")  # eval machine has no internet: never try to reach it
        from ultralytics import YOLO  # deferred: heavy import, only when actually detecting

        self.model = YOLO(model_path)
        self.conf = conf
        self.tracker = tracker
        self.device = resolve_device(device)
        self.half = self.device.startswith("cuda")  # T4: fp16; MPS/CPU stay fp32
        self.imgsz = imgsz
        self._classes = list(COCO_CLASS_NAMES.keys())

    def track_frame(self, frame_bgr: np.ndarray) -> list[Detection]:
        result = self.model.track(
            frame_bgr,
            persist=True,
            classes=self._classes,
            tracker=self.tracker,
            conf=self.conf,
            device=self.device,
            half=self.half,
            imgsz=self.imgsz,
            verbose=False,
        )[0]
        out: list[Detection] = []
        if result.boxes is None or result.boxes.id is None:
            return out
        h, w = frame_bgr.shape[:2]
        xyxy = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)
        clss = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), tid, cls_id, conf in zip(xyxy, ids, clss, confs):
            name = COCO_CLASS_NAMES.get(int(cls_id))
            if name is None:
                continue
            out.append(
                Detection(
                    track_id=int(tid),
                    cls=name,
                    cx_norm=float((x1 + x2) / 2.0 / w),
                    cy_norm=float(y2 / h),  # ground-contact point, not box centre
                    w_norm=float((x2 - x1) / w),
                    h_norm=float((y2 - y1) / h),
                    conf=float(conf),
                )
            )
        return out
