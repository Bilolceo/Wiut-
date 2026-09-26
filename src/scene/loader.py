"""Scene geometry: configs/scene.json loading, per-video alignment, homography."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _poly(points) -> np.ndarray:
    return np.array(points, dtype=np.float64)


@dataclass
class Zone:
    """A labelled polygon: used for crosswalks and no_u_turn_zones alike."""

    id: str
    polygon: np.ndarray


@dataclass
class StopLine:
    id: str
    line: np.ndarray
    controlled_by: str | None = None


@dataclass
class TrafficLight:
    id: str
    roi: tuple[float, float, float, float]
    lanes: list[str] = field(default_factory=list)
    kind: str = "vehicle"  # "vehicle" | "pedestrian"
    layout: str | None = None  # "vertical" enables the lamp-position check in classify_roi


@dataclass
class Lane:
    id: str
    polygon: np.ndarray
    direction: np.ndarray
    turn_allowed: list[str] = field(default_factory=list)


@dataclass
class Scene:
    reference_video: str
    crosswalks: list[Zone]
    stop_lines: list[StopLine]
    traffic_lights: list[TrafficLight]
    lanes: list[Lane]
    no_u_turn_zones: list[Zone]
    intersection: np.ndarray | None
    road_mask: np.ndarray | None
    homography: dict
    camera_alignment: dict
    raw: dict
    refuges: list[Zone] = field(default_factory=list)  # islands on the carriageway where pedestrians may wait
    solid_lines: list[Zone] = field(default_factory=list)  # polylines (Zone.polygon holds the vertices)
    bus_stops: list[Zone] = field(default_factory=list)  # vehicles dwelling here are neither jams nor breakdowns
    _H_cache: np.ndarray | None = field(default=None, init=False, repr=False)
    _H_computed: bool = field(default=False, init=False, repr=False)

    def homography_matrix(self) -> np.ndarray | None:
        """3x3 matrix mapping normalized [0,1] image coords -> world metres,
        built from homography.image_points / world_points_m (>=4 point pairs).
        Computed once and cached: to_metres() is called for every track point."""
        if self._H_computed:
            return self._H_cache
        self._H_computed = True
        pts_img = self.homography.get("image_points")
        pts_world = self.homography.get("world_points_m")
        if not pts_img or not pts_world or len(pts_img) < 4:
            return None
        import cv2

        src = np.array(pts_img, dtype=np.float32)
        dst = np.array(pts_world, dtype=np.float32)
        self._H_cache, _ = cv2.findHomography(src, dst, 0)
        return self._H_cache

    def to_metres(self, x_norm: float, y_norm: float) -> tuple[float, float] | None:
        H = self.homography_matrix()
        if H is None:
            return None
        p = H @ np.array([x_norm, y_norm, 1.0])
        if abs(p[2]) < 1e-9:
            return None
        return float(p[0] / p[2]), float(p[1] / p[2])

    def reference_size(self, fallback: tuple[int, int]) -> tuple[float, float]:
        """(w, h) in px of the reference frame the geometry was authored on."""
        w, h = self.raw.get("reference_frame_size") or fallback
        return float(w), float(h)

    def alignment_for(self, video_stem: str, video_size: tuple[int, int] | None = None) -> np.ndarray:
        """3x3 matrix mapping this video's native px -> reference video's native px.

        Precomputed per-video alignment if known (scripts/build_scene.py), else a
        pure rescale from this video's resolution to the reference resolution
        (identity for a same-size video).
        """
        per_video = (self.camera_alignment or {}).get("per_video", {})
        H = per_video.get(video_stem)
        if H:
            return np.array(H, dtype=np.float64)
        if video_size is None:
            return np.eye(3)
        ref_w, ref_h = self.reference_size(video_size)
        return np.diag([ref_w / video_size[0], ref_h / video_size[1], 1.0])

    def lane_at(self, x_norm: float, y_norm: float) -> Lane | None:
        """First lane polygon (in scene.json order) containing this point, or None."""
        from src.scene.geometry import point_in_polygon

        for lane in self.lanes:
            if point_in_polygon((x_norm, y_norm), lane.polygon):
                return lane
        return None


def load_scene(path: str | Path) -> Scene:
    raw = json.loads(Path(path).read_text())
    crosswalks = [Zone(c["id"], _poly(c["polygon"])) for c in raw.get("crosswalks", [])]
    stop_lines = [
        StopLine(s["id"], _poly(s["line"]), s.get("controlled_by")) for s in raw.get("stop_lines", [])
    ]
    traffic_lights = [
        TrafficLight(t["id"], tuple(t["roi"]), t.get("lanes", []), t.get("kind", "vehicle"), t.get("layout"))
        for t in raw.get("traffic_lights", [])
    ]
    lanes = [
        Lane(
            l["id"],
            _poly(l["polygon"]),
            np.array(l.get("direction", [0.0, 0.0]), dtype=np.float64),
            l.get("turn_allowed", []),
        )
        for l in raw.get("lanes", [])
    ]
    no_u_turn = [Zone(z["id"], _poly(z["polygon"])) for z in raw.get("no_u_turn_zones", [])]
    refuges = [Zone(z["id"], _poly(z["polygon"])) for z in raw.get("refuges", [])]
    solid_lines = [Zone(z["id"], _poly(z["line"])) for z in raw.get("solid_lines", [])]
    bus_stops = [Zone(z["id"], _poly(z["polygon"])) for z in raw.get("bus_stops", [])]
    intersection = _poly(raw["intersection"]["polygon"]) if raw.get("intersection") else None
    road_mask = _poly(raw["road_mask"]["polygon"]) if raw.get("road_mask") else None
    return Scene(
        reference_video=raw.get("reference_video", ""),
        crosswalks=crosswalks,
        stop_lines=stop_lines,
        traffic_lights=traffic_lights,
        lanes=lanes,
        no_u_turn_zones=no_u_turn,
        intersection=intersection,
        road_mask=road_mask,
        homography=raw.get("homography", {}),
        camera_alignment=raw.get("camera_alignment", {}),
        raw=raw,
        refuges=refuges,
        solid_lines=solid_lines,
        bus_stops=bus_stops,
    )
