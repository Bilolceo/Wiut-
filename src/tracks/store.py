"""Track store: per-video trajectory table, image-normalized + world metres."""
from __future__ import annotations

import bisect

from dataclasses import dataclass, field

VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle"}


@dataclass
class TrackPoint:
    frame_idx: int
    t_sec: float
    x_norm: float  # reference-frame-aligned, normalized [0,1] image coords
    y_norm: float
    x_m: float | None = None  # world metres via Scene.to_metres(); None if unavailable
    y_m: float | None = None
    w_norm: float = 0.0
    h_norm: float = 0.0
    conf: float = 1.0


@dataclass
class Track:
    track_id: int
    cls: str  # "car" | "bus" | "truck" | "motorcycle" | "bicycle" | "pedestrian" | ...
    points: list[TrackPoint] = field(default_factory=list)
    _times: list[float] = field(default_factory=list, repr=False, compare=False)

    def times(self) -> list[float]:
        """Sorted point timestamps, cached (points are only ever appended in time order)."""
        if len(self._times) != len(self.points):
            self._times = [p.t_sec for p in self.points]
        return self._times

    def __len__(self) -> int:
        return len(self.points)

    def duration_sec(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return self.points[-1].t_sec - self.points[0].t_sec

    def points_in_window(self, t_end: float, window_sec: float) -> list[TrackPoint]:
        times = self.times()
        lo = bisect.bisect_left(times, t_end - window_sec - 1e-9)
        hi = bisect.bisect_right(times, t_end + 1e-9)
        return self.points[lo:hi]

    def at(self, t: float, max_gap_sec: float = 0.25) -> TrackPoint | None:
        """The point observed nearest to time t, if within `max_gap_sec` (points are time-ordered)."""
        times = self.times()
        i = bisect.bisect_left(times, t)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(times) and abs(times[j] - t) <= max_gap_sec:
                if best is None or abs(times[j] - t) < abs(best.t_sec - t):
                    best = self.points[j]
        return best

    def is_stationary(self, t_end: float, window_sec: float, max_disp_m: float = 1.0) -> bool:
        """True if every point in [t_end - window_sec, t_end] stays within
        `max_disp_m` of the window's first point. Requires world coords and
        enough history to actually cover the window; returns False otherwise
        (conservative: no false positive from a track that just appeared)."""
        window = self.points_in_window(t_end, window_sec)
        if len(window) < 2:
            return False
        if window[0].t_sec > t_end - window_sec + 1e-6:
            return False  # track doesn't go back far enough yet
        if any(p.x_m is None for p in window):
            return False
        x0, y0 = window[0].x_m, window[0].y_m
        return all(((p.x_m - x0) ** 2 + (p.y_m - y0) ** 2) ** 0.5 <= max_disp_m for p in window)

    def speed_mps_at(self, t_end: float, lookback_sec: float = 1.0) -> float | None:
        """Average speed (m/s) over the last `lookback_sec` before t_end."""
        window = self.points_in_window(t_end, lookback_sec)
        if len(window) < 2 or any(p.x_m is None for p in window):
            return None
        a, b = window[0], window[-1]
        dt = b.t_sec - a.t_sec
        if dt <= 1e-6:
            return None
        d = ((b.x_m - a.x_m) ** 2 + (b.y_m - a.y_m) ** 2) ** 0.5
        return d / dt

    def image_heading_at(self, t_end: float, lookback_sec: float = 1.5):
        """Unit displacement vector in normalized image coords over the last
        `lookback_sec` before t_end -- comparable to scene lanes' `direction`
        (which is defined in the same image-space by build_scene.py), so this
        works even before a homography/world-metres calibration exists."""
        import numpy as np

        window = self.points_in_window(t_end, lookback_sec)
        if len(window) < 2:
            return None
        a, b = window[0], window[-1]
        v = np.array([b.x_norm - a.x_norm, b.y_norm - a.y_norm])
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-6 else None

    def heading_at(self, t_end: float, lookback_sec: float = 1.0):
        """Unit displacement vector (in metres, world frame) over the last
        `lookback_sec` before t_end, or None if not enough data."""
        import numpy as np

        window = self.points_in_window(t_end, lookback_sec)
        if len(window) < 2 or any(p.x_m is None for p in window):
            return None
        a, b = window[0], window[-1]
        v = np.array([b.x_m - a.x_m, b.y_m - a.y_m])
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-6 else None


@dataclass
class TrackStore:
    video_id: str
    fps: float
    tracks: dict[int, Track] = field(default_factory=dict)
    # Per-video outputs of frame-level analyzers (e.g. static-object and
    # fire/smoke monitors), keyed by analyzer name -- consumed by the rules
    # that are not trajectory-based (road_obstacle, fire_smoke).
    extras: dict = field(default_factory=dict)
    duration_sec: float = 0.0

    def add_point(self, track_id: int, cls: str, point: TrackPoint) -> None:
        tr = self.tracks.get(track_id)
        if tr is None:
            tr = Track(track_id=track_id, cls=cls)
            self.tracks[track_id] = tr
        tr.points.append(point)

    def vehicle_tracks(self) -> list[Track]:
        return [t for t in self.tracks.values() if t.cls in VEHICLE_CLASSES]

    def pedestrian_tracks(self) -> list[Track]:
        return [t for t in self.tracks.values() if t.cls == "pedestrian"]

    def sample_times(self) -> list[float]:
        """All distinct timestamps across every track, sorted -- the frames a
        rule should evaluate at."""
        times = {p.t_sec for t in self.tracks.values() for p in t.points}
        return sorted(times)
