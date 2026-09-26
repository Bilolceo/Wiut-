"""Road-user pairs present at the same instant, for the interaction rules (accident, near_miss)."""
from __future__ import annotations

from collections import defaultdict

from src.tracks.store import TrackPoint, TrackStore, Track


def points_by_time(track_store: TrackStore, classes: set[str]) -> dict[float, list[tuple[Track, TrackPoint]]]:
    """time -> [(track, point), ...] for tracks of the given classes observed at that sampled time."""
    index: dict[float, list[tuple[Track, TrackPoint]]] = defaultdict(list)
    for track in track_store.tracks.values():
        if track.cls in classes:
            for p in track.points:
                index[p.t_sec].append((track, p))
    return index


def world_distance(a: TrackPoint, b: TrackPoint) -> float | None:
    if a.x_m is None or b.x_m is None:
        return None
    return ((a.x_m - b.x_m) ** 2 + (a.y_m - b.y_m) ** 2) ** 0.5
