"""Common EventRule interface + Candidate type used by every rule module."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Candidate:
    start: float
    end: float
    label: str
    conf: float
    evidence: dict = field(default_factory=dict)


class EventRule(Protocol):
    def run(self, track_store, scene, light_log: dict | None = None) -> list[Candidate]: ...


def flags_to_segments(
    flag_times: list[tuple[float, bool]],
    min_duration_sec: float = 0.5,
    merge_gap_sec: float = 1.0,
) -> list[tuple[float, float]]:
    """Turn a chronological (t_sec, is_flagged) series into merged [start, end]
    segments: drop blips shorter than `min_duration_sec`, merge segments whose
    gap is <= `merge_gap_sec`. Matches CLAUDE.md's post-processing convention
    so every rule produces comparable candidates."""
    flag_times = sorted(flag_times, key=lambda p: p[0])
    raw_segments: list[list[float]] = []
    open_start = None
    prev_t = None
    for t, flagged in flag_times:
        if flagged and open_start is None:
            open_start = t
        elif not flagged and open_start is not None:
            raw_segments.append([open_start, prev_t if prev_t is not None else t])
            open_start = None
        prev_t = t
    if open_start is not None:
        raw_segments.append([open_start, prev_t])

    merged: list[list[float]] = []
    for seg in raw_segments:
        if merged and seg[0] - merged[-1][1] <= merge_gap_sec:
            merged[-1][1] = seg[1]
        else:
            merged.append(seg)

    return [(s, e) for s, e in merged if e - s >= min_duration_sec]


def nearest_in_time(entries: list[tuple[float, Any]], t: float, max_gap_sec: float = 1.5) -> Any | None:
    """`entries`: chronological [(t_sec, value), ...] (e.g. a traffic light's
    smoothed state log). Returns the value whose timestamp is closest to `t`,
    or None if even the nearest entry is farther than `max_gap_sec` away --
    a sparse/misaligned log should abstain, never silently match a sample
    from a different moment."""
    if not entries:
        return None
    best_t, best_v = min(entries, key=lambda e: abs(e[0] - t))
    if abs(best_t - t) > max_gap_sec:
        return None
    return best_v


def merge_intervals(intervals: list[tuple[float, float]], gap_sec: float = 0.5) -> list[tuple[float, float]]:
    """Merge overlapping/near-adjacent (start, end) intervals -- for rules
    whose candidates come from discrete instant events (e.g. a line crossing)
    rather than a continuous flag series, so flags_to_segments doesn't apply."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1] + gap_sec:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]
