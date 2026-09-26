"""Candidates -> final [start, end, label] events.

Per class: drop disabled classes and weak/short candidates, merge overlapping
or near-adjacent segments (same-class segments must never overlap), clip to
[0, duration], round to ms, sort by (start, label).
"""
from __future__ import annotations

from collections import defaultdict

from src.events.base import Candidate, merge_intervals


def finalize(candidates: list[Candidate], duration_sec: float, thresholds: dict) -> list[list]:
    per_class = thresholds.get("classes", {})
    gap = float(thresholds.get("merge_gap_sec", 1.0))
    by_label: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for c in candidates:
        cfg = per_class.get(c.label, {})
        if not cfg.get("enabled", False) or c.conf < cfg.get("min_conf", 0.0):
            continue
        by_label[c.label].append((max(0.0, c.start), min(duration_sec, c.end)))

    events: list[list] = []
    for label, spans in by_label.items():
        min_dur = per_class[label].get("min_duration_sec", 0.0)
        for start, end in merge_intervals([s for s in spans if s[1] > s[0]], gap):
            start, end = round(start, 3), round(min(end, duration_sec), 3)
            if end - start >= min_dur and 0.0 <= start < end:
                events.append([start, end, label])
    return sorted(events, key=lambda e: (e[0], e[2]))
