"""Small helpers for time series shown on the website and in the demo."""
from __future__ import annotations


def downsample_max(curve: list[list[float]], hz: float) -> list[list[float]]:
    """[t, v] samples -> one point per 1/hz s window, keeping the window's max (alarms stay visible)."""
    out, bucket, t0 = [], [], None
    for t, v in curve:
        if t0 is None:
            t0 = t
        if t - t0 >= 1.0 / hz and bucket:
            out.append([round(t0, 2), round(max(bucket), 3)])
            bucket, t0 = [], t
        bucket.append(v)
    if bucket:
        out.append([round(t0, 2), round(max(bucket), 3)])
    return out
