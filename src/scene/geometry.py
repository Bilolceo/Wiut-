"""Small dependency-free 2D geometry helpers used by scene/ and events/."""
from __future__ import annotations

import numpy as np


def point_in_polygon(point: tuple[float, float], polygon: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test. polygon: (N,2) array, not necessarily closed."""
    x, y = point
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    x1, y1 = polygon[-1]
    for i in range(n):
        x2, y2 = polygon[i]
        if (y1 > y) != (y2 > y):
            x_int = x1 + (y - y1) * (x2 - x1) / (y2 - y1 + 1e-12)
            if x < x_int:
                inside = not inside
        x1, y1 = x2, y2
    return inside


def point_segment_distance(point: tuple[float, float], seg: np.ndarray) -> float:
    """Distance from point to a 2-point line segment ``seg`` (shape (2,2))."""
    p = np.asarray(point, dtype=np.float64)
    a, b = seg[0], seg[1]
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-12:
        return float(np.linalg.norm(p - a))
    t = np.clip(((p - a) @ ab) / denom, 0.0, 1.0)
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def signed_distance_to_line(point: tuple[float, float], line: np.ndarray) -> float:
    """Signed perpendicular distance from `point` to the infinite line through
    `line` (shape (2,2), a->b). Sign flips across the line -- used to tell
    which side of a stop line a vehicle is on without caring about lane
    direction metadata. 0.0 for a degenerate (zero-length) line."""
    a = np.asarray(line[0], dtype=np.float64)
    b = np.asarray(line[1], dtype=np.float64)
    p = np.asarray(point, dtype=np.float64)
    ab = b - a
    n = float(np.linalg.norm(ab))
    if n < 1e-12:
        return 0.0
    cross = ab[0] * (p[1] - a[1]) - ab[1] * (p[0] - a[0])
    return float(cross / n)


def segments_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> bool:
    """True if segment p1-p2 crosses segment p3-p4 (standard CCW-orientation
    test). Used to detect a vehicle's frame-to-frame motion crossing a stop
    line, independent of the line's own orientation."""

    def ccw(a, b, c) -> bool:
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def point_polygon_distance(point: tuple[float, float], polygon: np.ndarray) -> float:
    """0.0 inside the polygon, else the distance to its nearest edge."""
    if point_in_polygon(point, polygon):
        return 0.0
    n = len(polygon)
    return min(point_segment_distance(point, np.array([polygon[i], polygon[(i + 1) % n]])) for i in range(n))


def in_any_zone(point: tuple[float, float], zones, margin: float = 0.0) -> bool:
    """True if `point` is inside (or within `margin` of) any zone's polygon."""
    return any(point_polygon_distance(point, z.polygon) <= margin for z in zones)


def polyline_segments(points: np.ndarray):
    """Consecutive (a, b) vertex pairs of an open polyline."""
    return [(points[i], points[i + 1]) for i in range(len(points) - 1)]


def signed_angle(u: np.ndarray, v: np.ndarray) -> float:
    """Signed angle in degrees from u to v (image coords, y down: positive = clockwise on screen)."""
    return float(np.degrees(np.arctan2(u[0] * v[1] - u[1] * v[0], u[0] * v[0] + u[1] * v[1])))
