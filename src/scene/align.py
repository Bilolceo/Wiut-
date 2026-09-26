"""Camera alignment of a video onto the scene's reference frame.

The camera mount shifts a little between recordings (C3902: ~137 px at 4K),
while the scene geometry is authored once on the reference frame. Test
videos are unknown in advance, so the alignment is estimated at runtime from
the first frame against a stored copy of the reference frame
(configs/reference_frame.jpg).

A 4-DOF similarity (shift + scale + rotation) is fitted, not a full
homography: with few matches (day vs dusk) a full homography overfits and
throws the frame corners thousands of pixels away. CLAHE makes ORB matching
robust to the lighting change.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)

MATCH_WIDTH = 1920  # matching resolution: plenty of ORB features, ~100 ms per frame
MAX_CORNER_SHIFT = 0.10  # reject fits moving a corner by more than 10 % of the width
MIN_INLIERS = 15


def _gray(frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = frame_bgr.shape[:2]
    s = MATCH_WIDTH / w
    small = cv2.resize(frame_bgr, (MATCH_WIDTH, round(h * s)), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)), s


def estimate_alignment(frame_bgr: np.ndarray, ref_bgr: np.ndarray, ref_native_size: tuple[int, int]) -> np.ndarray | None:
    """3x3 matrix mapping this frame's native px -> reference native px, or None if unreliable.

    `ref_bgr` may be a downscaled copy of the reference frame; `ref_native_size`
    (w, h) is the reference video's native resolution the geometry refers to.
    """
    g1, s1 = _gray(frame_bgr)
    g2, _ = _gray(ref_bgr)
    orb = cv2.ORB_create(nfeatures=8000)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < MIN_INLIERS or len(k2) < MIN_INLIERS:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < MIN_INLIERS:
        return None
    src = np.float32([k1[m.queryIdx].pt for m in good])
    dst = np.float32([k2[m.trainIdx].pt for m in good])
    A, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    if A is None or int(inliers.sum()) < MIN_INLIERS:
        return None

    # small-frame px -> small-ref px  ==>  native px -> ref native px
    ref_scale = MATCH_WIDTH / ref_native_size[0]
    H_small = np.vstack([A, [0.0, 0.0, 1.0]])
    H = np.diag([1 / ref_scale, 1 / ref_scale, 1.0]) @ H_small @ np.diag([s1, s1, 1.0])

    h, w = frame_bgr.shape[:2]
    corners = np.float32([[0, 0], [w, 0], [0, h], [w, h]]).reshape(-1, 1, 2)
    moved = cv2.perspectiveTransform(corners, H).reshape(-1, 2) * [w / ref_native_size[0], h / ref_native_size[1]]
    if np.max(np.linalg.norm(moved - corners.reshape(-1, 2), axis=1)) > MAX_CORNER_SHIFT * w:
        return None
    log.info("alignment: %d/%d inliers, shift (%.0f, %.0f) px", int(inliers.sum()), len(good), H[0, 2], H[1, 2])
    return H
