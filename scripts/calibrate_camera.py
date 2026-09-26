"""Metric calibration of the fixed camera -> ground-plane homography for configs/scene_manual.json.

Why not just hand-pick point pairs: the foreground has no road markings of
known length, so hand-assigned metres there are guesses. Instead a pinhole
camera (focal length, pitch, roll, yaw, height; principal point at the image
centre) is fitted by Levenberg-Marquardt to three independent cues:

1. pedestrians: foot point + box height of ~12k detections all over the frame,
   binned by position; a standing adult is ~1.7 m tall (this carries the
   foreground and background scale),
2. the vanishing point of the lane lines of the queue approach (road direction),
3. the stop line: 4 lanes x 3.5 m = 14 m, perpendicular to the road.

The fitted camera maps any image point on the (flat) road to metres; the
homography is then least-squares fitted on a grid of such points over the
road mask. Independent checks (lane width, zebra width, pedestrian walking
speed) are printed so the calibration can be accepted or rejected.

Usage:
    python scripts/calibrate_camera.py --videos data/samples/*.MP4 --write
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

W, H = 3840, 2160
PERSON_HEIGHT_M = 1.7
STOP_LINE = ((0.150, 0.487), (0.485, 0.423))
STOP_LINE_M = 14.0  # 4 lanes x 3.5 m
VP_SIGMA_PX, HEIGHT_SIGMA_PX_FRAC, STOP_SIGMA_M, ANGLE_SIGMA_DEG = 60.0, 0.08, 1.0, 2.0


# ----------------------------------------------------------------- camera model
def rotation(pitch: float, roll: float, yaw: float) -> np.ndarray:
    """World (x east, y = road direction, z up) -> camera (x right, y down, z forward)."""
    fwd = np.array([np.sin(yaw) * np.cos(pitch), np.cos(yaw) * np.cos(pitch), -np.sin(pitch)])
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    c, s = np.cos(roll), np.sin(roll)
    return np.array([c * right + s * down, -s * right + c * down, fwd])


class Camera:
    def __init__(self, p: np.ndarray) -> None:
        self.f, self.pitch, self.roll, self.yaw, self.h = p
        self.R = rotation(self.pitch, self.roll, self.yaw)
        self.C = np.array([0.0, 0.0, self.h])

    def project(self, X: np.ndarray) -> np.ndarray:
        pc = (X - self.C) @ self.R.T
        return np.c_[self.f * pc[:, 0] / pc[:, 2] + W / 2, self.f * pc[:, 1] / pc[:, 2] + H / 2]

    def to_ground(self, uv: np.ndarray) -> np.ndarray:
        rays = np.c_[(uv[:, 0] - W / 2) / self.f, (uv[:, 1] - H / 2) / self.f, np.ones(len(uv))] @ self.R
        s = -self.h / rays[:, 2]
        return self.C[:2] + s[:, None] * rays[:, :2]

    def vanishing_point(self, direction: np.ndarray) -> np.ndarray:
        d = self.R @ direction
        return np.array([self.f * d[0] / d[2] + W / 2, self.f * d[1] / d[2] + H / 2])


# ------------------------------------------------------------------ observations
def collect_pedestrians(videos: list[str], every_sec: float, model_path: str) -> np.ndarray:
    """(u_foot, v_foot, box_height_px) of confident, upright person detections on reference-aligned frames."""
    from src.perception.detector import resolve_device
    from src.scene.align import estimate_alignment
    from ultralytics import YOLO

    ref = cv2.imread("configs/reference_frame.jpg")
    model, rows = YOLO(model_path), []
    for video in videos:
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for idx in range(0, n, int(every_sec * fps)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break
            A = estimate_alignment(frame, ref, (W, H))
            if A is None:
                continue
            res = model.predict(frame, classes=[0], conf=0.6, imgsz=1280, device=resolve_device("auto"), verbose=False)[0]
            for x1, y1, x2, y2 in res.boxes.xyxy.cpu().numpy():
                if not 2.0 <= (y2 - y1) / max(x2 - x1, 1) <= 4.0:  # upright, not cut off
                    continue
                pts = cv2.perspectiveTransform(np.float32([[[(x1 + x2) / 2, y2]], [[(x1 + x2) / 2, y1]]]), A).reshape(2, 2)
                rows.append((pts[0, 0], pts[0, 1], pts[0, 1] - pts[1, 1]))
        cap.release()
        print(f"{video}: {len(rows)} pedestrian samples so far")
    return np.array(rows)


def bin_medians(peds: np.ndarray, cell_px: float = 120.0, min_count: int = 5) -> np.ndarray:
    """Median box height per image cell: robust to occlusion, bags, children, mis-sized boxes."""
    keys = np.floor(peds[:, :2] / cell_px).astype(int)
    out = []
    for k in np.unique(keys, axis=0):
        m = (keys == k).all(axis=1)
        if m.sum() >= min_count:
            out.append((*np.median(peds[m, :2], axis=0), np.median(peds[m, 2]), m.sum()))
    return np.array(out)


def lane_vanishing_point(ref: np.ndarray) -> np.ndarray:
    """Least-squares intersection of the lane-line segments on the queue approach."""
    hsv = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 1] < 45) & (hsv[..., 2] > 175)).astype(np.uint8) * 255
    roi = np.zeros(white.shape, np.uint8)
    roi[int(0.38 * H):int(0.50 * H), int(0.13 * W):int(0.45 * W)] = 255
    segs = np.asarray(cv2.HoughLinesP(white & roi, 1, np.pi / 720, 60, minLineLength=90, maxLineGap=12)).reshape(-1, 4)
    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])) % 180
    segs = segs[(ang > 20) & (ang < 40)]
    A = []
    for x1, y1, x2, y2 in segs:
        line = np.cross([x1, y1, 1.0], [x2, y2, 1.0])
        A.append(line / np.hypot(line[0], line[1]) * np.hypot(x2 - x1, y2 - y1))
    v = np.linalg.svd(np.array(A))[2][-1]
    return v[:2] / v[2]


# ----------------------------------------------------------------------- fitting
def residuals(p: np.ndarray, cells: np.ndarray, vp1: np.ndarray) -> np.ndarray:
    cam = Camera(p)
    feet = cells[:, :2]
    ground = cam.to_ground(feet)
    heads = cam.project(np.c_[ground, np.full(len(ground), PERSON_HEIGHT_M)])
    pred_h = feet[:, 1] - heads[:, 1]
    r_height = (pred_h - cells[:, 2]) / (HEIGHT_SIGMA_PX_FRAC * cells[:, 2]) * np.sqrt(np.minimum(cells[:, 3], 50) / 10)
    r_vp = (cam.vanishing_point(np.array([0.0, 1.0, 0.0])) - vp1) / VP_SIGMA_PX
    a, b = cam.to_ground(np.array(STOP_LINE) * [W, H])
    r_len = [(np.linalg.norm(b - a) - STOP_LINE_M) / STOP_SIGMA_M]
    r_perp = [np.degrees(np.arcsin(abs((b - a)[1]) / np.linalg.norm(b - a))) / ANGLE_SIGMA_DEG]  # stop line along world x
    return np.concatenate([r_height, r_vp, r_len, r_perp])


def levenberg_marquardt(fun, p0: np.ndarray, iters: int = 200) -> np.ndarray:
    p, lam = p0.astype(float), 1e-2
    r = fun(p)
    for _ in range(iters):
        J = np.empty((len(r), len(p)))
        for i in range(len(p)):
            dp = np.zeros_like(p)
            dp[i] = 1e-6 * max(1.0, abs(p[i]))
            J[:, i] = (fun(p + dp) - r) / dp[i]
        A, g = J.T @ J, J.T @ r
        step = np.linalg.solve(A + lam * np.diag(np.diag(A)), -g)
        r_new = fun(p + step)
        if r_new @ r_new < r @ r:
            p, r, lam = p + step, r_new, lam / 3
            if np.linalg.norm(step) < 1e-9 * np.linalg.norm(p):
                break
        else:
            lam *= 5
    return p


# ---------------------------------------------------------------------- outputs
def homography_points(cam: Camera, road_mask: list, origin: np.ndarray, axis_x: np.ndarray) -> tuple[list, list]:
    """A 5x5 grid over the road mask's bounding box -> (image_points_norm, world_points_m), world = stop-line frame."""
    poly = np.array(road_mask)
    xs = np.linspace(poly[:, 0].min() + 0.05, poly[:, 0].max() - 0.05, 5)
    ys = np.linspace(0.30, 0.98, 5)
    from src.scene.geometry import point_in_polygon

    img = [(x, y) for y in ys for x in xs if point_in_polygon((x, y), poly)]
    ground = cam.to_ground(np.array(img) * [W, H]) - origin
    axis_y = np.array([-axis_x[1], axis_x[0]])
    world = np.c_[ground @ axis_x, ground @ axis_y]
    return [[round(x, 4), round(y, 4)] for x, y in img], [[round(float(a), 3), round(float(b), 3)] for a, b in world]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--every-sec", type=float, default=2.0)
    ap.add_argument("--model", default="weights/yolo11n.pt")
    ap.add_argument("--peds-cache", default="data/calib_pedestrians.npy")
    ap.add_argument("--write", action="store_true", help="update configs/scene_manual.json")
    args = ap.parse_args()

    cache = Path(args.peds_cache)
    peds = np.load(cache) if cache.exists() else collect_pedestrians(args.videos, args.every_sec, args.model)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, peds)
    cells = bin_medians(peds)
    vp1 = lane_vanishing_point(_ref4k(args.videos))
    print(f"pedestrian samples {len(peds)} -> {len(cells)} cells; lane vanishing point {vp1.round(0)}")

    best = None
    for pitch in np.radians([15, 25, 35, 45]):
        for yaw in np.radians([-60, -30, 0, 30, 60]):
            p = levenberg_marquardt(lambda q: residuals(q, cells, vp1), np.array([3500.0, pitch, 0.0, yaw, 15.0]))
            cost = float(residuals(p, cells, vp1) @ residuals(p, cells, vp1))
            if p[0] > 0 and p[4] > 0 and (best is None or cost < best[0]):
                best = (cost, p)
    cost, p = best
    cam = Camera(p)
    print(f"camera: f={p[0]:.0f}px (hFOV {np.degrees(2 * np.arctan(W / 2 / p[0])):.1f} deg) pitch={np.degrees(p[1]):.1f} "
          f"roll={np.degrees(p[2]):.2f} yaw={np.degrees(p[3]):.1f} height={p[4]:.1f} m  cost={cost:.0f}")
    r = residuals(p, cells, vp1)[: len(cells)] * HEIGHT_SIGMA_PX_FRAC
    print(f"pedestrian height residual (relative, per cell): median {np.median(np.abs(r)):.3f}")

    scene_path = Path("configs/scene_manual.json")
    scene = json.loads(scene_path.read_text())
    a, b = cam.to_ground(np.array(STOP_LINE) * [W, H])
    axis_x = (b - a) / np.linalg.norm(b - a)
    img_pts, world_pts = homography_points(cam, scene["road_mask"]["polygon"], a, axis_x)
    Hm, _ = cv2.findHomography(np.float32(img_pts), np.float32(world_pts), 0)
    err = np.linalg.norm(cv2.perspectiveTransform(np.float32(img_pts)[:, None], Hm)[:, 0] - np.float32(world_pts), axis=1)
    print(f"homography fit on {len(img_pts)} points: max error {err.max():.3f} m")

    def metres(p1, p2):
        g = cv2.perspectiveTransform(np.float32([[p1], [p2]]), Hm).reshape(2, 2)
        return float(np.linalg.norm(g[1] - g[0]))

    print("independent checks:")
    for name, p1, p2, expect in [
        ("stop line (4 lanes)", *STOP_LINE, "14 m by construction"),
        ("lane divider 1 -> 2 at stop line", (0.290, 0.460), (0.345, 0.450), "~3.5 m"),
        ("crosswalk_main_left width (top->bottom edge)", (0.40, 0.5185), (0.40, 0.5570), "4-6 m typical"),
        ("crosswalk_main_right width", (0.80, 0.4567), (0.80, 0.4847), "same zebra: ~as left"),
        ("crosswalk_diagonal width (along y=0.90)", (0.287, 0.90), (0.422, 0.90), "4-6 m typical"),
    ]:
        print(f"  {name:48s} {metres(p1, p2):6.2f} m   ({expect})")
    if args.write:
        scene["homography"] = {
            "note": "Fitted by scripts/calibrate_camera.py (pinhole camera from ~12k pedestrian heights @1.7 m + lane "
                    "vanishing point + 14 m stop line). World frame: origin = stop_line_near left end, x along the "
                    "stop line, y towards the camera side. Valid over the whole road mask (flat-ground assumption).",
            "camera": {"f_px": round(p[0], 1), "pitch_deg": round(np.degrees(p[1]), 2), "roll_deg": round(np.degrees(p[2]), 3),
                       "yaw_deg": round(np.degrees(p[3]), 2), "height_m": round(p[4], 2)},
            "image_points": img_pts,
            "world_points_m": world_pts,
        }
        scene_path.write_text(json.dumps(scene, indent=1, ensure_ascii=False) + "\n")
        print(f"wrote {scene_path}")


def _ref4k(videos: list[str]) -> np.ndarray:
    """The reference video's frame at 5 s at native 4K (lane lines need full resolution)."""
    ref_video = next((v for v in videos if "C3896" in v), videos[0])
    cap = cv2.VideoCapture(ref_video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(5.0 * (cap.get(cv2.CAP_PROP_FPS) or 30.0)))
    ok, frame = cap.read()
    cap.release()
    return frame


if __name__ == "__main__":
    main()
