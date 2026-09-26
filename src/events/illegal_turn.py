"""illegal_turn: a turn the lane does not allow (lane.turn_allowed).

The manoeuvre is classified from the track's own heading change between
entering and leaving: |angle| < STRAIGHT_MAX_DEG is "straight", beyond
UTURN_MIN_DEG is "u_turn", otherwise "left"/"right" by the sign (image
coords, y down: clockwise on screen = right turn). Lanes whose turn_allowed
is empty or "unclassified" (build_scene.py's default, until a human confirms
them) are skipped: the rule never guesses what is allowed.
"""
from __future__ import annotations

from src.events.base import Candidate
from src.scene.geometry import signed_angle

STRAIGHT_MAX_DEG = 30.0
UTURN_MIN_DEG = 150.0
HEADING_SEC = 1.5
MIN_TRACK_SEC = 4.0
KNOWN_MANOEUVRES = {"straight", "left", "right", "u_turn"}


def classify_turn(angle_deg: float) -> str:
    a = abs(angle_deg)
    if a < STRAIGHT_MAX_DEG:
        return "straight"
    if a >= UTURN_MIN_DEG:
        return "u_turn"
    return "right" if angle_deg > 0 else "left"


class IllegalTurnRule:
    label = "illegal_turn"

    def run(self, track_store, scene, light_log: dict | None = None) -> list[Candidate]:
        candidates: list[Candidate] = []
        for track in track_store.vehicle_tracks():
            if track.duration_sec() < MIN_TRACK_SEC:
                continue
            entry = next((p for p in track.points if scene.lane_at(p.x_norm, p.y_norm) is not None), None)
            if entry is None:
                continue
            lane = scene.lane_at(entry.x_norm, entry.y_norm)
            allowed = set(lane.turn_allowed) & KNOWN_MANOEUVRES
            if not allowed:
                continue
            t_in = entry.t_sec + HEADING_SEC
            t_out = track.points[-1].t_sec
            h_in = track.image_heading_at(t_in, HEADING_SEC)
            h_out = track.image_heading_at(t_out, HEADING_SEC)
            if h_in is None or h_out is None or t_out - t_in < HEADING_SEC:
                continue
            manoeuvre = classify_turn(signed_angle(h_in, h_out))
            if manoeuvre not in allowed:
                candidates.append(
                    Candidate(
                        entry.t_sec, t_out, self.label, conf=0.5,
                        evidence={"track_id": track.track_id, "lane": lane.id, "manoeuvre": manoeuvre},
                    )
                )
        return candidates
