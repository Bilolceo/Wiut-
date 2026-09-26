"""illegal_u_turn, illegal_turn, solid_line_crossing on synthetic tracks."""
from __future__ import annotations

from conftest import add_track, line

from src.events.illegal_turn import IllegalTurnRule, classify_turn
from src.events.illegal_u_turn import IllegalUTurnRule
from src.events.solid_line_crossing import SolidLineCrossingRule
from src.tracks.store import TrackStore


def _store():
    return TrackStore(video_id="t", fps=10.0)


def _u_turn(x0=0.05, y=0.10):
    """Drive +x for 3 s, swing down 3 s, drive back -x for 3 s (a U-turn)."""
    return (
        line(0.0, 3.0, (x0, y), (x0 + 0.15, y))
        + line(3.1, 6.0, (x0 + 0.15, y + 0.005), (x0 + 0.15, y + 0.05))[1:]
        + line(6.1, 9.0, (x0 + 0.145, y + 0.05), (x0, y + 0.05))
    )


# -------------------------------------------------------------- illegal_u_turn
def test_u_turn_inside_no_u_turn_zone_is_flagged(make_scene):
    store = _store()
    add_track(store, 1, "car", _u_turn())
    cands = IllegalUTurnRule().run(store, make_scene())
    assert len(cands) == 1 and cands[0].label == "illegal_u_turn"


def test_u_turn_outside_the_zone_is_ignored(make_scene):
    store = _store()
    add_track(store, 1, "car", _u_turn(x0=0.60, y=0.60))
    assert IllegalUTurnRule().run(store, make_scene()) == []


def test_straight_drive_through_zone_is_ignored(make_scene):
    store = _store()
    add_track(store, 1, "car", line(0.0, 6.0, (0.02, 0.10), (0.28, 0.10)))
    assert IllegalUTurnRule().run(store, make_scene()) == []


# ---------------------------------------------------------------- illegal_turn
LANE_STRAIGHT_ONLY = [
    {"id": "lane_s", "polygon": [[0.0, 0.55], [0.4, 0.55], [0.4, 0.65], [0.0, 0.65]], "direction": [1.0, 0.0], "turn_allowed": ["straight"]}
]


def test_turn_classification_signs():
    assert classify_turn(5.0) == "straight"
    assert classify_turn(90.0) == "right"  # clockwise on screen (y down)
    assert classify_turn(-90.0) == "left"
    assert classify_turn(175.0) == "u_turn"


def test_right_turn_from_straight_only_lane_is_flagged(make_scene):
    store = _store()
    add_track(store, 1, "car", line(0.0, 3.0, (0.05, 0.60), (0.35, 0.60)) + line(3.1, 6.0, (0.35, 0.61), (0.35, 0.90)))
    cands = IllegalTurnRule().run(store, make_scene(lanes=LANE_STRAIGHT_ONLY))
    assert len(cands) == 1 and cands[0].evidence["manoeuvre"] == "right"


def test_straight_through_straight_only_lane_is_ignored(make_scene):
    store = _store()
    add_track(store, 1, "car", line(0.0, 6.0, (0.05, 0.60), (0.65, 0.60)))
    assert IllegalTurnRule().run(store, make_scene(lanes=LANE_STRAIGHT_ONLY)) == []


def test_unclassified_lane_never_flags(make_scene):
    lanes = [{**LANE_STRAIGHT_ONLY[0], "turn_allowed": ["unclassified"]}]
    store = _store()
    add_track(store, 1, "car", line(0.0, 3.0, (0.05, 0.60), (0.35, 0.60)) + line(3.1, 6.0, (0.35, 0.61), (0.35, 0.90)))
    assert IllegalTurnRule().run(store, make_scene(lanes=lanes)) == []


# --------------------------------------------------------- solid_line_crossing
def test_lane_change_across_solid_line_is_flagged(make_scene):
    store = _store()
    add_track(store, 1, "car", line(0.0, 4.0, (0.10, 0.46), (0.50, 0.54)))
    cands = SolidLineCrossingRule().run(store, make_scene())
    assert len(cands) == 1 and cands[0].label == "solid_line_crossing"


def test_jitter_straddling_the_line_is_ignored(make_scene):
    """Box-bottom jitter flipping sides every frame is not a lane change."""
    store = _store()
    add_track(store, 1, "car", [(i / 10, 0.10 + i * 0.005, 0.499 if i % 2 else 0.501) for i in range(40)])
    assert SolidLineCrossingRule().run(store, make_scene()) == []


def test_driving_parallel_to_solid_line_is_ignored(make_scene):
    store = _store()
    add_track(store, 1, "car", line(0.0, 4.0, (0.10, 0.45), (0.90, 0.45)))
    assert SolidLineCrossingRule().run(store, make_scene()) == []
