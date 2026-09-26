"""jaywalking + failure_to_yield on synthetic tracks (cases mirror what the sample videos showed)."""
from __future__ import annotations

from conftest import add_track, line

from src.events.failure_to_yield import FailureToYieldRule
from src.events.jaywalking import JaywalkingRule
from src.tracks.store import TrackStore


def _store():
    return TrackStore(video_id="t", fps=10.0)


# ------------------------------------------------------------------ jaywalking
def test_jaywalking_flags_pedestrian_crossing_mid_road(make_scene):
    store = _store()
    add_track(store, 1, "pedestrian", line(0.0, 4.0, (0.10, 0.20), (0.10, 0.28)))  # 2 m/s, far from any zebra
    cands = JaywalkingRule().run(store, make_scene())
    assert len(cands) == 1 and cands[0].label == "jaywalking"
    assert cands[0].end - cands[0].start >= 2.0


def test_jaywalking_ignores_pedestrian_on_crosswalk(make_scene):
    store = _store()
    add_track(store, 1, "pedestrian", line(0.0, 4.0, (0.42, 0.43), (0.58, 0.43)))
    assert JaywalkingRule().run(store, make_scene()) == []


def test_jaywalking_ignores_walking_along_zebra_edge(make_scene):
    """Feet 1 m outside the zebra polygon (seen on the samples) is still 'on the crossing'."""
    store = _store()
    add_track(store, 1, "pedestrian", line(0.0, 4.0, (0.42, 0.47), (0.58, 0.47)))
    assert JaywalkingRule().run(store, make_scene()) == []


def test_jaywalking_ignores_refuge_island(make_scene):
    store = _store()
    add_track(store, 1, "pedestrian", line(0.0, 5.0, (0.82, 0.85), (0.88, 0.85)))
    assert JaywalkingRule().run(store, make_scene()) == []


def test_jaywalking_ignores_brief_step_onto_road(make_scene):
    store = _store()
    add_track(store, 1, "pedestrian", line(0.0, 1.0, (0.10, 0.20), (0.10, 0.22)))
    assert JaywalkingRule().run(store, make_scene()) == []


def test_jaywalking_ignores_low_confidence_standing_detection(make_scene):
    """The moped rider waiting at the stop line: 'person', static, conf ~0.33."""
    store = _store()
    add_track(store, 1, "pedestrian", line(0.0, 15.0, (0.10, 0.20), (0.10, 0.20)), conf=0.33)
    assert JaywalkingRule().run(store, make_scene()) == []


def test_jaywalking_ignores_rider_on_bicycle(make_scene):
    store = _store()
    path = line(0.0, 4.0, (0.10, 0.20), (0.10, 0.28))
    add_track(store, 1, "pedestrian", path)
    add_track(store, 2, "bicycle", path)
    assert JaywalkingRule().run(store, make_scene()) == []


def test_jaywalking_abstains_without_road_mask(make_scene):
    store = _store()
    add_track(store, 1, "pedestrian", line(0.0, 4.0, (0.10, 0.20), (0.10, 0.28)))
    assert JaywalkingRule().run(store, make_scene(road_mask=None)) == []


# ------------------------------------------------------------ failure_to_yield
def _ped_on_zebra(store, t0=1.5, t1=4.5, x=0.50):
    """Walks across the zebra at ~1.7 m/s, mid-crossing when the car passes x at t=3 s."""
    add_track(store, 10, "pedestrian", line(t0, t1, (x, 0.405), (x, 0.455)))


def test_failure_to_yield_flags_car_driving_at_pedestrian(make_scene):
    store = _store()
    _ped_on_zebra(store)
    # car heading +x at 10 m/s, crosses the zebra at y=0.43 passing the pedestrian at x=0.50
    add_track(store, 1, "car", line(0.0, 6.0, (0.20, 0.43), (0.80, 0.43)))
    cands = FailureToYieldRule().run(store, make_scene())
    assert len(cands) == 1 and cands[0].evidence["pedestrian_ids"] == [10]


def test_failure_to_yield_ignores_car_when_crossing_is_empty(make_scene):
    store = _store()
    add_track(store, 1, "car", line(0.0, 6.0, (0.20, 0.43), (0.80, 0.43)))
    assert FailureToYieldRule().run(store, make_scene()) == []


def test_failure_to_yield_ignores_pedestrian_behind_the_car(make_scene):
    """Car already past the pedestrian's position when both are on the zebra."""
    store = _store()
    _ped_on_zebra(store, x=0.41)
    add_track(store, 1, "car", line(0.0, 3.0, (0.45, 0.43), (0.75, 0.43)))
    assert FailureToYieldRule().run(store, make_scene()) == []


def test_failure_to_yield_ignores_pedestrian_outside_vehicle_lane(make_scene):
    """Pedestrian on the same zebra but 5 m to the side of the car's line of travel."""
    store = _store()
    add_track(store, 10, "pedestrian", line(0.0, 6.0, (0.55, 0.405), (0.55, 0.405)))
    add_track(store, 1, "car", line(0.0, 6.0, (0.20, 0.455), (0.80, 0.455)))
    assert FailureToYieldRule().run(store, make_scene()) == []


def test_failure_to_yield_ignores_yielding_car(make_scene):
    """Car creeping at 0.5 m/s onto the zebra is yielding, not failing to."""
    store = _store()
    _ped_on_zebra(store)
    add_track(store, 1, "car", line(0.0, 6.0, (0.44, 0.43), (0.47, 0.43)))
    assert FailureToYieldRule().run(store, make_scene()) == []


def test_failure_to_yield_ignores_pedestrian_waiting_at_the_zebra_end(make_scene):
    """People standing still at the kerb end of the zebra (C3896 @ 174 s) are not crossing yet."""
    store = _store()
    add_track(store, 10, "pedestrian", line(0.0, 6.0, (0.50, 0.41), (0.50, 0.41)))
    add_track(store, 1, "car", line(0.0, 6.0, (0.20, 0.42), (0.80, 0.42)))
    assert FailureToYieldRule().run(store, make_scene()) == []
