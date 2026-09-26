"""All event rules, in one place, so detect_events runs every class uniformly."""
from __future__ import annotations

from src.events.accident import AccidentRule
from src.events.congestion import CongestionRule
from src.events.failure_to_yield import FailureToYieldRule
from src.events.fire_smoke import FireSmokeRule
from src.events.illegal_turn import IllegalTurnRule
from src.events.illegal_u_turn import IllegalUTurnRule
from src.events.jaywalking import JaywalkingRule
from src.events.near_miss import NearMissRule
from src.events.red_light import RedLightRule
from src.events.road_obstacle import RoadObstacleRule
from src.events.solid_line_crossing import SolidLineCrossingRule
from src.events.stop_line import StopLineRule
from src.events.stopped_vehicle import StoppedVehicleRule
from src.events.wrong_way import WrongWayRule

ALL_RULES = (
    AccidentRule,
    CongestionRule,
    FailureToYieldRule,
    FireSmokeRule,
    IllegalTurnRule,
    IllegalUTurnRule,
    JaywalkingRule,
    NearMissRule,
    RedLightRule,
    RoadObstacleRule,
    SolidLineCrossingRule,
    StopLineRule,
    StoppedVehicleRule,
    WrongWayRule,
)
