"""Scenario definitions for motion matching trajectory generation.

Each public function in this package defines a scenario: a sequence of locomotion
segments and skill triggers that produce one or more motion clips. Scenarios are
selected via ``--scenario`` on the CLI, dispatched at runtime by name.

To add a new scenario: define a function in ``builtin.py`` (or another module
imported below) that returns ``(configs, speed_up_ratio)``. Parkour scenarios
live in ``builtin.py`` (one-hot command), soccer ones in ``soccer.py``
(continuous ``(v, omega)`` command plus ball and kick fields).
"""

from motion_matching.scenarios import builtin as builtin  # re-export for getattr-style dispatch
from motion_matching.scenarios.builtin import (
    high_speed_cat_vault,
    high_speed_climb_58,
    high_speed_climb_76,
    high_speed_climb_94,
    high_speed_dash_vault,
    high_speed_hurdle,
    high_speed_jump_35,
    high_speed_roll_134,
    high_speed_speed_vault,
    high_speed_step,
    low_speed_climb_58,
    low_speed_climb_76,
    low_speed_climb_94,
    low_speed_roll_78,
    low_speed_roll_90,
    low_speed_step,
    random_locomotion,
)
from motion_matching.scenarios import soccer as soccer  # re-export for getattr-style dispatch
from motion_matching.scenarios.soccer import (
    stand_ball_outside_zone,
    stand_kick,
    walk_ball_offpath,
    walk_kick,
    walk_no_ball,
)

# These names are dispatched by ``getattr(motion_matching.scenarios, name)`` in
# run_generation, so they must stay bound on the package. Keep in sync with the
# scenario functions in builtin.py.
__all__ = [
    "builtin",
    "high_speed_cat_vault",
    "high_speed_climb_58",
    "high_speed_climb_76",
    "high_speed_climb_94",
    "high_speed_dash_vault",
    "high_speed_hurdle",
    "high_speed_jump_35",
    "high_speed_roll_134",
    "high_speed_speed_vault",
    "high_speed_step",
    "low_speed_climb_58",
    "low_speed_climb_76",
    "low_speed_climb_94",
    "low_speed_roll_78",
    "low_speed_roll_90",
    "low_speed_step",
    "random_locomotion",
    # Soccer (continuous (v, omega) command, kicks), see scenarios/soccer.py.
    "soccer",
    "stand_ball_outside_zone",
    "stand_kick",
    "walk_ball_offpath",
    "walk_kick",
    "walk_no_ball",
]
