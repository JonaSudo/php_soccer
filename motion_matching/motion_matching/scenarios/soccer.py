"""Soccer scenarios for LUPUS (Phase 2): walk -> kick -> walk, free kicks and negatives.

Every scenario here drives motion matching with a continuous command
``(v, omega)`` (forward speed in m/s, turn rate in rad/s, no side steps)
instead of the parkour one-hot. The command lands per frame in ``vel_cmd``
with shape ``(T, 2)``. During a kick it is held at ``(v_before, 0)``, as PHP
holds "straight ahead" during a skill.

A config is a normal scenario dict (``name``, ``steps``) plus a ``soccer``
block that tells :func:`build_soccer_fields` where the ball goes::

    "soccer": {"scenario": "walk_kick", "ball": {"mode": "kick"}, "seed": 7}

Ball modes: ``kick`` (annotated ball of the matched kick clip), ``outside_zone``
(just outside the stand kick zone), ``offpath`` (beside the walking corridor)
and ``far`` (parked far from the path; the teacher needs a ball in every motion).

All ranges below are placeholders tuned for a robot of Booster K1 size; they
are module constants and keyword arguments so they are easy to change once
LUPUS' real limits are known.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from motion_matching.io.kick_io import KICK_SIDE, build_kick_fields, load_ball_radius, map_frame
from motion_matching.utils.data_utils import adjust_vel_cmd_speed, resample_vel_cmd

# --- Command ranges (open question in the plan: what LUPUS has to manage) ---
V_MIN = 0.2  # m/s, slowest walking command
V_MAX = 0.8  # m/s
OMEGA_MAX = 0.8  # rad/s
V_KICK_MIN = 0.3  # m/s, slowest run-up that still is a "kick out of the run"

# --- Corridor and kick zone ---
# Kick when the ball is less than KICK_CORRIDOR beside the path, ignore it from
# IGNORE_LATERAL_MIN on. Nothing is generated in between (dead zone), so the
# decision stays unambiguous.
KICK_CORRIDOR = 0.10
IGNORE_LATERAL_MIN = 0.25
IGNORE_LATERAL_MAX = 1.0
# Stand kick zone, relative to the kick foot (ankle bone) in the heading frame.
KICK_ZONE_FORWARD = (0.10, 0.25)
KICK_ZONE_LATERAL = 0.05
# "Just outside": these margins beyond the zone are where negatives are sampled.
OUTSIDE_ZONE_MARGIN = (0.05, 0.15)
# walk_no_ball parks the ball at least this far from every path point.
FAR_BALL_MIN_DISTANCE = 5.0

STAND_DURATION = 1.0
END_STAND_DURATION = 2.5
SOCCER_SPEEDUP = 1.0

FEET = ("right", "left")


# --------------------------------------------------------------------------
# Step builders
# --------------------------------------------------------------------------


def sample_command(
    rng: np.random.Generator, v_min: float = V_MIN, v_max: float = V_MAX, omega_max: float = OMEGA_MAX
) -> tuple[float, float]:
    """Uniform ``v`` in ``[v_min, v_max]``, ``omega`` in ``[-omega_max, omega_max]``."""
    return float(rng.uniform(v_min, v_max)), float(rng.uniform(-omega_max, omega_max))


def stand_step(duration: float, kick_enable: float = 0.0) -> dict[str, Any]:
    return {
        "type": "segment",
        "speed": 0.0,
        "angle": 0.0,
        "duration": float(duration),
        "command": (0.0, 0.0),
        "kick_enable": float(kick_enable),
    }


def walk_step(v: float, omega: float, duration: float, kick_enable: float = 0.0) -> dict[str, Any]:
    """Locomotion at forward speed ``v`` turning at ``omega`` for ``duration`` s."""
    return {
        "type": "locomotion",
        "speed": float(v),
        "angle": float(np.rad2deg(omega) * duration),
        "duration": float(duration),
        "command": (float(v), float(omega)),
        "kick_enable": float(kick_enable),
    }


def kick_step(skill_name: str, v_before: float, kick_enable: float = 1.0) -> dict[str, Any]:
    """Trigger a kick skill; the command is held at ``(v_before, 0)`` throughout."""
    return {
        "type": "skill",
        "skill_name": skill_name,
        "speed": float(v_before),
        "command": (float(v_before), 0.0),
        "kick_enable": float(kick_enable),
    }


def _random_walk_body(
    rng: np.random.Generator,
    *,
    total_duration: tuple[float, float],
    kick_enable: float,
    v_min: float,
    v_max: float,
    omega_max: float,
    max_segments: int = 3,
) -> tuple[list[dict[str, Any]], float]:
    """1..max_segments random (v, omega) segments summing to a random duration."""
    total = float(rng.uniform(*total_duration))
    n = int(rng.integers(1, max_segments + 1))
    cuts = np.sort(rng.uniform(0.0, total, size=n - 1)) if n > 1 else np.array([])
    durations = np.diff(np.concatenate([[0.0], cuts, [total]]))
    steps = []
    v = 0.0
    for d in durations:
        if d < 0.05:
            continue
        v, omega = sample_command(rng, v_min, v_max, omega_max)
        steps.append(walk_step(v, omega, d, kick_enable))
    if not steps:  # every cut was tiny
        v, omega = sample_command(rng, v_min, v_max, omega_max)
        steps.append(walk_step(v, omega, total, kick_enable))
    return steps, v


def _cfg(name: str, steps: list, scenario: str, ball: dict, seed: int, kick_side: Optional[str] = None) -> dict:
    return {
        "name": name,
        "steps": steps,
        "soccer": {"scenario": scenario, "ball": ball, "seed": int(seed), "kick_side": kick_side},
    }


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------


def walk_kick(
    num_configs: int = 300,
    v_min: float = V_KICK_MIN,
    v_max: float = V_MAX,
    omega_max: float = OMEGA_MAX,
    seed: int = 2026,
) -> tuple[list[dict[str, Any]], float]:
    """Stand -> walk (random v, omega, 0.1-3 s) -> straight run-up -> kick -> 2 s walk -> stand.

    Alternates the kick foot so both standing legs are covered; the random
    walking duration spreads the gait phase at which the kick is entered.
    ``kick_enable`` is 1 for the whole motion.
    """
    rng = np.random.default_rng(seed)
    configs = []
    for i in range(num_configs):
        foot = FEET[i % 2]
        body, _ = _random_walk_body(
            rng, total_duration=(0.1, 3.0), kick_enable=1.0, v_min=v_min, v_max=v_max, omega_max=omega_max
        )
        # Straighten out before the kick: the kick holds omega = 0, and a
        # straight run-up lets motion matching find the entry window.
        v_before = float(rng.uniform(v_min, v_max))
        run_up = walk_step(v_before, 0.0, float(rng.uniform(0.6, 1.2)), 1.0)
        v_after, omega_after = sample_command(rng, v_min, v_max, omega_max)
        steps = [
            stand_step(STAND_DURATION, 1.0),
            *body,
            run_up,
            kick_step(f"kick_{foot}", v_before),
            walk_step(v_after, omega_after, 2.0, 1.0),
            stand_step(END_STAND_DURATION, 1.0),
        ]
        configs.append(_cfg(f"walk_kick_{i:04d}_{foot}", steps, "walk_kick", {"mode": "kick"}, seed + i, foot))
    return configs, SOCCER_SPEEDUP


def stand_kick(
    num_configs: int = 100,
    v_min: float = V_MIN,
    v_max: float = V_MAX,
    omega_max: float = OMEGA_MAX,
    seed: int = 3026,
) -> tuple[list[dict[str, Any]], float]:
    """Free kick: stand at the ball, wait 0.5-3 s with kick_enable 0, rising edge -> kick.

    Afterwards either keep standing or walk off. ``kick_enable`` stays 1 after
    the edge; the rising edge is the trigger.
    """
    rng = np.random.default_rng(seed)
    configs = []
    for i in range(num_configs):
        foot = FEET[i % 2]
        wait = float(rng.uniform(0.5, 3.0))
        steps = [stand_step(STAND_DURATION + wait, 0.0), kick_step(f"kick_stand_{foot}", 0.0)]
        if rng.random() < 0.5:
            steps.append(stand_step(END_STAND_DURATION, 1.0))
            tail = "stand"
        else:
            v, omega = sample_command(rng, v_min, v_max, omega_max)
            steps += [walk_step(v, omega, float(rng.uniform(1.0, 2.5)), 1.0), stand_step(END_STAND_DURATION, 1.0)]
            tail = "walk"
        configs.append(
            _cfg(f"stand_kick_{i:04d}_{foot}_{tail}", steps, "stand_kick", {"mode": "kick"}, seed + i, foot)
        )
    return configs, SOCCER_SPEEDUP


def stand_ball_outside_zone(num_configs: int = 100, seed: int = 4026) -> tuple[list[dict[str, Any]], float]:
    """Stand with the ball just outside the kick zone; never kick, kick_enable 0, 1 or 0 -> 1."""
    rng = np.random.default_rng(seed)
    configs = []
    for i in range(num_configs):
        foot = FEET[i % 2]
        duration = float(rng.uniform(2.0, 4.0))
        pattern = ("off", "on", "edge")[i % 3]
        if pattern == "edge":
            first = float(rng.uniform(0.5, duration - 0.5))
            steps = [stand_step(first, 0.0), stand_step(duration - first, 1.0)]
        else:
            steps = [stand_step(duration, 1.0 if pattern == "on" else 0.0)]
        ball = {"mode": "outside_zone", "foot": foot}
        configs.append(_cfg(f"stand_ball_outside_zone_{i:04d}_{foot}_{pattern}", steps, "stand_ball_outside_zone", ball, seed + i))
    return configs, SOCCER_SPEEDUP


def _walk_negative(
    scenario: str,
    ball_mode: str,
    num_configs: int,
    seed: int,
    v_min: float,
    v_max: float,
    omega_max: float,
) -> tuple[list[dict[str, Any]], float]:
    rng = np.random.default_rng(seed)
    configs = []
    for i in range(num_configs):
        kick_enable = float(i % 2)
        body, _ = _random_walk_body(
            rng, total_duration=(1.5, 6.0), kick_enable=kick_enable, v_min=v_min, v_max=v_max, omega_max=omega_max,
            max_segments=5,
        )
        steps = [stand_step(STAND_DURATION, kick_enable), *body, stand_step(END_STAND_DURATION, kick_enable)]
        configs.append(
            _cfg(f"{scenario}_{i:04d}_ke{int(kick_enable)}", steps, scenario, {"mode": ball_mode}, seed + i)
        )
    return configs, SOCCER_SPEEDUP


def walk_ball_offpath(
    num_configs: int = 100, v_min: float = V_MIN, v_max: float = V_MAX, omega_max: float = OMEGA_MAX, seed: int = 5026
) -> tuple[list[dict[str, Any]], float]:
    """Walk with the ball 0.25-1.0 m beside the path; it is never touched. kick_enable 0/1."""
    return _walk_negative("walk_ball_offpath", "offpath", num_configs, seed, v_min, v_max, omega_max)


def walk_no_ball(
    num_configs: int = 100, v_min: float = V_MIN, v_max: float = V_MAX, omega_max: float = OMEGA_MAX, seed: int = 6026
) -> tuple[list[dict[str, Any]], float]:
    """Random locomotion without a ball in reach (parked >= 5 m away). kick_enable 0/1."""
    return _walk_negative("walk_no_ball", "far", num_configs, seed, v_min, v_max, omega_max)


SOCCER_SCENARIOS = {
    "walk_kick": walk_kick,
    "stand_kick": stand_kick,
    "stand_ball_outside_zone": stand_ball_outside_zone,
    "walk_ball_offpath": walk_ball_offpath,
    "walk_no_ball": walk_no_ball,
}


# --------------------------------------------------------------------------
# Ball placement geometry
# --------------------------------------------------------------------------


def _yaw(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _heading_frame(quat_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yaw = float(_yaw(quat_wxyz))
    fwd = np.array([np.cos(yaw), np.sin(yaw)])
    left = np.array([-np.sin(yaw), np.cos(yaw)])
    return fwd, left


def path_distance(points_xy: np.ndarray, path_xy: np.ndarray) -> np.ndarray:
    """Distance of each point to the nearest segment of the polyline ``path_xy``."""
    points_xy = np.atleast_2d(points_xy).astype(np.float64)
    path_xy = np.asarray(path_xy, dtype=np.float64)
    if len(path_xy) == 1:
        return np.linalg.norm(points_xy - path_xy[0], axis=1)
    a, b = path_xy[:-1], path_xy[1:]
    ab = b - a
    denom = np.maximum(np.sum(ab * ab, axis=1), 1e-12)
    out = np.empty(len(points_xy))
    for i, p in enumerate(points_xy):
        t = np.clip(np.sum((p - a) * ab, axis=1) / denom, 0.0, 1.0)
        out[i] = np.min(np.linalg.norm(a + t[:, None] * ab - p, axis=1))
    return out


def ball_in_kick_zone(ball_xy: np.ndarray, foot_xy: np.ndarray, root_quat_wxyz: np.ndarray) -> bool:
    """Is the ball inside the stand kick zone in front of ``foot_xy``?"""
    fwd, left = _heading_frame(root_quat_wxyz)
    d = np.asarray(ball_xy, dtype=np.float64) - np.asarray(foot_xy, dtype=np.float64)
    ahead, side = float(d @ fwd), float(d @ left)
    return KICK_ZONE_FORWARD[0] <= ahead <= KICK_ZONE_FORWARD[1] and abs(side) <= KICK_ZONE_LATERAL


def sample_ball_outside_zone(
    rng: np.random.Generator, foot_xy: np.ndarray, root_quat_wxyz: np.ndarray
) -> np.ndarray:
    """Ball just outside the kick zone: a bit too far ahead, or a bit beside it."""
    fwd, left = _heading_frame(root_quat_wxyz)
    margin = float(rng.uniform(*OUTSIDE_ZONE_MARGIN))
    if rng.random() < 0.5:  # too far ahead
        ahead = KICK_ZONE_FORWARD[1] + margin
        side = float(rng.uniform(-KICK_ZONE_LATERAL, KICK_ZONE_LATERAL))
    else:  # beside the zone
        ahead = float(rng.uniform(*KICK_ZONE_FORWARD))
        side = float(rng.choice([-1.0, 1.0])) * (KICK_ZONE_LATERAL + margin)
    return np.asarray(foot_xy, dtype=np.float64) + ahead * fwd + side * left


def sample_ball_offpath(
    rng: np.random.Generator,
    root_xy: np.ndarray,
    root_quat_wxyz: np.ndarray,
    lateral: tuple[float, float] = (IGNORE_LATERAL_MIN, IGNORE_LATERAL_MAX),
    max_tries: int = 200,
) -> np.ndarray:
    """Ball beside the walked path: ``lateral`` m from the whole root path."""
    moving = np.flatnonzero(np.linalg.norm(np.diff(root_xy, axis=0), axis=1) > 1e-4)
    candidates = moving if len(moving) else np.arange(len(root_xy))
    for _ in range(max_tries):
        k = int(rng.choice(candidates))
        _, left = _heading_frame(root_quat_wxyz[k])
        offset = float(rng.uniform(*lateral)) * float(rng.choice([-1.0, 1.0]))
        ball = root_xy[k] + offset * left
        d = float(path_distance(ball, root_xy)[0])
        if lateral[0] <= d <= lateral[1]:
            return ball
    raise RuntimeError("could not place an off-path ball; path curls too tightly")


def sample_ball_far(
    rng: np.random.Generator, root_xy: np.ndarray, min_distance: float = FAR_BALL_MIN_DISTANCE
) -> np.ndarray:
    """Ball parked at least ``min_distance`` from every path point."""
    centre = root_xy.mean(axis=0)
    reach = float(np.max(np.linalg.norm(root_xy - centre, axis=1)))
    angle = float(rng.uniform(-np.pi, np.pi))
    ball = centre + (reach + min_distance + float(rng.uniform(0.0, 2.0))) * np.array([np.cos(angle), np.sin(angle)])
    assert path_distance(ball, root_xy)[0] >= min_distance
    return ball


# --------------------------------------------------------------------------
# Per-frame fields for process_scenario
# --------------------------------------------------------------------------


def _hold(values: np.ndarray, speed_up_ratio: float, required_fps: int) -> np.ndarray:
    """Exactly the vel_cmd resampling path, for any per-frame array."""
    v = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    v = adjust_vel_cmd_speed(v, speed_up_ratio)
    return resample_vel_cmd(v, required_fps, original_fps=60)


def build_soccer_fields(
    soccer_cfg: dict,
    *,
    frame_log: Sequence[tuple],
    kick_events: Sequence[dict],
    kick_enable: np.ndarray,
    qpos_out: np.ndarray,
    num_captured: int,
    num_out: int,
    speed_up_ratio: float,
    required_fps: int,
    ball_radius: Optional[float] = None,
) -> dict[str, np.ndarray]:
    """Ball, kick phase and metadata arrays for one generated soccer motion.

    ``frame_log`` / ``kick_enable`` are per captured (60 fps) frame,
    ``qpos_out`` is the speed-adjusted qpos still on the 60 fps grid, and
    ``num_out`` the frame count of the saved motion.
    """
    if len(frame_log) != num_captured or len(kick_enable) != num_captured:
        raise ValueError("frame log / kick_enable do not match the captured frames")
    rng = np.random.default_rng(soccer_cfg.get("seed"))
    radius = load_ball_radius() if ball_radius is None else float(ball_radius)
    mode = soccer_cfg["ball"]["mode"]
    n_adj = len(qpos_out)

    def to_out(i: int) -> int:
        return map_frame(map_frame(i, num_captured, n_adj), n_adj, num_out)

    commit_start = contact = None
    kick_side = None
    if mode == "kick":
        if len(kick_events) != 1:
            raise RuntimeError(f"kick scenario needs exactly one kick, got {len(kick_events)}")
        event = kick_events[0]
        ball_xy = np.asarray(event["ball_pos_w"][:2], dtype=np.float64)
        ball_z = float(event["ball_pos_w"][2])
        kick_side = event["kick_foot"]
        in_skill = [i for i, (state, *_rest) in enumerate(frame_log) if state == event["skill"]]
        if not in_skill:
            raise RuntimeError("kick skill produced no frames")
        commit_src = in_skill[0]
        # First frame at (or past) the annotated contact; the trigger frame is
        # captured before the clip frame advances, hence the >= test.
        contact_src = next((i for i in in_skill if frame_log[i][1] >= event["contact_db_frame"]), None)
        if contact_src is None:
            raise RuntimeError("kick clip ended before its contact frame")
        commit_start, contact = to_out(commit_src), to_out(contact_src)
        if contact <= commit_start:
            contact = min(commit_start + 1, num_out - 1)
    else:
        root_xy = qpos_out[:, 4:6].astype(np.float64)
        root_quat = qpos_out[:, 0:4].astype(np.float64)
        ball_z = radius
        if mode == "outside_zone":
            foot = soccer_cfg["ball"].get("foot", "right")
            feet = frame_log[-1][2]
            foot_xy = feet[1 if foot == "right" else 0][:2]
            ball_xy = sample_ball_outside_zone(rng, foot_xy, root_quat[-1])
        elif mode == "offpath":
            ball_xy = sample_ball_offpath(rng, root_xy, root_quat)
        elif mode == "far":
            ball_xy = sample_ball_far(rng, root_xy)
        else:
            raise ValueError(f"unknown soccer ball mode {mode!r}")

    enable_out = _hold(kick_enable, speed_up_ratio, required_fps)[:, 0]
    if len(enable_out) != num_out:
        raise RuntimeError(f"kick_enable resampled to {len(enable_out)} frames, motion has {num_out}")
    fields = build_kick_fields(
        num_frames=num_out,
        kick_enable=enable_out,
        commit_start=commit_start,
        contact=contact,
        ball_pos=np.array([ball_xy[0], ball_xy[1], ball_z], dtype=np.float32),
    )
    fields["kick_side"] = np.array(KICK_SIDE[kick_side], dtype=np.int8)
    fields["scenario"] = np.array(soccer_cfg.get("scenario", ""))
    return fields
