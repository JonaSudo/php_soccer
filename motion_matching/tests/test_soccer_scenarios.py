"""Soccer scenario recipes and ball placement geometry (no databases needed)."""

from __future__ import annotations

import numpy as np
import pytest

from motion_matching.scenarios import soccer as S
from motion_matching.scenarios.skill_registry import KICK_SKILL_REGISTRY


def _commands(cfg):
    return [step["command"] for step in cfg["steps"]]


@pytest.mark.parametrize("name", sorted({**S.SOCCER_SCENARIOS, **S.PHASE6_SCENARIOS}))
def test_every_step_has_a_continuous_command(name) -> None:
    configs, speed_up = {**S.SOCCER_SCENARIOS, **S.PHASE6_SCENARIOS}[name](num_configs=12)
    assert len(configs) == 12 and speed_up == S.SOCCER_SPEEDUP
    assert len({c["name"] for c in configs}) == 12
    for cfg in configs:
        assert cfg["soccer"]["scenario"] == name
        for step in cfg["steps"]:
            assert "vel_cmd_idx" not in step
            v, omega = step["command"]
            assert 0.0 <= v <= S.V_MAX + 1e-9
            assert abs(omega) <= S.OMEGA_MAX + 1e-9
            if step["type"] == "locomotion":
                # angle passed to motion matching is omega integrated over the step
                assert step["angle"] == pytest.approx(np.rad2deg(omega) * step["duration"])
                assert step["speed"] == v


def test_default_dataset_sizes() -> None:
    sizes = {name: len(fn()[0]) for name, fn in S.SOCCER_SCENARIOS.items()}
    assert sizes == {"walk_kick": 300, "stand_kick": 100, "stand_ball_outside_zone": 100,
                     "walk_ball_offpath": 100, "walk_no_ball": 100}


def test_walk_kick_holds_command_during_kick() -> None:
    configs, _ = S.walk_kick(num_configs=20)
    feet = set()
    for cfg in configs:
        steps = cfg["steps"]
        kicks = [i for i, s in enumerate(steps) if s["type"] == "skill"]
        assert len(kicks) == 1
        k = kicks[0]
        kick = steps[k]
        assert kick["skill_name"] in KICK_SKILL_REGISTRY
        feet.add(kick["skill_name"])
        run_up = steps[k - 1]
        assert run_up["command"][1] == 0.0
        assert kick["command"] == (run_up["command"][0], 0.0) and kick["speed"] == run_up["command"][0]
        assert run_up["command"][0] >= S.V_KICK_MIN
        assert all(s["kick_enable"] == 1.0 for s in steps)
        body_time = sum(s["duration"] for s in steps[1 : k - 1])
        assert 0.1 - 1e-9 <= body_time <= 3.0 + 1e-9
        assert steps[k + 1]["duration"] == 2.0
        assert steps[-1]["command"] == (0.0, 0.0)
    assert feet == {"kick_right", "kick_left"}


def test_stand_kick_rising_edge() -> None:
    configs, _ = S.stand_kick(num_configs=10)
    for cfg in configs:
        first, kick = cfg["steps"][0], cfg["steps"][1]
        assert first["kick_enable"] == 0.0 and first["command"] == (0.0, 0.0)
        assert S.STAND_DURATION + 0.5 <= first["duration"] <= S.STAND_DURATION + 3.0
        assert kick["skill_name"].startswith("kick_stand_") and kick["kick_enable"] == 1.0
        assert kick["command"] == (0.0, 0.0)
        assert all(s["kick_enable"] == 1.0 for s in cfg["steps"][1:])


def test_negatives_have_no_kick_and_both_enable_values() -> None:
    for fn in (S.stand_ball_outside_zone, S.walk_ball_offpath, S.walk_no_ball):
        configs, _ = fn(num_configs=6)
        enables = set()
        for cfg in configs:
            assert not any(s["type"] == "skill" for s in cfg["steps"])
            enables |= {s["kick_enable"] for s in cfg["steps"]}
        assert enables == {0.0, 1.0}


def test_recipes_are_deterministic() -> None:
    assert S.walk_kick(num_configs=5) == S.walk_kick(num_configs=5)


def test_corridor_has_a_dead_zone() -> None:
    assert S.KICK_CORRIDOR < S.IGNORE_LATERAL_MIN < S.IGNORE_LATERAL_MAX


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


@pytest.mark.parametrize("yaw", [0.0, 0.7, -2.0])
def test_outside_zone_is_outside_but_close(yaw) -> None:
    rng = np.random.default_rng(0)
    foot = np.array([0.3, -0.2])
    q = _yaw_quat(yaw)
    for _ in range(200):
        ball = S.sample_ball_outside_zone(rng, foot, q)
        assert not S.ball_in_kick_zone(ball, foot, q)
        assert np.linalg.norm(ball - foot) < S.KICK_ZONE_FORWARD[1] + 0.25
    fwd = np.array([np.cos(yaw), np.sin(yaw)])
    assert S.ball_in_kick_zone(foot + 0.15 * fwd, foot, q)


def _curved_path(n=300):
    t = np.linspace(0, 3, n)
    yaw = 0.6 * t
    xy = np.stack([np.sin(yaw) / 0.6, (1 - np.cos(yaw)) / 0.6], axis=1)
    quats = np.stack([_yaw_quat(y) for y in yaw])
    return xy, quats


def test_offpath_ball_respects_corridor() -> None:
    rng = np.random.default_rng(1)
    xy, quats = _curved_path()
    for _ in range(50):
        ball = S.sample_ball_offpath(rng, xy, quats)
        d = S.path_distance(ball, xy)[0]
        assert S.IGNORE_LATERAL_MIN <= d <= S.IGNORE_LATERAL_MAX


def test_far_ball_is_far() -> None:
    rng = np.random.default_rng(2)
    xy, _ = _curved_path()
    for _ in range(20):
        assert S.path_distance(S.sample_ball_far(rng, xy), xy)[0] >= S.FAR_BALL_MIN_DISTANCE


# ------------------------------------------------------------------ Phase 6
def test_phase2_recipes_unchanged_without_kick_dir() -> None:
    for fn in (S.walk_kick, S.stand_kick):
        configs, _ = fn(num_configs=4)
        assert all("kick_dir" not in cfg["soccer"] for cfg in configs)
    assert [c["steps"] for c in S.walk_kick(num_configs=4, kick_dir_max=0.3)[0]] == [
        c["steps"] for c in S.walk_kick(num_configs=4)[0]
    ]  # the kick direction has its own RNG stream


@pytest.mark.parametrize("fn", [S.walk_kick_dir, S.stand_kick_dir])
def test_kick_dir_scenarios_sample_within_range(fn) -> None:
    configs, _ = fn(num_configs=40)
    dirs = np.array([cfg["soccer"]["kick_dir"] for cfg in configs])
    assert np.all(np.abs(dirs) <= S.KICK_DIR_MAX) and dirs.min() < -0.1 and dirs.max() > 0.1
    scenario = fn.__name__
    assert all(cfg["name"].startswith(scenario) and cfg["soccer"]["scenario"] == scenario for cfg in configs)
    assert all(cfg["soccer"]["ball"]["mode"] == "kick" for cfg in configs)


def test_walk_dribble_recipe() -> None:
    configs, _ = S.walk_dribble(num_configs=10)
    for cfg in configs:
        assert cfg["soccer"]["ball"]["mode"] == "dribble"
        assert not any(s["type"] == "skill" for s in cfg["steps"])
        assert {s["kick_enable"] for s in cfg["steps"]} == {0.0}
        for v, omega in _commands(cfg):
            assert v <= S.DRIBBLE_V_MAX and abs(omega) <= S.DRIBBLE_OMEGA_MAX


def test_dribble_ball_lies_in_front_where_walking_starts() -> None:
    rng = np.random.default_rng(0)
    yaw = 0.7
    quat = np.array([[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]] * 6)
    root = np.array([[1.0, 1.0]] * 3 + [[1.0 + 0.1 * k * np.cos(yaw), 1.0 + 0.1 * k * np.sin(yaw)] for k in (1, 2, 3)])
    for _ in range(20):
        ball = S.sample_ball_dribble(rng, root, quat)
        rel = ball - root[2]
        ahead = rel @ np.array([np.cos(yaw), np.sin(yaw)])
        side = rel @ np.array([-np.sin(yaw), np.cos(yaw)])
        assert S.DRIBBLE_AHEAD[0] - 1e-9 <= ahead <= S.DRIBBLE_AHEAD[1] + 1e-9 and abs(side) <= S.DRIBBLE_LATERAL
