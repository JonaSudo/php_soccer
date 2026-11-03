"""CPU tests for the student command observation (noise, commit phase, flicker)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from wbt_training.config_values.kick_command_obs import StudentCommandObsTerm, frame_field


class FakeEnv(SimpleNamespace):
    pass


def make_env(n=4, frames=None):
    """Env whose motion command exposes per-frame kick fields on its loader."""
    frames = frames or {}
    t = 100
    motion = SimpleNamespace(
        _vel_cmd=frames.get("vel_cmd", torch.tensor([[0.5, 0.2]]).repeat(t, 1)),
        _kick_enable=frames.get("kick_enable", torch.ones(t)),
        _kick_phase=frames.get("kick_phase", torch.zeros(t)),
    )
    if "kick_mode" in frames:
        motion._kick_mode = frames["kick_mode"]
    if "kick_dir" in frames:
        motion._kick_dir = frames["kick_dir"]
    cmd = SimpleNamespace(motion=motion, time_steps=torch.zeros(n, dtype=torch.long))
    env = FakeEnv(num_envs=n, device="cpu", command_manager=SimpleNamespace(get_state=lambda name: cmd))
    return env, cmd


def make_term(env, **params):
    params.setdefault("seed", 0)
    return StudentCommandObsTerm(SimpleNamespace(params=params), env)


def test_frame_field_prefers_command_attribute_then_loader_then_zeros():
    env, cmd = make_env(2)
    cmd.time_steps = torch.tensor([0, 3])
    assert frame_field(cmd, "kick_enable", 2, "cpu").tolist() == [1.0, 1.0]
    cmd.kick_enable = torch.tensor([0.0, 1.0])
    assert frame_field(cmd, "kick_enable", 2, "cpu").tolist() == [0.0, 1.0]
    assert frame_field(cmd, "contact_frame_missing", 2, "cpu").tolist() == [0.0, 0.0]
    assert frame_field(cmd, "vel_cmd", 2, "cpu", width=2).shape == (2, 2)


def test_frame_field_tolerates_php_vel_cmd_property_raising():
    class Raising:
        @property
        def vel_cmd(self):
            raise RuntimeError("no field")

    assert frame_field(Raising(), "vel_cmd", 3, "cpu", width=2).shape == (3, 2)


def test_layout_and_noise_free_values():
    env, _ = make_env(3)
    term = make_term(env, v_noise_std=0.0, omega_noise_std=0.0, flicker_prob=0.0)
    out = term(env)
    assert out.shape == (3, 3) == (3, term.dim)
    assert torch.allclose(out, torch.tensor([[0.5, 0.2, 1.0]] * 3))


def test_noise_only_on_velocity_and_has_requested_std():
    env, _ = make_env(20000)
    term = make_term(env, v_noise_std=0.05, omega_noise_std=0.1, flicker_prob=0.0)
    out = term(env)
    assert out[:, 0].std().item() == pytest.approx(0.05, rel=0.05)
    assert out[:, 1].std().item() == pytest.approx(0.1, rel=0.05)
    assert (out[:, 2] == 1.0).all()


def test_modify_history_false_returns_cached_value():
    env, _ = make_env(4)
    term = make_term(env)
    a = term(env)
    b = term(env, modify_history=False)
    assert torch.equal(a, b)


def test_commit_phase_replaces_about_half_of_the_kicks_and_holds_the_values():
    n = 4000
    env, cmd = make_env(n, {"kick_phase": torch.cat([torch.zeros(10), torch.ones(40), torch.zeros(50)])})
    term = make_term(env, v_noise_std=0.0, omega_noise_std=0.0, flicker_prob=0.0, commit_onset_max_steps=5)
    outs = []
    for step in range(60):
        cmd.time_steps = torch.full((n,), step, dtype=torch.long)
        outs.append(term(env))
    outs = torch.stack(outs)  # [T, n, 3]
    ref = torch.tensor([0.5, 0.2, 1.0])
    # Before the commit phase nothing is replaced.
    assert torch.allclose(outs[:10], ref.expand(10, n, 3))
    # Late in the commit phase, about half of the envs observe replaced values.
    changed = (outs[30] - ref).abs().sum(-1) > 1e-6
    assert changed.float().mean().item() == pytest.approx(0.5, abs=0.04)
    # A replaced command is constant over the rest of the commit phase.
    assert torch.allclose(outs[30][changed], outs[45][changed])
    # Replacements respect the configured ranges; kick flag may flip to 0.
    v, w, k = outs[30][changed].unbind(-1)
    assert ((v >= 0.0) & (v <= 1.0)).all() and (w.abs() <= 1.0).all()
    assert set(k.unique().tolist()) <= {0.0, 1.0} and (k == 0).any()
    # After the kick the observation is the reference again.
    assert torch.allclose(outs[55], ref.expand(n, 3))


def test_flicker_only_while_idle_and_short():
    n, steps = 2000, 50
    env, cmd = make_env(n, {"kick_enable": torch.zeros(100)})
    term = make_term(env, v_noise_std=0.0, omega_noise_std=0.0, flicker_prob=0.02, flicker_max_steps=2)
    flags = []
    for step in range(steps):
        cmd.time_steps = torch.full((n,), step, dtype=torch.long)
        flags.append(term(env)[:, 2])
    flags = torch.stack(flags)  # [T, n]
    frac = flags.mean().item()
    assert 0.01 < frac < 0.06
    # No run of ones longer than flicker_max_steps.
    run = torch.zeros(n)
    longest = torch.zeros(n)
    for f in flags:
        run = (run + 1) * f
        longest = torch.maximum(longest, run)
    assert longest.max().item() <= 2


def test_no_flicker_when_kick_enabled():
    env, _ = make_env(500)
    term = make_term(env, v_noise_std=0.0, omega_noise_std=0.0, flicker_prob=1.0)
    assert (term(env)[:, 2] == 1.0).all()


def test_reset_clears_commit_state():
    n = 8
    env, cmd = make_env(n, {"kick_phase": torch.ones(100)})
    term = make_term(env, commit_replace_prob=1.0, commit_onset_max_steps=0)
    term(env)
    assert term.corrupt.all()
    term.reset(torch.tensor([0, 1]))
    assert not term.corrupt[:2].any() and term.corrupt[2:].all()


def test_phase6_layout_kick_dir_and_three_modes():
    t = 100
    env, _ = make_env(
        2,
        {"kick_mode": torch.full((t,), 2.0), "kick_dir": torch.full((t,), 0.3)},
    )
    term = make_term(env, mode="mode3", include_kick_dir=True, v_noise_std=0.0, omega_noise_std=0.0, kick_dir_noise_std=0.0)
    out = term(env)
    assert term.dim == 6 and out.shape == (2, 6)
    assert torch.allclose(out[0], torch.tensor([0.5, 0.2, 0.3, 0.0, 0.0, 1.0]))


def test_invalid_mode_raises():
    env, _ = make_env(1)
    with pytest.raises(ValueError):
        make_term(env, mode="binary")
