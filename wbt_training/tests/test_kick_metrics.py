"""CPU tests for kick metrics, the eval grid, command jitter and the eval callback."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wbt_training.utils.command_jitter import CommandJitter, CommandJitterCfg, load_residuals
from wbt_training.utils.kick_metrics import (
    KickEpisodeTracker,
    KickEvalGrid,
    KickMetricsCfg,
    summarize,
    table_to_markdown,
    write_report,
)


def step(n, **over):
    s = {
        "ball_pos_w": torch.tensor([[1.0, 0.05, 0.07]]).repeat(n, 1),
        "ball_vel_w": torch.zeros(n, 3),
        "base_pos_w": torch.tensor([[0.0, 0.0, 0.57]]).repeat(n, 1),
        "base_yaw": torch.zeros(n),
        "base_up_z": torch.ones(n),
        "base_fwd_vel": torch.full((n,), 0.5),
        "base_yaw_rate": torch.zeros(n),
        "cmd_v": torch.full((n,), 0.5),
        "cmd_w": torch.zeros(n),
        "kick_phase": torch.zeros(n),
        "contact": torch.zeros(n, dtype=torch.bool),
        "target_heading": torch.full((n,), float("nan")),
    }
    s.update({k: torch.as_tensor(v) for k, v in over.items()})
    return s


def run_kick(tracker, ball_dir_deg, speed=3.0, n=1):
    tracker.step(step(n))
    tracker.step(step(n, kick_phase=torch.ones(n)))
    tracker.step(step(n, kick_phase=torch.full((n,), 2.0), contact=torch.ones(n, dtype=torch.bool)))
    a = math.radians(ball_dir_deg)
    for _ in range(5):
        tracker.step(step(n, kick_phase=torch.full((n,), 2.0), ball_vel_w=torch.tensor([[speed * math.cos(a), speed * math.sin(a), 0.0]]).repeat(n, 1)))
    tracker.finish(torch.arange(n))


def test_successful_kick_record():
    t = KickEpisodeTracker(1)
    run_kick(t, 5.0)
    (r,) = t.records
    assert r["intended"] and r["kicked"] and r["success"] and not r["miskick"]
    assert r["dir_err_deg"] == pytest.approx(5.0, abs=1e-4)
    assert r["ball_speed"] == pytest.approx(3.0)
    assert r["approach"] == pytest.approx(1.0) and r["lateral"] == pytest.approx(0.05)
    assert r["speed"] == pytest.approx(0.5) and r["v_err"] == pytest.approx(0.0)


def test_wrong_direction_or_weak_kick_is_no_success():
    t = KickEpisodeTracker(1)
    run_kick(t, 20.0)
    run_kick(t, 0.0, speed=0.2)
    assert [r["success"] for r in t.records] == [False, False]
    assert t.records[0]["kicked"]


def test_target_heading_shifts_the_error():
    t = KickEpisodeTracker(1)
    t.step(step(1, kick_phase=torch.ones(1)))
    t.step(step(1, kick_phase=torch.ones(1), contact=torch.ones(1, dtype=torch.bool), target_heading=torch.tensor([math.radians(30)])))
    t.step(step(1, kick_phase=torch.ones(1), ball_vel_w=torch.tensor([[math.cos(0.5236), math.sin(0.5236), 0.0]])))
    t.finish([0])
    assert t.records[0]["dir_err_deg"] == pytest.approx(0.0, abs=1e-3)


def test_contact_without_kick_is_a_miskick_and_falls_are_seen():
    t = KickEpisodeTracker(2)
    t.step(step(2, contact=torch.tensor([True, False])))
    t.step(step(2, base_up_z=torch.tensor([1.0, 0.2])))
    t.finish([0, 1], terminated=[False, True])
    a, b = t.records
    assert a["miskick"] and not a["intended"] and not a["fell"]
    assert b["fell"] and b["terminated"] and not b["miskick"]
    s = summarize(t.records)
    assert s["miskick_rate"] == 0.5 and s["fall_rate"] == 0.5 and math.isnan(s["kick_success_rate"])


def test_contacts_after_the_kick_are_not_miskicks():
    t = KickEpisodeTracker(1)
    run_kick(t, 0.0)
    t.step(step(1, kick_phase=torch.ones(1)))
    t.step(step(1, kick_phase=torch.ones(1), contact=torch.ones(1, dtype=torch.bool)))
    t.step(step(1, kick_phase=torch.zeros(1), contact=torch.ones(1, dtype=torch.bool)))  # walks on, touches again
    t.finish([0])
    assert not t.records[1]["miskick"]


def test_finish_resets_only_the_given_envs():
    t = KickEpisodeTracker(2)
    t.step(step(2, kick_phase=torch.ones(2)))
    t.finish([0])
    assert not t.intended[0] and t.intended[1]
    t.finish([0])  # not started again -> no record
    assert len(t.records) == 1


def test_grid_bins_and_table():
    g = KickEvalGrid(approach=(0.5, 2.0), lateral=(0.0, 0.25), speed=(0.3, 0.9), episodes_per_cell=2)
    assert len(g.cells()) == 8 and len(g.scenario_specs()) == 8
    recs = []
    t = KickEpisodeTracker(1)
    for _ in range(2):
        run_kick(t, 0.0)
    recs = t.records
    assert g.cell_of(recs[0]) == (0.5, 0.0, 0.3)
    rows = g.table(recs)
    row = next(r for r in rows if r["n"])
    assert row["n"] == 2 and row["complete"] and row["kick_success_rate"] == 1.0
    md = table_to_markdown(rows)
    assert md.count("\n") == 2  # header, rule, one non-empty row


def test_write_report(tmp_path):
    t = KickEpisodeTracker(1)
    run_kick(t, 0.0)
    path = tmp_path / "r.json"
    write_report(str(path), t.records)
    data = json.loads(path.read_text())
    assert data["summary"]["kick_success_rate"] == 1.0 and len(data["records"]) == 1


# --- jitter -----------------------------------------------------------------


def test_jitter_ou_std_and_hold_rate():
    g = torch.Generator().manual_seed(0)
    cfg = CommandJitterCfg(sigma_v=0.1, sigma_omega=0.2, tau_s=0.1, update_hz=10.0, stale_prob=0.0)
    j = CommandJitter(4000, cfg, generator=g)
    vel = torch.tensor([[0.5, 0.0]]).repeat(4000, 1)
    outs = [j.apply(vel, torch.ones(4000))[0] for _ in range(100)]
    o = torch.stack(outs)
    assert (o[-1, :, 0] - 0.5).std().item() == pytest.approx(0.1, rel=0.1)
    assert (o[-1, :, 1]).std().item() == pytest.approx(0.2, rel=0.1)
    changes = (o[1:] != o[:-1]).any(-1).float().mean().item()
    assert changes == pytest.approx(10.0 / 50.0, abs=0.02)


def test_jitter_replays_log_residuals(tmp_path):
    t = np.arange(500)
    v = 0.5 + 0.1 * np.sin(t * 0.9)  # fast wiggle around 0.5
    path = tmp_path / "log.npz"
    np.savez(path, v=v, omega=np.zeros_like(v))
    res = load_residuals(str(path), 25)
    assert res.shape == (500, 2) and abs(float(res[50:-50, 0].mean())) < 0.01
    cfg = CommandJitterCfg(update_hz=50.0, stale_prob=0.0, log_file=str(path))
    j = CommandJitter(3, cfg, generator=torch.Generator().manual_seed(1))
    out, _ = j.apply(torch.zeros(3, 2), torch.zeros(3))
    assert torch.allclose(out, res[j.offset - 1])


def test_jitter_kick_glitch():
    cfg = CommandJitterCfg(update_hz=50.0, stale_prob=0.0, kick_glitch_prob=0.1)
    j = CommandJitter(5000, cfg, generator=torch.Generator().manual_seed(2))
    _, k = j.apply(torch.zeros(5000, 2), torch.zeros(5000))
    assert k.mean().item() == pytest.approx(0.1, abs=0.02)


def test_command_obs_uses_jitter():
    from wbt_training.config_values.kick_command_obs import StudentCommandObsTerm

    n = 2000
    motion = SimpleNamespace(_vel_cmd=torch.tensor([[0.5, 0.0]]).repeat(10, 1), _kick_enable=torch.ones(10), _kick_phase=torch.zeros(10))
    cmd = SimpleNamespace(motion=motion, time_steps=torch.zeros(n, dtype=torch.long))
    env = SimpleNamespace(num_envs=n, device="cpu", dt=0.02, command_manager=SimpleNamespace(get_state=lambda _n: cmd))
    params = {"seed": 0, "v_noise_std": 0.0, "omega_noise_std": 0.0, "flicker_prob": 0.0, "commit_replace_prob": 0.0, "jitter": {"sigma_v": 0.2, "tau_s": 0.02, "update_hz": 50.0}}
    term = StudentCommandObsTerm(SimpleNamespace(params=params), env)
    out = term(env)
    assert (out[:, 0] - 0.5).std().item() > 0.05
    term.reset(torch.tensor([0]))


# --- eval callback ----------------------------------------------------------


def test_eval_callback_records_episodes(tmp_path, monkeypatch):
    from wbt_training.utils import kick_eval_callback as K

    n = 2
    root = torch.zeros(n, 13)
    root[:, 2], root[:, 6], root[:, 7] = 0.57, 1.0, 0.4
    ball = torch.zeros(n, 13)
    ball[:, 0] = 1.0
    cmd = SimpleNamespace(
        kick_phase=torch.zeros(n),
        contact_heading=torch.full((n,), float("nan")),
        motion=SimpleNamespace(_vel_cmd=torch.tensor([[0.4, 0.0]])),
        time_steps=torch.zeros(n, dtype=torch.long),
    )
    env = SimpleNamespace(num_envs=n, device="cpu", dt=0.02, simulator=SimpleNamespace(robot_root_states=root), command_manager=SimpleNamespace(get_state=lambda _n: cmd))

    class FakeDetector:
        def __init__(self, env):
            self.resets = []

        def update(self):
            return torch.tensor([True, False])

        def reset(self, ids):
            self.resets.append(ids.tolist())

    import wbt_training.config_values.lupus_kick as LK

    monkeypatch.setattr(LK, "BallContactDetector", FakeDetector)
    monkeypatch.setattr(LK, "ball_state_w", lambda env, name="ball": ball)
    loop = SimpleNamespace(env=env, device="cpu", log_dir=str(tmp_path), eval_callbacks=[])
    cb = K.attach_kick_metrics(loop, K.KickMetricsConfig(enabled=True))
    assert loop.eval_callbacks == [cb]
    for _ in range(3):
        cb.on_post_eval_env_step({"dones": torch.zeros(n, dtype=torch.bool)})
    cb.on_post_eval_env_step({"dones": torch.tensor([True, True]), "extras": {"time_outs": torch.tensor([True, False])}})
    cb.on_post_evaluate_policy()
    data = json.loads((tmp_path / "kick_metrics.json").read_text())
    a, b = data["records"]
    assert a["miskick"] and not b["miskick"]
    assert not a["terminated"] and b["terminated"]
    assert a["v_err"] == pytest.approx(0.0, abs=1e-6)
    assert K.attach_kick_metrics(loop, K.KickMetricsConfig(enabled=False)) is None


def test_jitter_flat_params_are_cli_overridable():
    from wbt_training.config_values import experiment as E
    from wbt_training.config_values.kick_command_obs import StudentCommandObsTerm
    from wbt_training.utils.checkpoint import parse_saved_config_overrides

    P = "--observation.groups.actor-obs.terms.command.params"
    cfg = parse_saved_config_overrides(E.lupus_kick_distill_finetune, [f"{P}.jitter-enabled", "True", f"{P}.jitter-update-hz", "10"])
    params = cfg.observation.groups["actor_obs"].terms["command"].params
    assert params["jitter_enabled"] is True and params["jitter_update_hz"] == 10.0
    n = 4
    motion = SimpleNamespace(_vel_cmd=torch.zeros(5, 2), _kick_enable=torch.zeros(5), _kick_phase=torch.zeros(5))
    cmd = SimpleNamespace(motion=motion, time_steps=torch.zeros(n, dtype=torch.long))
    env = SimpleNamespace(num_envs=n, device="cpu", dt=0.02, command_manager=SimpleNamespace(get_state=lambda _n: cmd))
    term = StudentCommandObsTerm(SimpleNamespace(params=params), env)
    assert term.jitter is not None and term.jitter.cfg.update_hz == 10.0 and term.jitter.residuals is None
    off = StudentCommandObsTerm(SimpleNamespace(params=E.lupus_kick_distill_finetune.observation.groups["actor_obs"].terms["command"].params), env)
    assert off.jitter is None
