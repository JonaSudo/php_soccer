"""LUPUS Sim2Sim runner: same observation as holosoma's env, ONNX round trip, CPU only."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from wbt_training.config_values import lupus as L
from wbt_training.sim2sim import lupus_sim2sim as S

LUPUS_DIR = Path(__file__).resolve().parents[2] / "robots" / "lupus"
HAVE_MESHES = (LUPUS_DIR / "meshes").exists()
mujoco = pytest.importorskip("mujoco") if HAVE_MESHES else None
needs_model = pytest.mark.skipif(mujoco is None, reason="CAD meshes or mujoco not present")


def test_constants_match_holosoma_config():
    r = L.lupus_22dof
    assert list(S.NECK_DOF_NAMES) == L.NECK_DOF_NAMES
    defaults = {n: S.DEFAULT_JOINT_ANGLES.get(n, 0.0) for n in L.DOF_NAMES}
    assert defaults == L.DEFAULT_JOINT_ANGLES
    np.testing.assert_allclose(S.gains_for(L.DOF_NAMES, S.STIFFNESS), S.gains_for(L.DOF_NAMES, r.control.stiffness))
    np.testing.assert_allclose(S.gains_for(L.DOF_NAMES, S.DAMPING), S.gains_for(L.DOF_NAMES, r.control.damping))
    cfg = S.LupusSim2SimConfig()
    assert cfg.action_scale == r.control.action_scale
    assert cfg.action_clip == r.control.action_clip_value
    assert cfg.init_base_height == L.DEFAULT_BASE_HEIGHT


def test_obs_terms_are_the_actor_group_in_alphabetical_order():
    group = L.lupus_loco.observation.groups["actor_obs"]
    assert group.history_length == 1 and group.concatenate
    assert [n for n, _ in S.OBS_TERMS] == sorted(group.terms)
    assert {n: s for n, s in S.OBS_TERMS} == {n: t.scale for n, t in group.terms.items()}


def test_gait_clock_matches_locomotion_gait_eval_mode():
    clock = S.GaitClock(policy_dt=0.02, period=1.0, stand_value=np.pi)
    walk = np.array([0.5, 0.0, 0.0])
    clock.update(0, walk)
    np.testing.assert_allclose(clock.phase, [0.0, -np.pi])
    clock.update(25, walk)  # half a period
    np.testing.assert_allclose(clock.phase, [-np.pi, 0.0], atol=1e-9)
    clock.update(10, np.zeros(3))
    np.testing.assert_allclose(clock.phase, [np.pi, np.pi])


def _holosoma_env():
    import torch
    from holosoma.config_values import simulator, terrain
    from holosoma.train_agent import get_tyro_env_config
    from holosoma.utils.helpers import get_class
    from holosoma.utils.sim_utils import setup_simulator_imports

    from wbt_training.tests.test_lupus_config import _without_randomization

    obs = L.lupus_loco.observation
    obs = dataclasses.replace(
        obs, groups={k: dataclasses.replace(g, enable_noise=False) for k, g in obs.groups.items()}
    )
    cfg = dataclasses.replace(
        L.lupus_loco,
        simulator=simulator.mujoco,
        terrain=terrain.terrain_locomotion_plane,
        randomization=_without_randomization(L.lupus_loco.randomization),
        observation=obs,
        training=dataclasses.replace(L.lupus_loco.training, num_envs=1, headless=True),
    )
    setup_simulator_imports(cfg)
    env = get_class(cfg.env_class)(get_tyro_env_config(cfg), device="cpu")
    env.set_is_evaluating()
    env.reset_all()
    return env, torch


@needs_model
def test_observation_and_pd_match_holosoma_mujoco_env():
    env, torch = _holosoma_env()
    command = [0.5, 0.0, 0.3]
    env.command_manager.commands[:, :3] = torch.tensor(command)
    hd = env.simulator.root_data

    sim = S.LupusSim2Sim()
    sim.set_command(*command)
    sim.reset()
    assert sim.dof_names == L.DOF_NAMES
    assert sim.obs_dim == 79

    def sync_state():
        sim.data.qpos[:] = hd.qpos[: sim.model.nq]
        sim.data.qpos[:2] -= hd.qpos[:2]  # holosoma places the env at a terrain origin
        sim.data.qvel[:] = hd.qvel[: sim.model.nv]
        mujoco.mj_forward(sim.model, sim.data)
        sim.step_count = int(env.episode_length_buf[0])
        sim.gait.update(sim.step_count, sim.command)

    rng = np.random.default_rng(0)
    one_step_err = []
    for _ in range(30):
        action = rng.normal(0.0, 0.5, 22).astype(np.float32)
        action[-2:] = 3.0  # neck outputs must be ignored by PD, but observed
        sync_state()
        obs_dict, *_ = env.step({"actions": torch.tensor(action)[None]})
        sim.step(action)
        one_step_err.append(np.abs(sim.dof_pos - hd.qpos[sim.qpos_adr]).max())
        sync_state()
        np.testing.assert_allclose(sim.observation(), obs_dict["actor_obs"][0].numpy(), atol=1e-5)
    # Same PD, decimation and torque clip. Not bit-exact: holosoma's MuJoCo
    # scene adds its own floor next to lupus.xml's, so contacts differ slightly.
    assert max(one_step_err) < 0.06


def _export_policy(path: Path, action_bias: np.ndarray | None = None, dof_names=None) -> Path:
    import torch
    from holosoma.utils.inference_helpers import (
        attach_onnx_metadata,
        export_policy_as_onnx,
        get_control_gains_from_config,
    )

    net = torch.nn.Sequential(torch.nn.Linear(79, 32), torch.nn.ELU(), torch.nn.Linear(32, 22))
    with torch.no_grad():
        net[2].weight.zero_()
        net[2].bias.copy_(torch.as_tensor(action_bias if action_bias is not None else np.zeros(22)))
    export_policy_as_onnx(net, str(path), {"actor_obs": torch.zeros(1, 79)})
    kp, kd = get_control_gains_from_config(L.lupus_22dof)
    attach_onnx_metadata(
        str(path),
        {
            "dof_names": dof_names or L.lupus_22dof.dof_names,
            "kp": kp,
            "kd": kd,
            "action_scale": L.lupus_22dof.control.action_scale,
        },
    )
    return path


@needs_model
def test_onnx_policy_round_trip_and_metadata(tmp_path):
    pytest.importorskip("onnxruntime")
    bias = np.zeros(22)
    bias[-2:] = 4.0  # neck: must stay passive
    policy, meta = S.load_onnx_policy(_export_policy(tmp_path / "p.onnx", bias))
    assert json.loads(meta["dof_names"]) == L.DOF_NAMES

    sim = S.LupusSim2Sim()
    sim.configure(S.apply_onnx_metadata(sim.cfg, meta, sim.dof_names))
    np.testing.assert_allclose(sim.kp, S.gains_for(L.DOF_NAMES, L.STIFFNESS))
    res = S.rollout(sim, policy, (0.0, 0.0, 0.0), seconds=1.0)
    assert res.steps == 50 and not res.fell
    assert np.abs(sim.dof_pos[-2:]).max() < 0.05
    assert res.max_torque_ratio <= 1.0 + 1e-9


@needs_model
def test_onnx_with_wrong_dof_order_is_rejected(tmp_path):
    pytest.importorskip("onnxruntime")
    names = list(reversed(L.DOF_NAMES))
    _, meta = S.load_onnx_policy(_export_policy(tmp_path / "p.onnx", dof_names=names))
    sim = S.LupusSim2Sim()
    with pytest.raises(ValueError, match="DoF order"):
        S.apply_onnx_metadata(sim.cfg, meta, sim.dof_names)


@needs_model
def test_cli_runs_pd_hold(capsys):
    assert S.main(["--seconds", "0.5"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["steps"] == 25 and out["fell"] is False


@needs_model
def test_cpu_training_export_runs_in_sim2sim(tmp_path):
    """Phase 0 pipeline on CPU: train exp:lupus-loco (MuJoCo, 1 env) -> ONNX -> Sim2Sim."""
    import subprocess
    import sys

    pytest.importorskip("onnxruntime")
    repo = Path(__file__).resolve().parents[2]
    cmd = [
        sys.executable,
        str(repo / "wbt_training" / "train_agent.py"),
        "exp:lupus-loco",
        "simulator:mujoco",
        "terrain:terrain-locomotion-plane",
        "--training.num-envs", "1",
        "--training.headless", "True",
        "--algo.config.num-learning-iterations", "1",
        "--algo.config.num-steps-per-env", "8",
        "--algo.config.save-interval", "1",
        "--logger.type", "disabled",
    ]  # fmt: skip
    proc = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    onnx_files = sorted(tmp_path.glob("logs/**/model_*.onnx"))
    assert onnx_files, proc.stdout[-3000:]

    policy, meta = S.load_onnx_policy(onnx_files[-1])
    sim = S.LupusSim2Sim()
    sim.configure(S.apply_onnx_metadata(sim.cfg, meta, sim.dof_names))
    assert policy(sim.reset()).shape == (22,)
    res = S.rollout(sim, policy, (0.5, 0.0, 0.0), seconds=0.5, stop_on_fall=False)
    assert res.steps == 25
    assert np.isfinite([res.mean_vx, res.mean_wz, res.max_torque_ratio]).all()
