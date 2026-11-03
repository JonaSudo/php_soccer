"""LUPUS holosoma robot config: consistency with robots/lupus and a MuJoCo rollout."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import tyro
from holosoma.utils.tyro_utils import TYRO_CONIFG

from wbt_training.config_values import experiment
from wbt_training.config_values import lupus as L

LUPUS_DIR = Path(__file__).resolve().parents[2] / "robots" / "lupus"
HAVE_MESHES = (LUPUS_DIR / "meshes").exists()


def test_cli_selects_lupus_loco():
    cfg = tyro.cli(experiment.AnnotatedExperimentConfig, args=["exp:lupus-loco"], config=TYRO_CONIFG)
    assert cfg == experiment.DEFAULTS["lupus_loco"]
    assert cfg.robot == L.lupus_22dof


def test_dof_lists_have_one_entry_per_joint():
    r = L.lupus_22dof
    n = len(r.dof_names)
    assert n == r.actions_dim == r.dof_obs_size == 22
    for lst in (
        r.dof_pos_lower_limit_list,
        r.dof_pos_upper_limit_list,
        r.dof_vel_limit_list,
        r.dof_effort_limit_list,
        r.dof_armature_list,
        r.dof_joint_friction_list,
    ):
        assert len(lst) == n
    assert set(r.init_state.default_joint_angles) == set(r.dof_names)
    assert len(L.lupus_loco.reward.terms["pose"].params["pose_weights"]) == n
    assert r.dof_names[-2:] == L.NECK_DOF_NAMES  # neck last, policy joints first


def test_every_dof_has_gains_and_default_within_limits():
    r = L.lupus_22dof
    for i, name in enumerate(r.dof_names):
        assert sum(k in name for k in r.control.stiffness) >= 1, name
        assert sum(k in name for k in r.control.damping) >= 1, name
        assert r.dof_pos_lower_limit_list[i] <= r.init_state.default_joint_angles[name] <= r.dof_pos_upper_limit_list[i]
    # inverted-pendulum bound from robots/lupus/README.md
    assert r.control.stiffness["ankle_pitch"] > 50


def test_symmetry_flips_roll_and_yaw_like_motion_matching():
    # Same rule as LUPUS_SPEC.mirror_negate_joints in motion matching.
    assert L.FLIP_SIGN_JOINT_NAMES == [n for n in L.DOF_NAMES if n.endswith(("_roll", "_yaw"))]
    for a, b in L.SYMMETRY_JOINT_NAMES.items():
        assert L.SYMMETRY_JOINT_NAMES[b] == a


mujoco = pytest.importorskip("mujoco") if HAVE_MESHES else None


def _without_randomization(rand):
    """Same managers, every randomization switched off (nominal model)."""

    def off(term):
        params = dict(term.params)
        for key in list(params):
            if key == "enabled" or key.startswith("enable_"):
                params[key] = False
        if "joint_pos_scale_range" in params:
            params["joint_pos_scale_range"] = [1.0, 1.0]
        return dataclasses.replace(term, params=params)

    return dataclasses.replace(
        rand,
        setup_terms={k: off(t) for k, t in rand.setup_terms.items()},
        reset_terms={k: off(t) for k, t in rand.reset_terms.items()},
    )


@pytest.mark.skipif(mujoco is None, reason="CAD meshes or mujoco not present")
def test_names_order_and_base_height_match_mjcf():
    m = mujoco.MjModel.from_xml_path(str(LUPUS_DIR / "lupus.xml"))
    assert [m.joint(i).name for i in range(1, m.njnt)] == L.DOF_NAMES
    assert [m.body(i).name for i in range(1, m.nbody)] == L.BODY_NAMES
    d = mujoco.MjData(m)
    for name, q in L.DEFAULT_JOINT_ANGLES.items():
        d.qpos[m.jnt_qposadr[m.joint(name).id]] = q
    mujoco.mj_kinematics(m, d)
    sole = min(d.xpos[m.body(f"{s}_foot_contact_point").id][2] for s in ("left", "right"))
    standing_height = d.qpos[2] - sole
    assert L.DEFAULT_BASE_HEIGHT - standing_height == pytest.approx(0.01, abs=2e-3)


@pytest.mark.skipif(mujoco is None, reason="CAD meshes or mujoco not present")
def test_lupus_loco_env_stands_in_mujoco_and_ignores_neck_actions():
    import torch
    from holosoma.config_values import simulator, terrain
    from holosoma.train_agent import get_tyro_env_config
    from holosoma.utils.helpers import get_class
    from holosoma.utils.sim_utils import setup_simulator_imports

    cfg = dataclasses.replace(
        L.lupus_loco,
        simulator=simulator.mujoco,
        terrain=terrain.terrain_locomotion_plane,
        # nominal model, deterministic start
        randomization=_without_randomization(L.lupus_loco.randomization),
        training=dataclasses.replace(L.lupus_loco.training, num_envs=1, headless=True),
    )
    setup_simulator_imports(cfg)
    env = get_class(cfg.env_class)(get_tyro_env_config(cfg), device="cpu")
    env.reset_all()
    dones = []
    # 1 s at 50 Hz. A pure PD hold of the crouched default pose is only
    # marginally stable (slow pitch oscillation, falls after 1.5-3 s); keeping
    # balance is the policy's job, this checks model, contacts and neck masking.
    for k in range(50):
        actions = torch.zeros(1, 22)
        actions[:, -2:] = 5.0  # would be +1.25 rad on the neck if driven
        _, _, done, _ = env.step({"actions": actions})
        if done.any():
            dones.append(k)
    assert dones == []
    assert env.simulator.robot_root_states[0, 2].item() > 0.53
    assert env.simulator.dof_pos[0, -2:].abs().max().item() < 0.05


def test_lupus_tracking_npz_converts_and_loads_by_name(tmp_path):
    """motion_matching qpos_to_tracking(robot="lupus") layout -> motion_convert -> holosoma MotionLoader."""
    import numpy as np
    from holosoma.managers.command.terms.wbt import MotionLoader

    from wbt_training.training_runs.motion_convert import convert

    T, nb = 3, len(L.BODY_NAMES)
    joint_pos = np.arange(T * 22, dtype=float).reshape(T, 22) * 1e-3
    body_pos = np.arange(T * nb * 3, dtype=float).reshape(T, nb, 3)
    source, target = tmp_path / "lupus_tracking.npz", tmp_path / "lupus_holosoma.npz"
    np.savez(
        source,
        fps=np.array([50]),
        joint_pos=joint_pos,
        joint_vel=joint_pos + 1.0,
        body_pos_w=body_pos,
        body_quat_w=np.tile([1.0, 0.0, 0.0, 0.0], (T, nb, 1)),
        body_lin_vel_w=np.zeros((T, nb, 3)),
        body_ang_vel_w=np.zeros((T, nb, 3)),
        joint_names=np.array(L.DOF_NAMES),
        body_names=np.array(L.BODY_NAMES),
    )
    assert convert(str(source), str(target))
    with np.load(target, allow_pickle=False) as data:
        assert data["joint_pos"].shape == (T, 7 + 22)
        assert data["joint_vel"].shape == (T, 6 + 22)
        np.testing.assert_array_equal(data["joint_pos"][:, :3], body_pos[:, 0])
        np.testing.assert_array_equal(data["joint_pos"][:, 7:], joint_pos)

    motion = MotionLoader(str(target), L.BODY_NAMES, L.DOF_NAMES)
    np.testing.assert_allclose(motion._joint_pos.numpy(), joint_pos, rtol=1e-6)
    assert motion._body_indexes.tolist() == list(range(nb))
