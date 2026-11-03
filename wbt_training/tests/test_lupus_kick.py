"""LUPUS kick teacher (Phase 3): ball asset, head scan, kick fields, ball rewards, preset.

Everything here runs on CPU. The last test builds the full teacher env in
MuJoCo (CPU backend, 1 env) from a synthetic standing motion with a ball.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import tyro
from holosoma.utils.tyro_utils import TYRO_CONIFG

from wbt_training.config_values import experiment
from wbt_training.config_values import lupus as L
from wbt_training.config_values import lupus_kick as K
from wbt_training.utils import head_scan as H

REPO = Path(__file__).resolve().parents[2]
HAVE_MESHES = (REPO / "robots" / "lupus" / "meshes").exists()


# ---------------------------------------------------------------- ball asset
def test_ball_files_are_built_from_yaml():
    import importlib.util

    spec = importlib.util.spec_from_file_location("bb", REPO / "robots" / "ball" / "build_ball.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    s = mod.load_spec()
    assert (REPO / "robots/ball/ball.urdf").read_text() == mod.urdf_text(s), "rerun robots/ball/build_ball.py"
    assert (REPO / "robots/ball/ball.xml").read_text() == mod.mjcf_text(s), "rerun robots/ball/build_ball.py"
    assert K.BALL_RADIUS == pytest.approx(s["diameter"] / 2)
    # thin shell: I = 2/3 m r^2
    assert s["inertia"] == pytest.approx(2 / 3 * s["mass"] * s["radius"] ** 2)


def test_ball_mjcf_loads_and_rolls_on_a_plane():
    mujoco = pytest.importorskip("mujoco")
    xml = (REPO / "robots/ball/ball.xml").read_text()
    body = xml.split("<worldbody>")[1].split("</worldbody>")[0].replace('<body name="ball">', '<body name="ball" pos="0 0 0.075"><freejoint/>', 1)
    m = mujoco.MjModel.from_xml_string(f'<mujoco><worldbody><geom type="plane" size="5 5 .1"/>{body}</worldbody></mujoco>')
    d = mujoco.MjData(m)
    d.qvel[0] = 2.0
    for _ in range(500):
        mujoco.mj_step(m, d)
    assert d.qpos[2] == pytest.approx(K.BALL_RADIUS, abs=2e-3)  # stays on the ground
    assert d.qpos[0] > 0.3  # rolled forward


# ------------------------------------------------------------- head scan math
def test_triangle_wave_has_constant_speed_and_unit_amplitude():
    t = torch.linspace(0, 4, 4001)
    w = H.triangle_wave(t, torch.tensor(2.0), torch.tensor(0.0))
    assert w.max().item() == pytest.approx(1.0, abs=1e-3)
    assert w.min().item() == pytest.approx(-1.0, abs=1e-3)
    slope = (w[1:] - w[:-1]).abs() / (t[1] - t[0])
    assert torch.allclose(slope[(slope > 0.1)], torch.tensor(2.0), atol=0.05)  # 4 / period


def test_look_at_and_rate_limit():
    head = torch.tensor([[0.0, 0.0, 0.4]])
    yaw, pitch = H.look_at_angles(torch.tensor([[1.0, 1.0, 0.4]]), head)
    assert yaw.item() == pytest.approx(math.pi / 4) and pitch.item() == pytest.approx(0.0, abs=1e-6)
    yaw, pitch = H.look_at_angles(torch.tensor([[0.4, 0.0, 0.0]]), head)
    assert pitch.item() == pytest.approx(math.pi / 4)  # ball below: positive = look down
    out = H.rate_limit(torch.zeros(2), torch.tensor([1.0, -0.05]), 0.1)
    assert out.tolist() == pytest.approx([0.1, -0.05])


@pytest.mark.skipif(not HAVE_MESHES, reason="CAD meshes not present")
def test_head_sign_conventions_match_lupus_fk():
    mujoco = pytest.importorskip("mujoco")
    m = mujoco.MjModel.from_xml_path(str(REPO / "robots/lupus/lupus.xml"))
    d = mujoco.MjData(m)

    def head_x(yaw, pitch):
        d.qpos[:] = 0
        d.qpos[3] = 1
        d.qpos[m.jnt_qposadr[m.joint("head_yaw").id]] = yaw
        d.qpos[m.jnt_qposadr[m.joint("head_pitch").id]] = pitch
        mujoco.mj_kinematics(m, d)
        return d.xmat[m.body("head_link").id].reshape(3, 3)[:, 0]

    assert head_x(0.5, 0)[1] * H.DEFAULT_PARAMS["yaw_left_sign"] > 0.4  # positive yaw looks left
    assert head_x(0, 0.5)[2] * H.DEFAULT_PARAMS["pitch_down_sign"] < -0.4  # positive pitch looks down


# --------------------------------------------------------------- fake env
class _Sampler:
    """Deterministic stand-in for holosoma's TermSampler (uniform draws)."""

    def __init__(self, seed=0):
        self.g = torch.Generator().manual_seed(seed)

    def draw(self, spec, *, env_ids, coords=(), device="cpu"):
        lo, hi = spec
        return lo + (hi - lo) * torch.rand(len(env_ids), generator=self.g)


class _Registry:
    def __init__(self, names):
        self.names = names

    def get_names_by_type(self, _type):
        return list(self.names)


def _fake_env(n=4, ball_xy=(0.5, 0.0), kick=0.0, with_ball=True):
    dof = len(L.DOF_NAMES)
    body_names = list(L.BODY_NAMES)
    nb = len(body_names)
    body_pos = torch.zeros(n, nb, 3)
    body_pos[:, :, 2] = 0.3
    body_pos[:, body_names.index("head_link")] = torch.tensor([0.0, 0.0, 0.4185 + 0.573])
    body_rot = torch.zeros(n, nb, 4)
    body_rot[..., 3] = 1.0
    root = torch.zeros(n, 13)
    root[:, 2] = 0.573
    root[:, 6] = 1.0
    ball = torch.zeros(n, 13)
    ball[:, 0], ball[:, 1], ball[:, 2] = ball_xy[0], ball_xy[1], K.BALL_RADIUS
    ball[:, 6] = 1.0
    term = SimpleNamespace(passive_targets=torch.zeros(n, 2))

    sim = SimpleNamespace(
        dof_pos=torch.zeros(n, dof),
        body_names=body_names,
        _rigid_body_pos=body_pos,
        _rigid_body_rot=body_rot,
        robot_root_states=root,
        contact_forces=torch.zeros(n, nb, 3),
        object_registry=_Registry([K.BALL_NAME] if with_ball else []),
        ball=ball,
    )
    sim.get_actor_states = lambda names, env_ids: sim.ball[env_ids]
    cmd = SimpleNamespace(
        kick_enable=torch.full((n,), kick),
        kick_phase=torch.zeros(n, dtype=torch.long),
        frames_since_contact=torch.full((n,), -1, dtype=torch.long),
        contact_heading=torch.full((n,), float("nan")),
        motion_end_reset=torch.zeros(n, dtype=torch.bool),
    )
    env = SimpleNamespace(
        num_envs=n,
        device="cpu",
        dt=0.02,
        robot_config=L.lupus_22dof,
        simulator=sim,
        episode_length_buf=torch.zeros(n, dtype=torch.long),
        action_manager=SimpleNamespace(get_term=lambda name: term),
        command_manager=SimpleNamespace(get_state=lambda name: cmd),
    )
    return env, term, cmd


def _head_scan(env, **params):
    gen = H.HeadScanGenerator(SimpleNamespace(params=params), env)
    gen.setup(_Sampler())
    gen.reset(None, _Sampler(1))
    return gen


def test_head_scan_sweeps_within_limits_and_speed():
    env, term, _ = _fake_env(n=8, with_ball=False)
    gen = _head_scan(env, hold_prob=0.0, track_prob=0.0)
    yaws = []
    prev = None
    for k in range(400):
        env.episode_length_buf[:] = k
        gen.step(None)
        a = gen.angles.clone()
        assert torch.all(a >= gen.lower - 1e-6) and torch.all(a <= gen.upper + 1e-6)
        if prev is not None:
            assert (a - prev).abs().max().item() <= gen.p["max_speed"] * env.dt + 1e-6
        prev = a
        yaws.append(a[:, 0])
    yaws = torch.stack(yaws)
    # sweeps both ways (amplitude >= 0.5 rad)
    assert (yaws.max(0).values - yaws.min(0).values).min().item() > 0.8
    # written as offset from default / action_scale into the passive neck targets
    scale = L.lupus_22dof.control.action_scale
    assert torch.allclose(term.passive_targets, (gen.angles - gen.default) / scale)


def test_head_scan_tracks_ball_when_kick_is_pending():
    env, _, _ = _fake_env(n=2, ball_xy=(0.4, 0.3), kick=1.0)
    gen = _head_scan(env, hold_prob=0.0, track_prob=0.0, track_on_kick_prob=1.0)
    for k in range(100):
        env.episode_length_buf[:] = k
        gen.step(None)
    assert gen.tracking.all()
    assert gen.angles[:, 0] == pytest.approx(torch.full((2,), math.atan2(0.3, 0.4)), abs=1e-3)
    assert torch.all(gen.angles[:, 1] > 0.3)  # looks down at the ball


def test_head_scan_without_tracking_ignores_ball():
    env, _, _ = _fake_env(n=2, ball_xy=(0.4, 0.3), kick=1.0)
    gen = _head_scan(env, hold_prob=1.0, track_prob=0.0, track_on_kick_prob=0.0)
    for k in range(100):
        env.episode_length_buf[:] = k
        gen.step(None)
    assert not gen.tracking.any()
    assert torch.allclose(gen.angles[:, 0], gen.ep["yaw_center"], atol=1e-5)


# ------------------------------------------------------------ kick fields
def test_frames_since_contact_resets_per_clip():
    contact = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=bool)
    start = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=bool)
    assert K.frames_since_contact(contact, start).tolist() == [-1, -1, 0, 1, -1, -1, -1, 0, 1]


def _write_motion(path, n=150, contact=100, kick=True, ball_x=0.30):
    """Synthetic standing LUPUS motion in holosoma format (+ ball and kick fields)."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(REPO / "robots/lupus/lupus.xml"))
    d = mujoco.MjData(m)
    d.qpos[2] = L.DEFAULT_BASE_HEIGHT - 0.01
    d.qpos[3] = 1.0
    for name, q in L.DEFAULT_JOINT_ANGLES.items():
        d.qpos[m.jnt_qposadr[m.joint(name).id]] = q
    mujoco.mj_kinematics(m, d)
    body_names = [b for b in L.BODY_NAMES if not b.endswith("contact_point")]  # as retargeted motions
    ids = [m.body(b).id for b in body_names]
    pos = np.repeat(d.xpos[ids][None], n, 0)
    quat = np.repeat(d.xquat[ids][None], n, 0)  # wxyz
    jp = np.concatenate([d.qpos[:7], [d.qpos[m.jnt_qposadr[m.joint(j).id]] for j in L.DOF_NAMES]])
    phase = np.zeros(n, dtype=np.int8)
    if kick:
        phase[contact - 50 : contact] = 1
        phase[contact:] = 2
    cf = np.zeros(n, dtype=bool)
    if kick:
        cf[contact] = True
    np.savez(
        path,
        fps=np.array(50),
        joint_names=np.array(L.DOF_NAMES),
        body_names=np.array(body_names),
        joint_pos=np.repeat(jp[None], n, 0).astype(np.float32),
        joint_vel=np.zeros((n, 6 + len(L.DOF_NAMES)), np.float32),
        body_pos_w=pos.astype(np.float32),
        body_quat_w=quat.astype(np.float32),
        body_lin_vel_w=np.zeros_like(pos, dtype=np.float32),
        body_ang_vel_w=np.zeros_like(pos, dtype=np.float32),
        object_pos_w=np.repeat([[ball_x, 0.0, K.BALL_RADIUS]], n, 0).astype(np.float32),
        object_quat_w=np.repeat([[1.0, 0, 0, 0]], n, 0).astype(np.float32),
        object_lin_vel_w=np.zeros((n, 3), np.float32),
        kick_enable=np.full(n, 1.0 if kick else 0.0, np.float32),
        kick_phase=phase,
        contact_frame=cf,
    )
    return path


@pytest.mark.skipif(not HAVE_MESHES, reason="CAD meshes not present")
def test_load_kick_fields_concatenates_and_tolerates_missing(tmp_path):
    a = _write_motion(tmp_path / "a.npz", n=120, contact=80)
    b = tmp_path / "b.npz"
    with np.load(a) as data:
        np.savez(b, **{k: data[k] for k in data.files if k not in ("kick_enable", "kick_phase", "contact_frame")})
    f = K.load_kick_fields([str(a), str(b)], 240)
    assert f["kick_phase"][79] == 1 and f["kick_phase"][80] == 2 and f["kick_phase"][120:].max() == 0
    assert f["frames_since_contact"][80] == 0 and f["frames_since_contact"][119] == 39
    assert (f["frames_since_contact"][120:] == -1).all()  # contact does not leak into the next clip
    with pytest.raises(ValueError, match="prepend"):
        K.load_kick_fields([str(a)], 121)


# ------------------------------------------------------------- ball geometry
def test_closest_point_on_axis_aligned_box():
    box = {"center": np.zeros(3), "rot": np.eye(3), "half": np.array([0.15, 0.05, 0.015])}
    link = torch.tensor([[0.0, 0.0, 0.0]])
    q = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    p = K.closest_point_on_box(torch.tensor([[0.5, 0.0, 0.0]]), link, q, box)
    assert p.tolist()[0] == pytest.approx([0.15, 0.0, 0.0])
    # rotated link (90 deg yaw): box long axis now along world y
    qz = torch.tensor([[0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]])
    p = K.closest_point_on_box(torch.tensor([[0.0, 0.5, 0.0]]), link, qz, box)
    assert p.tolist()[0] == pytest.approx([0.0, 0.15, 0.0], abs=1e-6)


def test_foot_boxes_are_the_lupus_sole_plates():
    for name in K.FOOT_BODY_NAMES:
        b = K.FOOT_BOXES[name]
        assert sorted((2 * b["half"]).round(3).tolist()) == [0.03, 0.11, 0.3]
        assert np.allclose(b["rot"] @ b["rot"].T, np.eye(3), atol=1e-9)


def _foot_on_ball(env, foot="left_foot_link", above=True):
    """Place a foot so its sole box touches the ball from above (or from behind)."""
    sim = env.simulator
    i = sim.body_names.index(foot)
    box = K.FOOT_BOXES[foot]
    # sole center in world with identity link orientation
    ball = sim.ball[0, :3]
    lowest_face = torch.tensor(box["rot"] @ np.array([0, 0, 0]) + box["center"], dtype=torch.float32)
    # half extent of the box along world z (identity link orientation)
    hz = float(np.abs(box["rot"]).dot(box["half"])[2])
    hx = float(np.abs(box["rot"]).dot(box["half"])[0])
    if above:
        target_center = ball + torch.tensor([0.0, 0.0, K.BALL_RADIUS + hz])
    else:
        target_center = ball - torch.tensor([K.BALL_RADIUS + hx, 0.0, 0.0])
    sim._rigid_body_pos[:, i] = target_center - lowest_face


def test_ball_step_on_only_counts_feet_on_top_with_force():
    env, _, _ = _fake_env(n=1)
    term = K.BallStepOn(SimpleNamespace(params={}), env)
    fid = env.simulator.body_names.index("left_foot_link")
    assert term(env).item() == 0.0  # feet far away
    _foot_on_ball(env, above=True)
    assert term.on_ball(env)[0, 0]
    assert term(env).item() == 0.0  # no vertical force yet
    env.simulator.contact_forces[:, fid, 2] = 50.0
    assert term(env).item() == pytest.approx(1.0)
    env2, _, _ = _fake_env(n=1)
    _foot_on_ball(env2, above=False)  # side contact (kick) is fine
    env2.simulator.contact_forces[:, fid, 2] = 50.0
    assert not K.BallStepOn(SimpleNamespace(params={}), env2).on_ball(env2).any()


def test_ball_untouched_needs_velocity_jump_near_robot_and_phase_zero():
    env, _, cmd = _fake_env(n=1)
    term = K.BallUntouched(SimpleNamespace(params={}), env)
    env.episode_length_buf[:] = 10
    assert term(env).item() == 0.0  # first call only primes the detector
    _foot_on_ball(env, above=False)
    env.simulator.ball[:, 7] = 1.0  # ball got hit
    assert term(env).item() == 1.0
    env.simulator.ball[:, 7] = 2.0
    cmd.kick_phase[:] = 1  # during a kick contacts are allowed
    assert term(env).item() == 0.0
    # far away robot: a jump in ball velocity (e.g. bounce) is not a robot contact
    env3, _, _ = _fake_env(n=1, ball_xy=(3.0, 0.0))
    t3 = K.BallUntouched(SimpleNamespace(params={}), env3)
    env3.episode_length_buf[:] = 10
    t3(env3)
    env3.simulator.ball[:, 7] = 1.0
    assert t3(env3).item() == 0.0
    # reset invalidates the previous velocity (teleport)
    t3.reset(torch.tensor([0]))
    _foot_on_ball(env3, above=False)
    env3.simulator.ball[:, 7] = 5.0
    assert t3(env3).item() == 0.0


def test_ball_kick_direction_rewards_heading_and_caps_speed():
    env, _, cmd = _fake_env(n=4)
    cmd.frames_since_contact = torch.tensor([5, 5, 5, 100])
    cmd.contact_heading = torch.tensor([0.0, math.pi / 2, float("nan"), 0.0])
    env.simulator.ball[:, 7] = torch.tensor([3.0, 3.0, 10.0, 3.0])
    r = K.ball_kick_direction(env, window_s=0.5, max_speed=4.0, lateral_weight=1.0)
    assert r[0].item() == pytest.approx(3.0)  # straight along heading
    assert r[1].item() == pytest.approx(-3.0)  # all sideways
    assert r[2].item() == pytest.approx(4.0)  # no latch: current heading (0), speed capped
    assert r[3].item() == 0.0  # outside the 0.5 s window


# ------------------------------------------------------------------ preset
def test_cli_selects_lupus_kick_teacher():
    cfg = tyro.cli(experiment.AnnotatedExperimentConfig, args=["exp:lupus-kick-teacher"], config=TYRO_CONIFG)
    assert cfg == experiment.DEFAULTS["lupus_kick_teacher"]
    assert cfg.robot.control.action_scale == 1.0
    assert cfg.algo.config.num_learning_iterations == 20000


def test_preset_matches_the_plan():
    cfg = experiment.lupus_kick_teacher
    assert cfg.robot.dof_names == L.DOF_NAMES
    assert cfg.action == L.lupus_joint_pos  # neck stays passive
    assert list(cfg.scene.rigid_objects) == [K.BALL_NAME]
    actor = cfg.observation.groups["actor_obs"].terms
    assert {"object_pos_b", "object_lin_vel_b", "kick_enable", "motion_ref_pos_b"} <= set(actor)
    assert "height_scan" not in actor
    rew = cfg.reward.terms
    assert not any(k.startswith("object_global_ref") for k in rew)
    assert rew["ball_kick_direction"].weight > 0 and rew["ball_untouched"].weight < 0 and rew["ball_step_on"].weight < 0
    bt = cfg.termination.terms["bad_tracking"].params
    assert bt["bad_ref_pos_threshold"] == 0.5 and bt["bad_object_pos_threshold"] > 100
    mc = cfg.command.setup_terms["motion_command"].params["motion_config"]
    assert mc.use_adaptive_timesteps_sampler
    assert not mc.enable_default_pose_prepend and not mc.enable_default_pose_append
    for name in mc.body_names_to_track + bt["bad_motion_body_pos_body_names"]:
        assert name in L.BODY_NAMES
    assert "head_link" not in mc.body_names_to_track
    rand = cfg.randomization
    for stage in (rand.setup_terms, rand.reset_terms, rand.step_terms):
        assert "head_scan" in stage
    assert rand.setup_terms["setup_action_delay_buffers"].params["enabled"]
    assert rand.setup_terms["push_randomizer_state"].params["enabled"]
    mass = rand.setup_terms["randomize_object_rigid_body_mass_startup"].params["mass_distribution_params"]
    assert mass == pytest.approx([-0.1 * K.BALL["mass"], 0.1 * K.BALL["mass"]])


def test_undesired_contacts_pattern_spares_only_feet():
    import re

    pat = experiment.lupus_kick_teacher.reward.terms["undesired_contacts"].params["undesired_contacts_body_names"]
    spared = [b for b in L.BODY_NAMES if not re.match(pat, b)]
    assert sorted(spared) == sorted(["left_foot_link", "right_foot_link", "left_foot_contact_point", "right_foot_contact_point"])


# -------------------------------------------------------- MuJoCo integration
@pytest.mark.skipif(not HAVE_MESHES, reason="CAD meshes not present")
def test_teacher_env_runs_in_mujoco_with_ball_and_head_scan(tmp_path):
    pytest.importorskip("mujoco")
    from holosoma.config_values import simulator, terrain
    from holosoma.train_agent import get_tyro_env_config
    from holosoma.utils.helpers import get_class
    from holosoma.utils.sim_utils import setup_simulator_imports

    motion = _write_motion(tmp_path / "kick.npz", n=150, contact=100, ball_x=0.30)
    base = experiment.lupus_kick_teacher
    mc = base.command.setup_terms["motion_command"].params["motion_config"]
    command = dataclasses.replace(
        base.command,
        setup_terms={
            "motion_command": dataclasses.replace(
                base.command.setup_terms["motion_command"],
                params={
                    **base.command.setup_terms["motion_command"].params,
                    "motion_config": dataclasses.replace(
                        mc, motion_file=str(motion), use_adaptive_timesteps_sampler=False, start_at_timestep_zero_prob=1.0
                    ),
                },
            )
        },
    )
    cfg = dataclasses.replace(
        base,
        simulator=simulator.mujoco,
        terrain=terrain.terrain_locomotion_plane,
        command=command,
        # robot material DR is IsaacSim-only; the rest (incl. ball DR, head scan) runs
        randomization=dataclasses.replace(base.randomization, ignore_unsupported=True),
        training=dataclasses.replace(base.training, num_envs=1, headless=True, preprocess_hook=None),
    )
    setup_simulator_imports(cfg)
    env = get_class(cfg.env_class)(get_tyro_env_config(cfg), device="cpu")
    env.reset_all()
    cmd = env.command_manager.get_state("motion_command")
    assert isinstance(cmd, K.LupusKickMotionCommand)
    ball0 = K.ball_state_w(env)[0, :3] - env.simulator.robot_root_states[0, :3]
    assert cmd.kick_phase.item() == 0
    neck = []
    for _ in range(40):
        actions = torch.zeros(1, 22)
        actions[:, -2:] = 5.0  # policy neck outputs must be ignored
        obs, rew, done, _ = env.step({"actions": actions})
        neck.append(env.simulator.dof_pos[0, -2:].clone())
        assert torch.isfinite(rew).all()
    actor = obs["actor_obs"]
    assert torch.isfinite(actor).all()
    # ball is in the scene, at its reference rest pose relative to the robot
    # (+-4 cm ball and +-5 cm root reset noise)
    assert (ball0[:2] - torch.tensor([0.30, 0.0])).abs().max().item() < 0.1
    ball = K.ball_state_w(env)[0]
    assert ball[2].item() == pytest.approx(K.BALL_RADIUS, abs=0.01)
    # neck follows the head-scan generator, not the +5 rad policy output
    neck = torch.stack(neck)
    assert neck.abs().max().item() < 1.1
    gen = env.randomization_manager.get_state("head_scan")
    assert isinstance(gen, H.HeadScanGenerator)
    term = env.action_manager.get_term("joint_control")
    assert torch.allclose(term.passive_targets, (gen.angles - gen.default) / cfg.robot.control.action_scale)
    # a reset after the contact frame parks the ball behind the robot
    cmd._park_ball(torch.tensor([0]))
    rel = K.ball_state_w(env)[0, :3] - env.simulator.robot_root_states[0, :3]
    assert torch.linalg.norm(rel[:2]).item() == pytest.approx(cmd.park_distance, abs=1e-3)
