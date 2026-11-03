"""CPU tests for the LUPUS ball student: obs term, no-depth module, single ONNX export."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from holosoma.agents.modules.module_utils import setup_student_teacher_module
from holosoma.config_types.algo import StudentTeacherModuleConfig
from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.managers.observation.manager import ObservationManager

from wbt_training.config_values import ball_distillation as BD
from wbt_training.utils import exporter
from wbt_training.utils.student_teacher import BallDistillationPPO, RoutingColumnDepthStudentTeacherCritic


def fake_sim(n, ball_xy=(1.5, 0.0), head_pitch=0.4):
    q_head = torch.tensor([0.0, math.sin(head_pitch / 2), 0.0, math.cos(head_pitch / 2)])
    body_list = ["base_link", "neck_link", "head_link"]
    pos = torch.zeros(n, 3, 3)
    pos[:, 2] = torch.tensor([0.0, 0.0, 1.0])
    rot = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(n, 3, 1)
    rot[:, 2] = q_head
    root = torch.zeros(n, 13)
    root[:, 2] = 0.57
    root[:, 6] = 1.0
    ball = torch.zeros(n, 13)
    ball[:, 0], ball[:, 1], ball[:, 2] = ball_xy[0], ball_xy[1], 0.07
    return SimpleNamespace(
        _body_list=body_list,
        _rigid_body_pos=pos,
        _rigid_body_rot=rot,
        robot_root_states=root,
        get_actor_states=lambda names, env_ids: ball[env_ids],
        object_registry=SimpleNamespace(get_names_by_type=lambda _t: ["ball"]),
    )


def test_ball_detection_term_reads_simulator_and_resets():
    n = 3
    env = SimpleNamespace(num_envs=n, device="cpu", dt=0.02, simulator=fake_sim(n))
    params = {
        **BD.ball_detection_term.params,
        "depth_sigma_k": 0.0,
        "bearing_sigma_deg": 0.0,
        "depth_bias_range": [0.0, 0.0],
        "dropout_range": [0.0, 0.0],
        "latency_s_range": [0.0, 0.0],
        "update_hz_range": [50.0, 50.0],
    }
    term = BD.BallDetectionObsTerm(SimpleNamespace(params=params), env)
    out = term(env)
    assert term.object_name == "ball"
    assert torch.allclose(out, torch.tensor([[1.5, 0.0, 1.0, 0.0]] * n), atol=1e-5)
    assert torch.equal(term(env, modify_history=False), out)
    term.reset(torch.tensor([1]))
    assert term.detector.held_xy[1].abs().sum() == 0


def test_no_depth_backbone_gives_zero_width_latent():
    bb = BD.NoDepthBackbone(0)
    assert bb(torch.zeros(5, 1)).shape == (5, 0)
    with pytest.raises(ValueError):
        BD.NoDepthBackbone(32)


def obs_value(env, name):
    return env.values[name]


def student_obs_env(history=3):
    """Real ObservationManager over a fake env with the student's term names."""
    dims = {
        "actions": 22,
        "base_ang_vel": 3,
        "ball_detection": 4,
        "command": 3,
        "dof_pos": 22,
        "dof_vel": 22,
        "robot_anchor_projected_gravity": 3,
    }
    env = SimpleNamespace(num_envs=1, device="cpu", dt=0.02)
    env.values = {k: torch.arange(d, dtype=torch.float32)[None] + 100 * i for i, (k, d) in enumerate(dims.items())}
    terms = {k: ObsTermCfg(func=f"{__name__}:obs_value", params={"name": k}) for k in dims}
    group = ObsGroupCfg(concatenate=True, enable_noise=False, history_length=history, terms=terms)
    env.observation_manager = ObservationManager(ObservationManagerCfg(groups={"actor_obs": group}), env, "cpu")
    return env, dims


def test_student_actor_group_has_the_planned_terms():
    g = BD.lupus_ball_student_actor_obs
    assert set(g.terms) == {
        "actions",
        "base_ang_vel",
        "ball_detection",
        "command",
        "dof_pos",
        "dof_vel",
        "robot_anchor_projected_gravity",
    }
    assert 5 <= g.history_length <= 10
    assert "height_scan" not in g.terms and "depth_cam" not in g.terms


def test_make_ball_distill_observation_adds_routing_and_placeholder():
    teacher = ObsGroupCfg(concatenate=True, enable_noise=True, history_length=1, terms={"object_pos_b": ObsTermCfg(func="x:y")})
    critic = ObsGroupCfg(concatenate=True, enable_noise=True, history_length=1, terms={"object_pos_b": ObsTermCfg(func="x:y")})
    obs = BD.make_ball_distill_observation(teacher, critic)
    assert set(obs.groups) == {"actor_obs", "critic_obs", "teacher_obs", "depth_camera"}
    assert "which_motion" in obs.groups["teacher_obs"].terms
    assert not obs.groups["teacher_obs"].enable_noise and not obs.groups["critic_obs"].enable_noise
    assert obs.groups["depth_camera"].terms["depth_cam"].func.endswith(":no_depth_placeholder")
    p6 = BD.with_phase6_command(obs, kick_dir=True, dribble=True)
    params = p6.groups["actor_obs"].terms["command"].params
    assert params["include_kick_dir"] and params["mode"] == "mode3"
    assert obs.groups["actor_obs"].terms["command"].params["mode"] == "enable"  # original untouched


def test_observation_layout_matches_sorted_history_concat():
    env, dims = student_obs_env(history=3)
    layout = exporter.observation_layout(env)
    assert [e["name"] for e in layout] == sorted(dims)
    assert layout[-1]["stop"] == 3 * sum(dims.values())
    vec = env.observation_manager.compute_group("actor_obs")
    for e in layout:
        # Newest frame is the last `dim` columns of the term's block.
        newest = vec[0, e["stop"] - e["dim"] : e["stop"]]
        assert torch.equal(newest, env.values[e["name"]][0])
    assert exporter.layout_from_metadata(exporter.layout_to_metadata(layout)) == layout


def build_module(actor_dim, teacher_dim=10, critic_dim=12, actions=22):
    cfg = StudentTeacherModuleConfig(
        student_hidden_dims=[64, 32],
        teacher_hidden_dims=[32],
        critic_hidden_dims=[32],
        activation="ELU",
        depth_backbone=BD.NO_DEPTH_BACKBONE,
        depth_output_dim=0,
    )
    return setup_student_teacher_module(
        num_actor_obs=actor_dim,
        num_teacher_obs=teacher_dim,
        num_critic_obs=critic_dim,
        num_actions=actions,
        module_config=cfg,
        device="cpu",
        init_noise_std=0.01,
        module_cls=lambda **kw: RoutingColumnDepthStudentTeacherCritic(routing_column=0, **kw),
    )


def test_student_input_is_exactly_actor_obs():
    module = build_module(actor_dim=50)
    obs = torch.randn(4, 50)
    x = module._student_input(obs, torch.zeros(4, 1))
    assert torch.equal(x, obs)
    assert module.student[0].in_features == 50
    assert list(module.depth_backbone.parameters()) == []


def test_single_onnx_export_numerics_and_metadata(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    env, dims = student_obs_env(history=3)
    env.robot_config = SimpleNamespace(dof_names=[f"j{i}" for i in range(22)], control=SimpleNamespace(stiffness={"j": 10.0}, damping={"j": 1.0}))
    action = SimpleNamespace(action_scales=torch.ones(22))
    env.action_manager = SimpleNamespace(
        get_term=lambda _n: action,
        cfg=SimpleNamespace(terms={"joint_control": SimpleNamespace(params={"passive_dof_names": ["j20", "j21"]})}),
    )
    env.default_dof_pos_base = torch.zeros(22)
    env.command_manager = SimpleNamespace(cfg=SimpleNamespace(setup_terms={"motion_command": None}), get_state=lambda _n: None)
    obs_dim = 3 * sum(dims.values())
    module = build_module(actor_dim=obs_dim)

    path = exporter.export_ball_student_onnx(env, module, str(tmp_path))
    assert path.endswith("policy.onnx") and sorted(p.name for p in tmp_path.iterdir()) == ["policy.onnx"]
    model = onnx.load(path)
    onnx.checker.check_model(model)
    meta = {p.key: p.value for p in model.metadata_props}
    assert meta["policy_format"] == "lupus_ball_student_v1"
    assert int(meta["obs_dim"]) == obs_dim
    assert exporter.layout_from_metadata(meta["obs_layout"]) == exporter.observation_layout(env)
    assert meta["passive_dof_names"] == "j20,j21"

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    assert [(i.name, i.shape) for i in sess.get_inputs()] == [("obs", [1, obs_dim])]
    x = torch.randn(1, obs_dim)
    (y,) = sess.run(None, {"obs": x.numpy()})
    with torch.no_grad():
        ref = module.student(x)
    assert torch.allclose(torch.from_numpy(y), ref, atol=1e-5)


def test_ball_distillation_ppo_rejects_depth_latent():
    algo = BallDistillationPPO.__new__(BallDistillationPPO)
    algo.config = SimpleNamespace(module=SimpleNamespace(depth_output_dim=32))
    with pytest.raises(ValueError):
        algo._build_policy()
