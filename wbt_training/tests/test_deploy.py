"""CPU tests for the LUPUS deploy runtime and the MuJoCo kick Sim2Sim (plan, Phase 5)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.managers.observation.manager import ObservationManager

from wbt_training.config_values import lupus as L
from wbt_training.config_values.ball_distillation import STUDENT_HISTORY, ball_detection_term, student_command_term
from wbt_training.deploy.policy_io import (
    BallInput,
    ObsBuilder,
    PolicyMetadata,
    RobotState,
    StudentPolicy,
    command_vector,
    parse_list,
    projected_gravity_wxyz,
)
from wbt_training.deploy.runner import BehaviorCommand, PolicyNode
from wbt_training.utils import exporter
from wbt_training.utils.ball_detection import BallDetector, BallVisionCfg

STUDENT_DIMS = {
    "actions": 22,
    "ball_detection": 4,
    "base_ang_vel": 3,
    "command": 3,
    "dof_pos": 22,
    "dof_vel": 22,
    "robot_anchor_projected_gravity": 3,
}
LAYOUT = exporter.layout_from_metadata(";".join(f"{k}:{d}:{STUDENT_HISTORY}" for k, d in sorted(STUDENT_DIMS.items())))
OBS_DIM = LAYOUT[-1]["stop"]


def test_parse_list_reads_both_metadata_formats():
    assert parse_list("[1.0, 2.5]") == [1.0, 2.5]
    assert parse_list("0.100,-0.200,a_b") == [0.1, -0.2, "a_b"]
    assert parse_list("") == []


def obs_value(env, name):
    return env.values[name]


def test_obs_builder_matches_holosoma_history_and_reset():
    dims = {"b_term": 2, "a_term": 3}
    env = SimpleNamespace(num_envs=1, device="cpu")
    terms = {k: ObsTermCfg(func=f"{__name__}:obs_value", params={"name": k}) for k in dims}
    cfg = ObservationManagerCfg(groups={"actor_obs": ObsGroupCfg(concatenate=True, enable_noise=False, history_length=3, terms=terms)})
    env.observation_manager = mgr = ObservationManager(cfg, env, "cpu")
    builder = ObsBuilder(exporter.observation_layout(env) if False else exporter.layout_from_metadata("a_term:3:3;b_term:2:3"))
    rng = np.random.default_rng(0)
    for step in range(6):
        if step == 4:
            mgr.reset(torch.tensor([0]))
            builder.reset()
        env.values = {k: torch.tensor(rng.normal(size=(1, d)), dtype=torch.float32) for k, d in dims.items()}
        ref = mgr.compute_group("actor_obs")[0].numpy()
        got = builder.build({k: v[0].numpy() for k, v in env.values.items()})[0]
        np.testing.assert_allclose(got, ref, atol=1e-6)


def test_obs_builder_rejects_wrong_width():
    b = ObsBuilder(exporter.layout_from_metadata("x:2:1"))
    with pytest.raises(ValueError):
        b.build({"x": np.zeros(3)})


def test_ball_input_matches_training_detector_semantics():
    cfg = BallVisionCfg(
        depth_sigma_k=0.0,
        bearing_sigma_deg=0.0,
        depth_bias_range=(0.0, 0.0),
        dropout_range=(0.0, 0.0),
        latency_s_range=(0.04, 0.04),
        update_hz_range=(50.0, 50.0),
    )
    det = BallDetector(1, cfg, dt=0.02)
    det._phase[:] = 0.999  # frame at every step
    inp = BallInput(cfg.time_since_seen_cap)
    head = torch.tensor([[0.0, 0.0, 1.0]])
    q_head = torch.tensor([[0.0, math.sin(0.2), 0.0, math.cos(0.2)]])
    base = torch.tensor([[0.0, 0.0, 0.57]])
    q = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    history = []
    for k in range(12):
        x = 2.0 + 0.1 * k if k < 8 else -2.0  # ball leaves the view at k = 8
        ball = torch.tensor([[x, 0.0, 0.07]])
        out = det.step(ball, head, q_head, base, q)[0].numpy()
        t = 0.02 * k
        history.append((t, x > 0, x))
        cap_t, seen, cx = history[max(0, k - 2)]  # 40 ms latency = 2 steps
        inp.on_frame(cap_t, seen, cx, 0.0)
        got = inp.observe(t)
        if k >= 2:
            np.testing.assert_allclose(got, out, atol=1e-5)


def test_command_vector_layouts():
    assert command_vector(0.5, 0.1, True).tolist() == pytest.approx([0.5, 0.1, 1.0])
    six = command_vector(0.5, 0.1, kick_dir=0.2, mode=2, cfg={"include_kick_dir": True, "mode": "mode3"})
    assert six.tolist() == pytest.approx([0.5, 0.1, 0.2, 0.0, 0.0, 1.0])


def make_student_onnx(path: Path, bias: np.ndarray | None = None) -> Path:
    """ONNX with the real student layout and LUPUS metadata; constant output = ``bias``."""
    net = torch.nn.Sequential(torch.nn.Linear(OBS_DIM, 8), torch.nn.ELU(), torch.nn.Linear(8, 22))
    with torch.no_grad():
        net[2].weight.zero_()
        net[2].bias.copy_(torch.as_tensor(bias if bias is not None else np.zeros(22)))
    torch.onnx.export(net, (torch.zeros(1, OBS_DIM),), str(path), input_names=["obs"], output_names=["actions"], opset_version=13, dynamo=False)
    kp = [next(v for k, v in L.STIFFNESS.items() if k in n) for n in L.DOF_NAMES]
    kd = [next(v for k, v in L.DAMPING.items() if k in n) for n in L.DOF_NAMES]
    exporter.attach_onnx_metadata(
        str(path),
        {
            "joint_names": L.DOF_NAMES,
            "default_joint_pos": [L.DEFAULT_JOINT_ANGLES[n] for n in L.DOF_NAMES],
            "action_scale": [1.0] * 22,
            "kp": kp,
            "kd": kd,
            "obs_layout": exporter.layout_to_metadata(LAYOUT),
            "obs_dim": OBS_DIM,
            "passive_dof_names": L.NECK_DOF_NAMES,
            "policy_dt": 0.02,
            "ball_vision_cfg": json.dumps(ball_detection_term.params),
            "command_obs_cfg": json.dumps(student_command_term.params),
            "policy_format": "lupus_ball_student_v1",
        },
    )
    return path


def test_metadata_requires_ball_student_format():
    with pytest.raises(ValueError):
        PolicyMetadata.from_dict({"policy_format": "something_else"})


def state():
    return RobotState(
        joint_pos=np.array([L.DEFAULT_JOINT_ANGLES[n] for n in L.DOF_NAMES]),
        joint_vel=np.zeros(22),
        base_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        base_ang_vel=np.zeros(3),
    )


def test_student_policy_targets_and_passive_neck(tmp_path):
    pytest.importorskip("onnxruntime")
    bias = np.full(22, 0.1)
    pol = StudentPolicy(make_student_onnx(tmp_path / "p.onnx", bias))
    assert pol.meta.obs_dim == OBS_DIM and pol.meta.passive_dof_names == L.NECK_DOF_NAMES
    q = pol.step(state(), command_vector(0.5, 0.0, True), np.array([1.0, 0.0, 1.0, 0.0], np.float32))
    default = pol.meta.default_joint_pos
    np.testing.assert_allclose(q[:20], default[:20] + 0.1, atol=1e-6)
    np.testing.assert_allclose(q[20:], default[20:])  # neck not driven by the policy
    np.testing.assert_allclose(pol.last_action, bias, atol=1e-6)  # raw action observed next step
    terms = pol.terms(state(), command_vector(0, 0), np.zeros(4))
    np.testing.assert_allclose(terms["robot_anchor_projected_gravity"], [0, 0, -1])
    np.testing.assert_allclose(projected_gravity_wxyz([math.cos(0.25), 0, math.sin(0.25), 0])[0], math.sin(0.5), atol=1e-9)


class FakeRobot:
    def __init__(self):
        self.sent = []
        self.cmd = BehaviorCommand(v=0.3, kick_enable=False, neck=(0.5, 0.2))

    def read_state(self):
        return state()

    def send_targets(self, q, kp, kd):
        self.sent.append((q.copy(), kp, kd))

    def command(self):
        return self.cmd


def test_policy_node_fixed_rate_neck_and_latency(tmp_path):
    pytest.importorskip("onnxruntime")
    pol = StudentPolicy(make_student_onnx(tmp_path / "p.onnx"))
    now = [0.0]
    slow = {"compute": 0.005}

    def clock():
        return now[0]

    def sleep(s):
        now[0] += s

    robot = FakeRobot()
    node = PolicyNode(pol, robot, robot, clock=clock)
    orig_tick = node.tick

    def tick():
        q = orig_tick()
        now[0] += slow["compute"]
        return q

    node.tick = tick
    node.on_vision_frame(capture_time=-0.06, found=True, x=1.0, y=0.1)
    stats = node.run(seconds=1.0, sleep=sleep)
    assert len(robot.sent) == 50 and stats.overruns == 0
    assert stats.loop_period.stats()["mean"] == pytest.approx(0.02, abs=1e-9)
    assert stats.vision_latency.stats()["mean"] == pytest.approx(0.06)
    q, kp, _ = robot.sent[-1]
    assert q[20:].tolist() == [0.5, 0.2]  # behavior's neck targets
    np.testing.assert_allclose(kp, pol.meta.kp)
    slow["compute"] = 0.03  # slower than the period -> overruns, no burst
    node.run(seconds=0.2, sleep=sleep)
    assert node.stats.overruns > 0


# --- MuJoCo kick Sim2Sim ----------------------------------------------------------

LUPUS_DIR = Path(__file__).resolve().parents[2] / "robots" / "lupus"
HAVE_MESHES = (LUPUS_DIR / "meshes").exists()
mujoco = pytest.importorskip("mujoco") if HAVE_MESHES else None
needs_model = pytest.mark.skipif(mujoco is None, reason="CAD meshes or mujoco not present")


@needs_model
def test_kick_sim2sim_stand_still_without_kick(tmp_path):
    pytest.importorskip("onnxruntime")
    from wbt_training.sim2sim import lupus_kick_sim2sim as K

    sim = K.LupusKickSim2Sim(make_student_onnx(tmp_path / "p.onnx"))
    assert sim.decimation == 4
    sc = K.KickScenario(approach=1.0, lateral=0.0, v=0.0, kick_enable=False, seconds=2.0, head_scan=False)
    rec = sim.run(sc)
    assert rec["steps"] == 100 and not rec["fell"] and not rec["miskick"] and not rec["intended"]
    ball = sim.data.qpos[sim.ball_q : sim.ball_q + 3]
    assert abs(ball[0] - 1.0) < 0.02 and abs(ball[2] - sim.ball_radius) < 0.01
    # The camera saw the ball (head pitched down 0.6 rad) and frames arrived late.
    assert sim.node.ball.last_capture is not None
    lat = sim.node.stats.vision_latency.stats()
    assert 0.04 - 1e-6 <= lat["mean"] <= 0.1 + 0.02
    obs_ball = sim.node.ball.observe(sim.t)
    assert obs_ball[0] == pytest.approx(1.0, abs=0.2) and obs_ball[2] == 1.0


@needs_model
def test_kick_sim2sim_ball_pushed_into_robot_counts_contact(tmp_path):
    pytest.importorskip("onnxruntime")
    from wbt_training.sim2sim import lupus_kick_sim2sim as K

    sim = K.LupusKickSim2Sim(make_student_onnx(tmp_path / "p.onnx"))
    sc = K.KickScenario(approach=0.5, lateral=K.FOOT_OFFSET_Y, v=0.0, kick_enable=False, seconds=1.0, head_scan=True, jitter={})
    sim.reset(sc)
    # Roll the ball at the standing robot: it is not a kick, so it is a miskick.
    orig_reset = sim.reset

    def reset_with_roll(s, base_height=0.573):
        orig_reset(s, base_height)
        sim.data.qvel[sim.ball_v] = -2.0

    sim.reset = reset_with_roll
    rec = sim.run(sc)
    assert rec["miskick"] and rec["lateral"] == pytest.approx(0.0, abs=0.01)
    assert sim.data.qpos[sim.ball_q] > -0.2  # stopped by the foot, did not pass through


@needs_model
def test_kick_sim2sim_cli(tmp_path, capsys):
    pytest.importorskip("onnxruntime")
    from wbt_training.sim2sim import lupus_kick_sim2sim as K

    onnx = make_student_onnx(tmp_path / "p.onnx")
    out = tmp_path / "rep.json"
    assert K.main(["--onnx", str(onnx), "--stand-kick", "--approach", "0.25", "--seconds", "1.0", "--out", str(out)]) == 0
    rec = json.loads(out.read_text())
    assert rec["steps"] == 50 and "latency" in rec
