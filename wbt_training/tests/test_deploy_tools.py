"""CPU tests for the ankle linkage, the calibration tools, dribble metrics and tuning hooks."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from wbt_training.deploy import calibrate as C
from wbt_training.deploy.ankle_linkage import AnkleLinkage, AnkleLinkageFirmware
from wbt_training.deploy.policy_io import RobotState
from wbt_training.utils.kick_metrics import KickEpisodeTracker, KickMetricsCfg, summarize

# --------------------------------------------------------------------------- ankle linkage


@pytest.mark.parametrize("model", ["linear", "sine"])
def test_ankle_linkage_round_trip_and_jacobian(model):
    lk = AnkleLinkage(model=model, signs=(1.0, -1.0), offsets=(0.1, -0.2))
    for q in ([0.0, 0.0], [0.3, -0.2], [-0.5, 0.4]):
        q = np.array(q)
        m = lk.joint_to_motor(q)
        np.testing.assert_allclose(lk.motor_to_joint(m), q, atol=1e-9)
        eps = 1e-6
        num = np.stack([(lk.joint_to_motor(q + eps * e) - lk.joint_to_motor(q - eps * e)) / (2 * eps) for e in np.eye(2)], 1)
        np.testing.assert_allclose(lk.jacobian(q), num, atol=1e-6)
        qd = np.array([0.7, -1.1])
        np.testing.assert_allclose(lk.motor_vel_to_joint(q, lk.joint_vel_to_motor(q, qd)), qd, atol=1e-9)


def test_ankle_linkage_matches_simulation_ratios():
    """Linear model = constant ratios of hardware.yaml (tau_joint = 2 tau_motor ratio)."""
    lk = AnkleLinkage.from_hardware()
    r, d, w = lk.crank_radius_r, lk.pitch_lever_d, lk.roll_lever_w
    q = np.zeros(2)
    # equal motor torques -> pure pitch torque 2 tau d/r; opposite -> roll 2 tau w/r
    np.testing.assert_allclose(lk.motor_torque_to_joint(q, [1.0, 1.0]), [2 * d / r, 0.0], atol=1e-12)
    np.testing.assert_allclose(lk.motor_torque_to_joint(q, [1.0, -1.0]), [0.0, 2 * w / r], atol=1e-12)
    # power is conserved
    qd, tm = np.array([0.4, -0.3]), np.array([2.0, 5.0])
    assert np.dot(lk.joint_vel_to_motor(q, qd), tm) == pytest.approx(np.dot(qd, lk.motor_torque_to_joint(q, tm)))
    # exact diagonal gains when k_pitch / d^2 == k_roll / w^2
    k = np.array([100.0, 100.0 * w * w / (d * d)])
    assert lk.gain_coupling(q, k) == pytest.approx(0.0, abs=1e-12)
    assert lk.gain_coupling(q, [100.0, 100.0]) > 0.0


def test_ankle_firmware_wrapper_converts_both_ways():
    names = ["left_ankle_pitch", "left_ankle_roll", "right_ankle_pitch", "right_ankle_roll", "left_knee"]
    lk = AnkleLinkage()
    q_joint = np.array([0.2, -0.1, -0.3, 0.15, 0.5])

    class Motors:
        sent = None

        def read_state(self):
            q = q_joint.copy()
            q[0:2] = lk.joint_to_motor(q_joint[0:2])
            q[2:4] = lk.joint_to_motor(q_joint[2:4])
            qd = np.zeros(5)
            qd[0:2] = lk.joint_vel_to_motor(q_joint[0:2], [1.0, 0.0])
            return RobotState(q, qd, np.array([1.0, 0, 0, 0]), np.zeros(3))

        def send_targets(self, q, kp, kd):
            self.sent = (q, kp, kd)

    inner = Motors()
    fw = AnkleLinkageFirmware(inner, names, {"left": lk, "right": lk})
    st = fw.read_state()
    np.testing.assert_allclose(st.joint_pos, q_joint, atol=1e-9)
    np.testing.assert_allclose(st.joint_vel[0:2], [1.0, 0.0], atol=1e-9)
    kp = np.full(5, 100.0)
    fw.send_targets(q_joint, kp, np.full(5, 2.0))
    q, kpm, _ = inner.sent
    np.testing.assert_allclose(q[0:2], lk.joint_to_motor(q_joint[0:2]))
    assert q[4] == q_joint[4] and kpm[4] == 100.0
    np.testing.assert_allclose(kpm[0:2], lk.motor_gains(q_joint[0:2], [100.0, 100.0]))


# --------------------------------------------------------------------------- calibration


def test_ball_track_fit_recovers_damping():
    c, v0 = 0.45, 3.0
    t = np.linspace(0, 6, 121)
    s = v0 / c * (1 - np.exp(-c * t))
    fit = C.fit_ball_track(t, s * math.cos(0.3), s * math.sin(0.3))
    assert fit["c"] == pytest.approx(c, rel=0.02)
    assert fit["v0"] == pytest.approx(v0, rel=0.03)
    assert fit["rms_exp"] < fit["rms_const_decel"]
    sug = C.ball_suggestion([fit["c"]])
    assert sug["robots/ball/ball.yaml"]["linear_damping"]["value"] == pytest.approx(c, rel=0.02)


def test_ball_rolls_fit():
    v0 = np.array([1.0, 2.0, 3.0])
    fit = C.fit_ball_rolls(v0, v0 / 0.5)
    assert fit["c"] == pytest.approx(0.5) and fit["n"] == 3


def test_damp_ball_matches_effective_rolling_decay():
    """MuJoCo ball with linear + angular damping c rolls like v' = -c v (as in IsaacSim)."""
    mujoco = pytest.importorskip("mujoco")
    from wbt_training.sim2sim.lupus_kick_sim2sim import damp_ball

    xml = """<mujoco><option timestep="0.005"/><worldbody><geom type="plane" size="10 10 0.1"/>
    <body name="ball" pos="0 0 0.075"><freejoint/>
    <inertial pos="0 0 0" mass="0.2" diaginertia="0.00075 0.00075 0.00075"/>
    <geom type="sphere" size="0.075" friction="0.8 0.005 0.0001"/></body></worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    r, v0, c = 0.075, 2.0, 0.4
    d.qvel[0], d.qvel[4] = v0, v0 / r  # rolling without slip about +y
    for _ in range(int(2.0 / m.opt.timestep)):
        damp_ball(d.qvel[0:6], c, c, m.opt.timestep)
        mujoco.mj_step(m, d)
    assert d.qvel[0] == pytest.approx(v0 * math.exp(-c * 2.0), rel=0.05)


def test_vision_fit_recovers_noise_and_ground_projection(tmp_path):
    rng = np.random.default_rng(0)
    n = 600
    dist = rng.uniform(0.3, 5.0, n)
    az = rng.uniform(-0.6, 0.6, n)
    tx, ty = dist * np.cos(az), dist * np.sin(az)
    k, bearing = 0.06, math.radians(1.5)
    dd = dist * (1 + 0.02 + k * rng.standard_normal(n))
    da = az + bearing * rng.standard_normal(n)
    det_x, det_y = dd * np.cos(da), dd * np.sin(da)
    miss = dist > 4.2
    det_x[miss] = np.nan
    det_y[miss] = np.nan
    # ground projection: error grows with d^2, better than depth up to ~1.2 m
    ge = 0.05 * dist**2
    gp_x, gp_y = tx + ge * rng.standard_normal(n) / math.sqrt(2), ty + ge * rng.standard_normal(n) / math.sqrt(2)
    path = tmp_path / "vision.csv"
    with open(path, "w") as fh:
        fh.write("true_x,true_y,det_x,det_y,gp_x,gp_y\n")
        for row in zip(tx, ty, det_x, det_y, gp_x, gp_y):
            fh.write(",".join("" if np.isnan(v) else f"{v:.6f}" for v in row) + "\n")
    fit = C.fit_vision(C.read_csv(path), bins=8)
    assert fit["depth_sigma_k"] == pytest.approx(k, rel=0.15)
    assert fit["depth_bias_mean"] == pytest.approx(0.02, abs=0.01)
    assert fit["bearing_sigma_deg"] == pytest.approx(1.5, rel=0.15)
    assert 3.5 < fit["max_range"] < 4.8
    assert 0.5 < fit["near_ground_projection_range"] < 2.5
    sug = C.vision_suggestion(fit)
    assert "depth_sigma_k" in next(iter(sug.values()))


def test_latency_suggestion_and_cli(tmp_path, capsys):
    stats = {
        "loop_period_s": {"n": 100, "mean": 0.02, "p50": 0.02, "p99": 0.021, "max": 0.022},
        "compute_s": {"n": 100, "mean": 0.002, "p50": 0.002, "p99": 0.004, "max": 0.005},
        "vision_latency_s": {"n": 80, "mean": 0.06, "p50": 0.055, "p99": 0.09, "max": 0.1},
        "overruns": 0,
    }
    p = tmp_path / "rec.json"
    p.write_text(json.dumps({"latency": stats}))
    assert C.main(["latency", str(p), "--actuation-latency-ms", "15"]) == 0
    out = json.loads(capsys.readouterr().out)
    lo, hi = out["suggested"]["BallVisionCfg.latency_s_range"]
    assert lo < 0.055 and hi > 0.09
    assert out["suggested"]["lupus_kick.py setup_action_delay_buffers.ctrl_delay_step_range"] == [0, 1]


# --------------------------------------------------------------------------- dribble metrics


def _step(n, ball_x, mode, contact=False, ball_vx=0.0, cmd_v=0.4):
    return {
        "ball_pos_w": torch.tensor([[ball_x, 0.0, 0.07]]).repeat(n, 1),
        "ball_vel_w": torch.tensor([[ball_vx, 0.0, 0.0]]).repeat(n, 1),
        "base_pos_w": torch.tensor([[0.0, 0.0, 0.57]]).repeat(n, 1),
        "base_yaw": torch.zeros(n),
        "base_up_z": torch.ones(n),
        "base_fwd_vel": torch.full((n,), cmd_v),
        "base_yaw_rate": torch.zeros(n),
        "cmd_v": torch.full((n,), cmd_v),
        "cmd_w": torch.zeros(n),
        "kick_phase": torch.zeros(n),
        "contact": torch.full((n,), contact),
        "target_heading": torch.zeros(n),
        "kick_mode": torch.full((n,), mode),
        "feet_pos_w": torch.tensor([[[0.0, 0.133, 0.03], [0.0, -0.133, 0.03]]]).repeat(n, 1, 1),
    }


def test_dribble_metrics_and_contacts_are_not_miskicks():
    tr = KickEpisodeTracker(1, KickMetricsCfg(dribble_loss_dist=0.6))
    for x in (0.2, 0.25, 0.3):
        tr.step(_step(1, x, 2, contact=True, ball_vx=0.4))
    tr.finish([0])
    for x in (0.2, 0.9):  # ball gets away
        tr.step(_step(1, x, 2, ball_vx=1.0))
    tr.finish([0])
    tr.step(_step(1, 0.2, 0, contact=True))  # ignore mode: contact is a miskick
    tr.finish([0])
    a, b, c = tr.records
    assert a["dribble"] and not a["miskick"] and not a["dribble_lost"]
    assert a["dribble_dist"] == pytest.approx(np.mean([math.hypot(x, 0.133) for x in (0.2, 0.25, 0.3)]), rel=1e-5)
    assert a["dribble_speed_err"] == pytest.approx(0.0, abs=1e-6)
    assert b["dribble_lost"] and b["dribble_speed_err"] == pytest.approx(0.6, rel=1e-5)
    assert not c["dribble"] and c["miskick"]
    s = summarize(tr.records)
    assert s["dribble_episodes"] == 2 and s["dribble_loss_rate"] == pytest.approx(0.5)


def test_metrics_without_optional_keys_unchanged():
    tr = KickEpisodeTracker(1)
    st = _step(1, 1.0, 0)
    del st["kick_mode"], st["feet_pos_w"]
    tr.step(st)
    tr.finish([0])
    assert not tr.records[0]["dribble"]
    assert math.isnan(summarize(tr.records)["dribble_ball_dist_mean"])


# --------------------------------------------------------------------------- dribble tuning hooks


def test_dribble_reward_cfg_and_sweep():
    from wbt_training.config_values import lupus_phase6 as P6
    from wbt_training.training_runs.dribble_sweep import sweep_commands

    terms = P6.DribbleRewardCfg(near_foot_weight=1.5, near_foot_sigma=0.2).terms()
    assert terms["ball_near_foot"].weight == 1.5 and terms["ball_near_foot"].params == {"sigma": 0.2}
    cmds = sweep_commands("file:///b", [0.25, 1.0], [0.5], iterations=5000)
    assert len(cmds) == 2
    assert "--reward.terms.ball-near-foot.weight 0.25" in cmds[0]
    assert "run_lupus_kick_teacher.sh" in cmds[0] and "EXP_PRESET=exp:lupus-dribble-teacher" in cmds[0]


def test_dribble_weights_overridable_on_cli():
    tyro = pytest.importorskip("tyro")
    from holosoma.utils.tyro_utils import TYRO_CONIFG

    from wbt_training.config_values import experiment

    cfg = tyro.cli(
        experiment.AnnotatedExperimentConfig,
        args=["exp:lupus-dribble-teacher", "--reward.terms.ball-near-foot.weight", "1.25"],
        config=TYRO_CONIFG,
    )
    assert cfg.reward.terms["ball_near_foot"].weight == 1.25


# --------------------------------------------------------------------------- soccer fields through convert/combine


def _raw_soccer_motion(path, T, side):
    np.savez(
        path,
        fps=np.array(50),
        joint_pos=np.zeros((T, 22)),
        joint_vel=np.zeros((T, 22)),
        body_pos_w=np.zeros((T, 25, 3)),
        body_quat_w=np.tile([1.0, 0, 0, 0], (T, 25, 1)),
        body_lin_vel_w=np.zeros((T, 25, 3)),
        body_ang_vel_w=np.zeros((T, 25, 3)),
        joint_names=np.array([f"j{i}" for i in range(22)]),
        body_names=np.array([f"b{i}" for i in range(25)]),
        vel_cmd=np.zeros((T, 2), np.float32),
        object_pos_w=np.ones((T, 3), np.float32),
        kick_enable=np.ones(T, np.float32),
        kick_phase=np.zeros(T, np.int8),
        contact_frame=np.zeros(T, bool),
        kick_side=np.array(side),
        scenario=np.array("walk_kick"),
    )


def test_motion_convert_keeps_kick_fields_and_combine_takes_scalars(tmp_path, monkeypatch):
    from wbt_training.training_runs.motion_convert import convert
    from wbt_training.utils import multi_motion_helpers as M

    outs = []
    for i, (T, side) in enumerate(((4, 0), (3, 1))):
        raw = tmp_path / f"raw{i}.npz"
        _raw_soccer_motion(raw, T, side)
        out = tmp_path / f"m{i}_motion.npz"
        assert convert(str(raw), str(out))
        with np.load(out) as d:
            assert d["object_pos_w"].shape == (T, 3) and d["kick_enable"].shape == (T,)
            assert d["kick_side"].tolist() == [side] * T and "scenario" not in d.files
        outs.append(out)
    monkeypatch.setattr(M, "_scratch_dir", lambda: tmp_path)
    with np.load(M.combine_motions(outs)) as c:
        assert c["kick_side"].tolist() == [0] * 4 + [1] * 3
        assert c["kick_phase"].shape == (7,) and c["joint_pos"].shape == (7, 29)
    # raw files with 0-d fields combine as well (numeric -> per frame, strings dropped)
    raws = [tmp_path / "raw0.npz", tmp_path / "raw1.npz"]
    with np.load(M.combine_motions(raws)) as c:
        assert c["kick_side"].tolist() == [0] * 4 + [1] * 3 and "scenario" not in c.files
