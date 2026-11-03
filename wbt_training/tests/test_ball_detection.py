"""CPU tests for the ball-detection noise and timing model (plan, Phase 4)."""

from __future__ import annotations

import math

import pytest
import torch

from wbt_training.utils.ball_detection import (
    BallDetector,
    BallVisionCfg,
    camera_pose_w,
    in_field_of_view,
    noisy_camera_measurement,
    world_to_robot_xy,
)

IDENTITY = torch.tensor([0.0, 0.0, 0.0, 1.0])


def quat_yaw(yaw: float) -> torch.Tensor:
    return torch.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])


def quat_pitch(pitch: float) -> torch.Tensor:
    return torch.tensor([0.0, math.sin(pitch / 2), 0.0, math.cos(pitch / 2)])


def noiseless(**kw) -> BallVisionCfg:
    base = dict(
        depth_sigma_k=0.0,
        depth_bias_range=(0.0, 0.0),
        bearing_sigma_deg=0.0,
        dropout_range=(0.0, 0.0),
        latency_s_range=(0.0, 0.0),
        update_hz_range=(50.0, 50.0),
    )
    base.update(kw)
    return BallVisionCfg(**base)


def p_cam_for(ball_w, head_pos, head_quat, cfg):
    cam_pos, cam_rot = camera_pose_w(head_pos, head_quat, cfg)
    return (cam_rot.transpose(-1, -2) @ (ball_w - cam_pos).unsqueeze(-1)).squeeze(-1)


def test_cfg_round_trips_through_obs_params():
    cfg = BallVisionCfg(depth_sigma_k=0.1, update_hz_range=(20.0, 30.0))
    params = cfg.as_params()
    assert isinstance(params["update_hz_range"], list)
    assert BallVisionCfg.from_params({**params, "object_name": "ball"}) == cfg


def test_fov_front_visible_behind_and_far_not():
    cfg = BallVisionCfg()
    head = torch.tensor([[0.0, 0.0, 1.0]]).repeat(4, 1)
    quat = IDENTITY.repeat(4, 1)
    balls = torch.tensor([[3.0, 0.0, 0.5], [-2.0, 0.0, 0.07], [9.0, 0.0, 0.07], [2.0, 2.5, 0.9]])
    vis = in_field_of_view(p_cam_for(balls, head, quat, cfg), cfg)
    assert vis.tolist() == [True, False, False, False]


def test_ball_at_feet_needs_head_pitched_down():
    # LUPUS head ~1 m high: a ball 0.4 m ahead on the ground is ~65 deg below
    # the horizon, outside a 65 deg vertical FOV unless the head looks down.
    cfg = BallVisionCfg()
    head = torch.tensor([[0.0, 0.0, 1.0]])
    ball = torch.tensor([[0.4, 0.0, 0.07]])
    assert not in_field_of_view(p_cam_for(ball, head, IDENTITY[None], cfg), cfg).item()
    down = quat_pitch(0.79)[None]  # head_pitch upper limit, positive = down
    assert in_field_of_view(p_cam_for(ball, head, down, cfg), cfg).item()


def test_head_yaw_moves_fov():
    cfg = BallVisionCfg()
    head = torch.tensor([[0.0, 0.0, 1.0]])
    ball = torch.tensor([[0.0, 3.0, 0.07]])  # 90 deg to the left
    assert not in_field_of_view(p_cam_for(ball, head, IDENTITY[None], cfg), cfg).item()
    assert in_field_of_view(p_cam_for(ball, head, quat_yaw(1.0)[None], cfg), cfg).item()


def test_depth_noise_grows_with_distance_and_bearing_stays_tight():
    cfg = BallVisionCfg(depth_sigma_k=0.05, bearing_sigma_deg=1.0, depth_bias_range=(0.0, 0.0))
    g = torch.Generator().manual_seed(0)
    n = 20000
    cam_pos = torch.zeros(n, 3)
    cam_rot = torch.eye(3).repeat(n, 1, 1)
    for d in (1.0, 4.0):
        p = torch.tensor([[d, 0.0, 0.0]]).repeat(n, 1)
        m = noisy_camera_measurement(p, cam_pos, cam_rot, torch.zeros(n), cfg, g)
        rng = m.norm(dim=-1)
        assert rng.std().item() == pytest.approx(0.05 * d, rel=0.05)
        bearing = torch.atan2(m[:, 1], m[:, 0])
        assert bearing.std().item() == pytest.approx(math.radians(1.0), rel=0.05)


def test_depth_bias_scales_range():
    cfg = noiseless()
    p = torch.tensor([[2.0, 0.0, 0.0]])
    m = noisy_camera_measurement(p, torch.zeros(1, 3), torch.eye(3)[None], torch.tensor([0.05]), cfg)
    assert m[0, 0].item() == pytest.approx(2.1, abs=1e-5)


def test_near_ground_projection_is_exact_without_angle_noise():
    cfg = noiseless(depth_sigma_k=0.3, near_ground_projection_range=1.5, ground_projection_pitch_sigma_deg=0.0)
    g = torch.Generator().manual_seed(1)
    cam_pos = torch.tensor([[0.0, 0.0, 1.0]])
    ball_w = torch.tensor([[0.8, 0.1, cfg.ball_radius]])
    m = noisy_camera_measurement(ball_w - cam_pos, cam_pos, torch.eye(3)[None], torch.zeros(1), cfg, g)
    assert torch.allclose(m, ball_w, atol=1e-5)  # the 30 % depth noise is bypassed


def test_world_to_robot_heading_frame_ignores_tilt():
    ball = torch.tensor([[1.0, 1.0, 0.07]])
    base = torch.zeros(1, 3)
    xy = world_to_robot_xy(ball, base, quat_yaw(math.pi / 2)[None], "heading")
    assert torch.allclose(xy, torch.tensor([[1.0, -1.0]]), atol=1e-6)
    # A pitched base does not change the heading-frame position.
    tilted = world_to_robot_xy(ball, base, quat_pitch(0.3)[None], "heading")
    assert torch.allclose(tilted, torch.tensor([[1.0, 1.0]]), atol=1e-6)
    with pytest.raises(ValueError):
        world_to_robot_xy(ball, base, IDENTITY[None], "camera")


def scene(n, ball_x=2.0):
    return dict(
        ball_pos_w=torch.tensor([[ball_x, 0.0, 0.07]]).repeat(n, 1),
        head_pos_w=torch.tensor([[0.0, 0.0, 1.0]]).repeat(n, 1),
        head_quat_w=quat_pitch(0.3).repeat(n, 1),
        base_pos_w=torch.tensor([[0.0, 0.0, 0.57]]).repeat(n, 1),
        base_quat_w=IDENTITY.repeat(n, 1),
    )


def test_noiseless_detector_reports_truth_immediately():
    det = BallDetector(2, noiseless(), dt=0.02)
    out = det.step(**scene(2))
    assert torch.allclose(out, torch.tensor([[2.0, 0.0, 1.0, 0.0]] * 2), atol=1e-5)


def test_ball_out_of_view_holds_last_value_and_ages():
    det = BallDetector(1, noiseless(), dt=0.02)
    det.step(**scene(1, 2.0))
    s = scene(1, 2.0)
    s["ball_pos_w"] = torch.tensor([[-2.0, 0.0, 0.07]])  # behind the robot
    for _ in range(10):
        out = det.step(**s)
    assert out[0, :2].tolist() == pytest.approx([2.0, 0.0], abs=1e-5)
    assert out[0, 2].item() == 0.0
    assert out[0, 3].item() == pytest.approx(0.2, abs=1e-5)
    for _ in range(200):
        out = det.step(**s)
    assert out[0, 3].item() == pytest.approx(det.cfg.time_since_seen_cap)


def test_never_seen_starts_at_cap():
    det = BallDetector(1, noiseless(), dt=0.02)
    s = scene(1, -2.0)
    out = det.step(**s)
    assert out[0].tolist() == pytest.approx([0.0, 0.0, 0.0, det.cfg.time_since_seen_cap])


def test_update_rate_and_dropout_statistics():
    cfg = noiseless(update_hz_range=(30.0, 30.0), dropout_range=(0.2, 0.2))
    g = torch.Generator().manual_seed(2)
    n, steps = 512, 250  # 5 s at 50 Hz
    det = BallDetector(n, cfg, dt=0.02, generator=g)
    s = scene(n)
    updates = torch.zeros(n)
    for t in range(steps):
        # Move the ball each step so every new detection changes the value.
        s["ball_pos_w"] = torch.tensor([[2.0 + 0.001 * t, 0.0, 0.07]]).repeat(n, 1)
        prev = det.held_xy[:, 0].clone()
        det.step(**s)
        updates += (det.held_xy[:, 0] != prev).float()
    rate = updates.mean().item() / (steps * 0.02)
    assert rate == pytest.approx(30.0 * 0.8, rel=0.05)


def test_latency_delays_the_reported_position():
    cfg = noiseless(latency_s_range=(0.06, 0.06))  # 3 policy steps
    det = BallDetector(1, cfg, dt=0.02)
    s = scene(1)
    xs = []
    for t in range(10):
        s["ball_pos_w"] = torch.tensor([[2.0 + 0.1 * t, 0.0, 0.07]])
        xs.append(det.step(**s)[0, 0].item())
    # After the buffer has filled, the report lags the truth by 3 steps.
    assert xs[6] == pytest.approx(2.0 + 0.1 * 3, abs=1e-5)
    assert det.age[0].item() == pytest.approx(0.06, abs=1e-6)


def test_reset_rerandomises_and_clears_state():
    cfg = BallVisionCfg()
    g = torch.Generator().manual_seed(3)
    det = BallDetector(64, cfg, dt=0.02, generator=g)
    for _ in range(5):
        det.step(**scene(64))
    rates = det.rate_hz.clone()
    det.reset(torch.tensor([0, 1]))
    assert det.held_xy[:2].abs().sum() == 0
    assert (det.age[:2] == cfg.time_since_seen_cap).all()
    assert torch.equal(det.rate_hz[2:], rates[2:])
    lo, hi = cfg.update_hz_range
    assert ((det.rate_hz >= lo) & (det.rate_hz <= hi)).all()
    assert ((det.dropout_p >= cfg.dropout_range[0]) & (det.dropout_p <= cfg.dropout_range[1])).all()
    assert det.latency_steps.max().item() <= det.buffer_len - 1
