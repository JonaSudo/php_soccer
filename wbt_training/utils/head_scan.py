"""Head-scan generator: moves the neck joints in simulation like the behavior.

On LUPUS the neck (``head_yaw``, ``head_pitch``) belongs to the behavior, not
to the policy. The behavior scans left/right a lot and sometimes tracks the
ball. The policy must learn to live with the moving head mass, so in training
the neck follows a random scan trajectory per episode (plan, Phase 3,
"Domain Randomization").

The generator is a stateful randomization term (setup + reset + step under the
same name). Each episode draws its own scan pattern from the term's sampler:

* **sweep**: triangle wave on ``head_yaw`` (constant angular speed between the
  turning points, like a behavior scan), with a small pitch nod per sweep;
* **ball tracking**: one optional window per episode where the head looks at
  the ball, plus (per episode, with ``track_on_kick_prob``) the behavior rule
  from the plan's risk table: when a kick is pending (``kick_enable`` = 1 and
  the ball closer than ``track_on_kick_dist``) the head follows the ball;
* **hold**: with ``hold_prob`` the head stays still for the whole episode.

The commanded angle is rate-limited (``max_speed``) and clipped to the joint
limits, then written as an offset from the default pose into
``PassiveNeckJointPositionActionTerm.passive_targets`` (action units). The
policy's own neck outputs are ignored by that term.

The math lives in pure functions (:func:`triangle_wave`, :func:`look_at_angles`,
:func:`rate_limit`) so it can be tested without a simulator.
"""

from __future__ import annotations

from typing import Any

import torch
from holosoma.managers.randomization.base import RandomizationTermBase
from holosoma.utils.rotations import quat_rotate_inverse

# Defaults roughly follow a soccer behavior scan; tune once the LUPUS behavior
# is known (plan, open question "Kopf-Scan").
DEFAULT_PARAMS: dict[str, Any] = {
    "yaw_joint": "head_yaw",
    "pitch_joint": "head_pitch",
    "head_body": "head_link",
    "action_term": "joint_control",
    "object_name": "ball",
    # +1 when positive head_pitch looks down (LUPUS, checked by FK in tests)
    "pitch_down_sign": 1.0,
    # +1 when positive head_yaw looks left (LUPUS, checked by FK in tests)
    "yaw_left_sign": 1.0,
    "yaw_amplitude": [0.5, 1.0],  # rad, half sweep width
    "yaw_center": [-0.1, 0.1],  # rad
    "yaw_speed": [1.0, 3.0],  # rad/s along the sweep
    "pitch_center": [0.15, 0.6],  # rad, looking down
    "pitch_amplitude": [0.0, 0.15],  # rad, nod per sweep
    "hold_prob": 0.1,
    "track_prob": 0.5,  # episodes with one ball-tracking window
    "track_start_s": [0.0, 8.0],
    "track_duration_s": [0.5, 3.0],
    "track_on_kick_prob": 0.7,
    "track_on_kick_dist": 1.0,  # m
    "max_speed": 6.0,  # rad/s, below the neck motor's 7.85 rad/s
    "limit_margin": 0.02,  # rad inside the joint limits
    "enabled": True,
}


def triangle_wave(t: torch.Tensor, period: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """Triangle wave in [-1, 1] with constant slope (a sweep at constant speed)."""
    u = torch.remainder(t / period + phase, 1.0)
    return 4.0 * torch.abs(u - 0.5) - 1.0


def look_at_angles(
    target_b: torch.Tensor,
    head_b: torch.Tensor,
    yaw_left_sign: float = 1.0,
    pitch_down_sign: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Neck yaw/pitch that point the head from ``head_b`` at ``target_b`` (both base frame)."""
    d = target_b - head_b
    yaw = torch.atan2(d[:, 1], d[:, 0])
    pitch_down = torch.atan2(-d[:, 2], torch.linalg.norm(d[:, :2], dim=1).clamp_min(1e-6))
    return yaw_left_sign * yaw, pitch_down_sign * pitch_down


def rate_limit(prev: torch.Tensor, target: torch.Tensor, max_step: float) -> torch.Tensor:
    """Move from ``prev`` towards ``target`` by at most ``max_step`` per call."""
    return prev + torch.clamp(target - prev, -max_step, max_step)


class HeadScanGenerator(RandomizationTermBase):
    """Drives the passive neck joints with a random scan per episode (see module doc)."""

    _STREAMS = (
        "yaw_amplitude",
        "yaw_center",
        "yaw_speed",
        "pitch_center",
        "pitch_amplitude",
        "phase",
        "hold",
        "track",
        "track_start_s",
        "track_duration_s",
        "track_on_kick",
    )

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        self.p = {**DEFAULT_PARAMS, **(cfg.params or {})}
        self.enabled = bool(self.p["enabled"])
        dof_names = list(env.robot_config.dof_names)
        self.yaw_id = dof_names.index(self.p["yaw_joint"])
        self.pitch_id = dof_names.index(self.p["pitch_joint"])
        lower = torch.tensor(env.robot_config.dof_pos_lower_limit_list, dtype=torch.float32, device=env.device)
        upper = torch.tensor(env.robot_config.dof_pos_upper_limit_list, dtype=torch.float32, device=env.device)
        ids = torch.tensor([self.yaw_id, self.pitch_id], device=env.device)
        margin = float(self.p["limit_margin"])
        self.lower = lower[ids] + margin
        self.upper = upper[ids] - margin
        defaults = env.robot_config.init_state.default_joint_angles
        self.default = torch.tensor(
            [defaults[self.p["yaw_joint"]], defaults[self.p["pitch_joint"]]], dtype=torch.float32, device=env.device
        )
        n = env.num_envs
        self.ep = {k: torch.zeros(n, device=env.device) for k in self._STREAMS}
        self.angles = self.default.repeat(n, 1)  # commanded [yaw, pitch]
        self.needs_init = torch.ones(n, dtype=torch.bool, device=env.device)
        self.tracking = torch.zeros(n, dtype=torch.bool, device=env.device)

    # ------------------------------------------------------------ lifecycle

    def setup(self, sampler: Any) -> None:
        self._draw(torch.arange(self.env.num_envs, device=self.env.device), sampler)

    def reset(self, env_ids: torch.Tensor | None, sampler: Any) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, device=self.env.device)
        env_ids = torch.as_tensor(env_ids, device=self.env.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        self._draw(env_ids, sampler)
        # the motion command resets dof_pos after this term; start from there on the next step
        self.needs_init[env_ids] = True

    def step(self, sampler: Any) -> None:
        if not self.enabled:
            return
        env = self.env
        if self.needs_init.any():
            ids = torch.where(self.needs_init)[0]
            dof_pos = env.simulator.dof_pos[ids]
            self.angles[ids, 0] = dof_pos[:, self.yaw_id]
            self.angles[ids, 1] = dof_pos[:, self.pitch_id]
            self.needs_init[ids] = False
        target = self.target_angles()
        limited = rate_limit(self.angles, target, self.p["max_speed"] * env.dt)
        self.angles = torch.minimum(torch.maximum(limited, self.lower), self.upper)
        term = env.action_manager.get_term(self.p["action_term"])
        scale = float(env.robot_config.control.action_scale)
        term.passive_targets[:] = (self.angles - self.default) / scale

    # -------------------------------------------------------------- targets

    def scan_angles(self, t: torch.Tensor) -> torch.Tensor:
        """Sweep target [num_envs, 2] at episode time ``t`` (seconds)."""
        ep = self.ep
        period = 4.0 * ep["yaw_amplitude"] / ep["yaw_speed"].clamp_min(1e-3)
        tri = triangle_wave(t, period, ep["phase"])
        yaw = ep["yaw_center"] + ep["yaw_amplitude"] * tri
        # nod once per sweep: down at the turning points, up in the middle
        pitch = ep["pitch_center"] + ep["pitch_amplitude"] * (2.0 * tri.abs() - 1.0)
        hold = ep["hold"] > 0.5
        yaw = torch.where(hold, ep["yaw_center"], yaw)
        pitch = torch.where(hold, ep["pitch_center"], pitch)
        return torch.stack([yaw, self.p["pitch_down_sign"] * pitch], dim=1)

    def target_angles(self) -> torch.Tensor:
        env = self.env
        t = env.episode_length_buf.to(torch.float32) * env.dt
        target = self.scan_angles(t)
        ball = self._ball_and_head_b()
        if ball is None:
            self.tracking.zero_()
            return target
        ball_b, head_b = ball
        ep = self.ep
        in_window = (ep["track"] > 0.5) & (t >= ep["track_start_s"]) & (t < ep["track_start_s"] + ep["track_duration_s"])
        kick_enable = self._kick_enable()
        near = torch.linalg.norm(ball_b[:, :2], dim=1) < self.p["track_on_kick_dist"]
        on_kick = (ep["track_on_kick"] > 0.5) & (kick_enable > 0.5) & near
        self.tracking = in_window | on_kick
        yaw_t, pitch_t = look_at_angles(ball_b, head_b, self.p["yaw_left_sign"], self.p["pitch_down_sign"])
        look = torch.stack([yaw_t, pitch_t], dim=1)
        return torch.where(self.tracking[:, None], look, target)

    # -------------------------------------------------------------- helpers

    def _draw(self, env_ids: torch.Tensor, sampler: Any) -> None:
        dev = self.env.device
        p = self.p
        for axis, key in enumerate(self._STREAMS):
            if key in ("hold", "track", "track_on_kick"):
                prob = {"hold": p["hold_prob"], "track": p["track_prob"], "track_on_kick": p["track_on_kick_prob"]}[key]
                u = sampler.draw([0.0, 1.0], env_ids=env_ids, coords=(axis,), device=dev)
                self.ep[key][env_ids] = (u < prob).float()
            elif key == "phase":
                self.ep[key][env_ids] = sampler.draw([0.0, 1.0], env_ids=env_ids, coords=(axis,), device=dev)
            else:
                self.ep[key][env_ids] = sampler.draw(p[key], env_ids=env_ids, coords=(axis,), device=dev)

    def _ball_and_head_b(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Ball and head position in the robot base frame, or None without a ball."""
        sim = self.env.simulator
        name = self.p["object_name"]
        registry_names = _individual_names(sim)
        if name not in registry_names:
            return None
        env_ids = torch.arange(self.env.num_envs, device=self.env.device)
        ball_w = sim.get_actor_states([name], env_ids)[:, :3]
        root = sim.robot_root_states
        head_idx = sim.body_names.index(self.p["head_body"])
        head_w = sim._rigid_body_pos[:, head_idx]
        q = root[:, 3:7]
        ball_b = quat_rotate_inverse(q, ball_w - root[:, :3], w_last=True)
        head_b = quat_rotate_inverse(q, head_w - root[:, :3], w_last=True)
        return ball_b, head_b

    def _kick_enable(self) -> torch.Tensor:
        cmd = self.env.command_manager.get_state("motion_command")
        ke = getattr(cmd, "kick_enable", None)
        if ke is None:
            return torch.zeros(self.env.num_envs, device=self.env.device)
        return ke.view(-1)


def _individual_names(sim: Any) -> list[str]:
    from holosoma.simulator.shared.object_registry import ObjectType

    return list(sim.object_registry.get_names_by_type(ObjectType.INDIVIDUAL))


__all__ = ["DEFAULT_PARAMS", "HeadScanGenerator", "look_at_angles", "rate_limit", "triangle_wave"]
