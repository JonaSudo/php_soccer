"""Observation / algo building blocks for the LUPUS kick student (plan, Phase 4).

Replaces PHP's depth distillation (``depth_distillation.py``): the student gets
only what LUPUS has on the field, i.e. proprioception, the behavior's command
and the noisy ball position from the vision stack. There is no depth camera and
no depth backbone. The rest of the PHP recipe (DAgger + PPO, relaxing
termination thresholds, PPO gate, adaptive sampling off) is kept.

Groups (names fixed by holosoma's ``DistillationPPO``):

* ``actor_obs`` (student, history :data:`STUDENT_HISTORY` frames): projected
  gravity, base angular velocity, joint positions and velocities of all 22 DoFs
  (the two neck joints are pure information: the behavior moves the head, the
  policy can account for it but does not drive it), last action, the command
  ``[v, omega, kick_enable]`` (:mod:`kick_command_obs`) and ``ball_detection``
  ``[x, y, visible, time_since_seen]`` (:class:`BallDetectionObsTerm`).
* ``teacher_obs`` / ``critic_obs``: the teacher's privileged view with the true
  ball state, taken from the teacher preset (``lupus_kick.py``, Phase 3).
* ``depth_camera``: a one-column zero placeholder, only because
  ``DistillationPPO`` requires the group. :class:`NoDepthBackbone` maps it to a
  zero-width latent, so the student MLP sees exactly ``actor_obs`` and exports
  as one ONNX (:class:`wbt_training.utils.student_teacher.BallDistillationPPO`).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
from torch import nn

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.managers.observation.base import ObservationTermBase

from wbt_training.utils.ball_detection import BallDetector, BallVisionCfg

#: Frames of history in the student's actor_obs (plan: 5-10 frames).
STUDENT_HISTORY = 8

# ---------------------------------------------------------------------------
# Ball detection observation term
# ---------------------------------------------------------------------------


class BallDetectionObsTerm(ObservationTermBase):
    """``[x, y, visible, time_since_seen]`` of the ball as LUPUS's vision reports it.

    Reads the ball's world position from the simulator object registry
    (``params["object_name"]``, default: the first free body, as holosoma's
    ``MotionCommand`` does), the camera pose from the ``head_link`` body (moved
    by the head-scan generator through the passive neck joints) and the robot
    root, then runs :class:`~wbt_training.utils.ball_detection.BallDetector`.
    Every :class:`BallVisionCfg` field can be overridden through ``params``.
    """

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = dict(cfg.params or {})
        self.vision_cfg = BallVisionCfg.from_params(params)
        self.object_name = params.get("object_name")
        self.head_body_name = params.get("head_body_name", "head_link")
        self.num_envs = env.num_envs
        self.device = env.device
        dt = float(getattr(env, "dt", self.vision_cfg.policy_dt))
        self.detector = BallDetector(self.num_envs, self.vision_cfg, device=self.device, dt=dt)
        self._head_index: int | None = None
        self._cached: torch.Tensor | None = None

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self.detector.reset(env_ids)

    def _resolve(self, env: Any) -> None:
        if self._head_index is None:
            bodies = list(env.simulator._body_list)
            if self.head_body_name not in bodies:
                raise KeyError(f"BallDetectionObsTerm: body {self.head_body_name!r} not in {bodies}")
            self._head_index = bodies.index(self.head_body_name)
        if self.object_name is None:
            from holosoma.simulator.shared.object_registry import ObjectType

            names = env.simulator.object_registry.get_names_by_type(ObjectType.INDIVIDUAL)
            if not names:
                raise RuntimeError("BallDetectionObsTerm needs a ball (free rigid body) in the scene")
            self.object_name = names[0]

    def __call__(self, env: Any, *, modify_history: bool = True, **kwargs: Any) -> torch.Tensor:
        if not modify_history and self._cached is not None:
            return self._cached.clone()
        self._resolve(env)
        sim = env.simulator
        ball = sim.get_actor_states([self.object_name], torch.arange(self.num_envs, device=self.device))[:, :3]
        root = sim.robot_root_states
        out = self.detector.step(
            ball_pos_w=ball,
            head_pos_w=sim._rigid_body_pos[:, self._head_index],
            head_quat_w=sim._rigid_body_rot[:, self._head_index],
            base_pos_w=root[:, :3],
            base_quat_w=root[:, 3:7],
        )
        self._cached = out.clone()
        return out


# ---------------------------------------------------------------------------
# No-depth placeholder
# ---------------------------------------------------------------------------


def no_depth_placeholder(env: Any) -> torch.Tensor:
    """One zero column per env for the ``depth_camera`` group ``DistillationPPO`` requires."""
    return torch.zeros(env.num_envs, 1, device=env.device)


class NoDepthBackbone(nn.Module):
    """Depth backbone with a zero-width latent: the student sees only ``actor_obs``.

    ``StudentTeacherModuleConfig.depth_backbone`` must point here and
    ``depth_output_dim`` must be 0.
    """

    def __init__(self, output_dim: int = 0, *args: Any, **kwargs: Any):
        super().__init__()
        if output_dim != 0:
            raise ValueError("NoDepthBackbone has no latent; set depth_output_dim=0")
        self.output_dim = 0

    def forward(self, depth_obs: torch.Tensor) -> torch.Tensor:
        return depth_obs.new_zeros(depth_obs.shape[0], 0)


NO_DEPTH_BACKBONE = "wbt_training.config_values.ball_distillation.NoDepthBackbone"

# ---------------------------------------------------------------------------
# Student observation group
# ---------------------------------------------------------------------------

#: Student command term; ``params`` are those of ``StudentCommandObsTerm``.
student_command_term = ObsTermCfg(
    func="wbt_training.config_values.kick_command_obs:StudentCommandObsTerm",
    scale=1.0,
    noise=0.0,
    params={
        "command_name": "motion_command",
        "mode": "enable",
        "include_kick_dir": False,
        "v_noise_std": 0.03,
        "omega_noise_std": 0.05,
        # Ranges the behavior can send (plan, open question: v_min/v_max/omega_max).
        "v_range": [0.0, 1.0],
        "omega_max": 1.0,
        "commit_replace_prob": 0.5,
        "commit_onset_max_steps": 25,
        "flicker_prob": 0.01,
        "flicker_max_steps": 2,
        "flicker_gap_steps": 10,
        # Phase-5 jitter test (eval only), see utils/command_jitter.py
        "jitter_enabled": False,
        "jitter_sigma_v": 0.05,
        "jitter_sigma_omega": 0.10,
        "jitter_tau_s": 0.3,
        "jitter_update_hz": 20.0,
        "jitter_stale_prob": 0.05,
        "jitter_kick_glitch_prob": 0.0,
        "jitter_log_file": "",
    },
)

ball_detection_term = ObsTermCfg(
    func="wbt_training.config_values.ball_distillation:BallDetectionObsTerm",
    scale=1.0,
    noise=0.0,  # all noise comes from the vision model
    params={"head_body_name": "head_link", **BallVisionCfg().as_params()},
)

student_actor_terms: dict[str, ObsTermCfg] = {
    "robot_anchor_projected_gravity": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:robot_anchor_projected_gravity", scale=1.0, noise=0.05
    ),
    "base_ang_vel": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:base_ang_vel", scale=1.0, noise=0.2),
    # All 22 DoFs: the neck angle/velocity is information about the behavior's
    # head motion, the policy does not drive the neck.
    "dof_pos": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:dof_pos", scale=1.0, noise=0.01),
    "dof_vel": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:dof_vel", scale=1.0, noise=0.5),
    "actions": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:actions", scale=1.0, noise=0.0),
    "command": student_command_term,
    "ball_detection": ball_detection_term,
}

lupus_ball_student_actor_obs = ObsGroupCfg(
    concatenate=True, enable_noise=True, history_length=STUDENT_HISTORY, terms=student_actor_terms
)

lupus_no_depth_obs = ObsGroupCfg(
    concatenate=False,
    enable_noise=False,
    history_length=1,
    terms={
        "depth_cam": ObsTermCfg(
            func="wbt_training.config_values.ball_distillation:no_depth_placeholder", scale=1.0, noise=0.0
        )
    },
)


def make_ball_distill_observation(
    teacher_obs: ObsGroupCfg,
    critic_obs: ObsGroupCfg,
    *,
    actor_obs: ObsGroupCfg = lupus_ball_student_actor_obs,
    routing_term: bool = True,
) -> ObservationManagerCfg:
    """Student observation manager around a teacher preset's privileged groups.

    ``teacher_obs`` is the teacher's actor group (true ball state, no noise);
    the ``which_motion`` routing term is added unless already present, as in
    PHP's depth distillation.
    """
    terms = dict(teacher_obs.terms)
    if routing_term and "which_motion" not in terms:
        terms["which_motion"] = ObsTermCfg(
            func="wbt_training.config_values.motion_command:which_motion",
            scale=1.0,
            noise=0.0,
            params={
                "bad_motion_body_pos_body_names": (
                    "left_foot_contact_point",
                    "right_foot_contact_point",
                ),
            },
        )
    return ObservationManagerCfg(
        groups={
            "actor_obs": actor_obs,
            "critic_obs": replace(critic_obs, enable_noise=False),
            "teacher_obs": replace(teacher_obs, enable_noise=False, terms=terms),
            "depth_camera": lupus_no_depth_obs,
        }
    )


def with_phase6_command(obs: ObservationManagerCfg, *, kick_dir: bool = True, dribble: bool = False) -> ObservationManagerCfg:
    """Phase 6: add ``kick_dir`` and/or the three-valued kick mode to the student command."""
    actor = obs.groups["actor_obs"]
    cmd = actor.terms["command"]
    params = {**cmd.params, "include_kick_dir": kick_dir, "mode": "mode3" if dribble else "enable"}
    actor = replace(actor, terms={**actor.terms, "command": replace(cmd, params=params)})
    return replace(obs, groups={**obs.groups, "actor_obs": actor})


__all__ = [
    "BallDetectionObsTerm",
    "NO_DEPTH_BACKBONE",
    "NoDepthBackbone",
    "STUDENT_HISTORY",
    "ball_detection_term",
    "lupus_ball_student_actor_obs",
    "lupus_no_depth_obs",
    "make_ball_distill_observation",
    "no_depth_placeholder",
    "student_command_term",
    "with_phase6_command",
]
