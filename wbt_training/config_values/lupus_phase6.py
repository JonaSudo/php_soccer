"""Phase 6: kick direction and dribble mode for the LUPUS walk-kick policy.

Only new inputs, rewards and scenarios, no rebuild (plan, Phase 6):

**Kick direction.** ``kick_dir`` (rad, relative to the walking direction,
first +-30 deg, default 0) comes per frame from the motion NPZ (the Phase-2
scenarios ``walk_kick``/``stand_kick`` sample it per episode with
``kick_dir_max``). The teacher observes it; :func:`ball_kick_direction_dir`
measures the kick against ``heading at contact + kick_dir`` instead of the
heading. No new clips: PPO learns to place the foot differently. The student
sees it in its command (commit-phase replacement covers it too).

**Dribble mode.** ``kick_mode`` replaces the binary flag: 0 ignore, 1 kick,
2 dribble. In ``walk_dribble`` episodes (Phase-2 scenario) the reference is
plain walking with the ball in front of the foot. Rewards:
:func:`ball_near_foot` (ball close to the nearer foot) and
:func:`ball_along_heading` (ball rolls along the heading at roughly walking
speed); ``ball_step_on`` stays. :class:`BallUntouchedUnlessDribble` replaces
``ball_untouched``, so touching the ball is only penalised when it should be
ignored. Only with this mode "ball on the path, kick off" becomes a trained
state.

Presets (``experiment.py``): ``lupus_kickdir_teacher``,
``lupus_kickdir_distill_finetune``, ``lupus_dribble_teacher``,
``lupus_dribble_distill_finetune``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
from holosoma.config_types.command import CommandTermCfg
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.observation import ObsTermCfg
from holosoma.config_types.reward import RewardTermCfg

from wbt_training.config_values import lupus_kick as LK
from wbt_training.config_values.kick_command_obs import KICK_MODE_DRIBBLE, frame_field

_SELF = "wbt_training.config_values.lupus_phase6"
KICK_DIR_MAX = float(np.deg2rad(30.0))


def load_phase6_fields(files: list[str], total_frames: int) -> dict[str, np.ndarray]:
    """``kick_dir`` (float, rad) and ``kick_mode`` (int) over ``files``; missing -> 0 / from kick_enable."""
    kd, km = [], []
    for path in files:
        with np.load(path) as data:
            n = int(data["joint_pos"].shape[0])
            kd.append(np.asarray(data["kick_dir"], np.float32).reshape(n) if "kick_dir" in data.files else np.zeros(n, np.float32))
            if "kick_mode" in data.files:
                km.append(np.asarray(data["kick_mode"], np.int64).reshape(n))
            elif "kick_enable" in data.files:
                km.append((np.asarray(data["kick_enable"]).reshape(n) > 0.5).astype(np.int64))
            else:
                km.append(np.zeros(n, np.int64))
    out = {"kick_dir": np.concatenate(kd), "kick_mode": np.concatenate(km)}
    if len(out["kick_dir"]) != total_frames:
        raise ValueError(f"phase-6 fields have {len(out['kick_dir'])} frames, motion buffer {total_frames}")
    return out


class LupusPhase6MotionCommand(LK.LupusKickMotionCommand):
    """``LupusKickMotionCommand`` plus per-frame ``kick_dir`` and ``kick_mode``."""

    def setup(self) -> None:
        super().setup()
        fields = load_phase6_fields(self._motion_files_for(self.motion), int(self.motion.time_step_total))
        assert self._kick is not None
        self._kick["kick_dir"] = torch.as_tensor(fields["kick_dir"], device=self.device).float()
        self._kick["kick_mode"] = torch.as_tensor(fields["kick_mode"], device=self.device).long()

    @property
    def kick_dir(self) -> torch.Tensor:
        return self._field("kick_dir")

    @property
    def kick_mode(self) -> torch.Tensor:
        return self._field("kick_mode")


def _cmd(env: Any) -> LupusPhase6MotionCommand:
    cmd = env.command_manager.get_state("motion_command")
    if not hasattr(cmd, "kick_mode"):
        raise TypeError(f"Phase-6 terms need LupusPhase6MotionCommand, got {type(cmd).__name__}")
    return cmd


# --------------------------------------------------------------------------- observations


def kick_dir(env: Any) -> torch.Tensor:
    """Teacher observation: ``kick_dir`` in rad, [num_envs, 1]."""
    return _cmd(env).kick_dir.view(env.num_envs, 1)


def kick_mode_onehot(env: Any) -> torch.Tensor:
    """Teacher observation: one-hot (ignore, kick, dribble), [num_envs, 3]."""
    return torch.nn.functional.one_hot(_cmd(env).kick_mode.clamp(0, 2), 3).float()


# --------------------------------------------------------------------------- rewards


def ball_kick_direction_dir(
    env: Any,
    window_s: float = 0.5,
    max_speed: float = 4.0,
    lateral_weight: float = 1.0,
    object_name: str = LK.BALL_NAME,
) -> torch.Tensor:
    """:func:`lupus_kick.ball_kick_direction` measured against heading at contact + ``kick_dir``."""
    cmd = _cmd(env)
    fsc = cmd.frames_since_contact
    active = (fsc >= 0) & (fsc.float() * env.dt <= window_s)
    heading = torch.where(
        torch.isnan(cmd.contact_heading),
        LK.yaw_from_quat_xyzw(env.simulator.robot_root_states[:, 3:7]),
        cmd.contact_heading,
    ) + cmd.kick_dir
    v = LK.ball_state_w(env, object_name)[:, 7:9]
    h = torch.stack([torch.cos(heading), torch.sin(heading)], dim=1)
    along = (v * h).sum(dim=1).clamp(-max_speed, max_speed)
    lateral = torch.abs(v[:, 0] * h[:, 1] - v[:, 1] * h[:, 0])
    return (along - lateral_weight * lateral) * active.float()


def _feet_xy(env: Any) -> torch.Tensor:
    names = list(env.simulator.body_names)
    ids = [names.index(n) for n in LK.FOOT_BODY_NAMES]
    return env.simulator._rigid_body_pos[:, ids, :2]  # [N, 2, 2]


def ball_near_foot(env: Any, sigma: float = 0.15, object_name: str = LK.BALL_NAME) -> torch.Tensor:
    """Dribble mode: ``exp(-(d / sigma)^2)`` of the ball's distance to the nearer foot."""
    cmd = _cmd(env)
    ball = LK.ball_state_w(env, object_name)[:, :2]
    d = (_feet_xy(env) - ball[:, None]).norm(dim=-1).min(dim=1).values
    return torch.exp(-((d / sigma) ** 2)) * (cmd.kick_mode == KICK_MODE_DRIBBLE).float()


def ball_along_heading(
    env: Any, sigma: float = 0.3, lateral_weight: float = 0.5, object_name: str = LK.BALL_NAME
) -> torch.Tensor:
    """Dribble mode: ball velocity along the robot heading close to the commanded v, little sideways."""
    cmd = _cmd(env)
    yaw = LK.yaw_from_quat_xyzw(env.simulator.robot_root_states[:, 3:7])
    h = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=1)
    v = LK.ball_state_w(env, object_name)[:, 7:9]
    along = (v * h).sum(dim=1)
    lateral = torch.abs(v[:, 0] * h[:, 1] - v[:, 1] * h[:, 0])
    target = frame_field(cmd, "vel_cmd", env.num_envs, along.device, width=2)[:, 0]
    r = torch.exp(-(((along - target) / sigma) ** 2)) - lateral_weight * lateral
    return r * (cmd.kick_mode == KICK_MODE_DRIBBLE).float()


class BallUntouchedUnlessDribble(LK.BallUntouched):
    """``ball_untouched`` that allows contacts in dribble mode."""

    def __call__(self, env: Any, **kwargs: Any) -> torch.Tensor:
        return super().__call__(env, **kwargs) * (_cmd(env).kick_mode != KICK_MODE_DRIBBLE).float()


# --------------------------------------------------------------------------- presets


def _with_command(cfg: ExperimentConfig) -> ExperimentConfig:
    func = f"{_SELF}:LupusPhase6MotionCommand"
    c = cfg.command
    return replace(
        cfg,
        command=replace(
            c,
            setup_terms={"motion_command": replace(c.setup_terms["motion_command"], func=func)},
            reset_terms={"motion_command": CommandTermCfg(func=func)},
            step_terms={"motion_command": CommandTermCfg(func=func)},
        ),
    )


def _teacher_obs_terms(dribble: bool) -> dict[str, ObsTermCfg]:
    terms = {"kick_dir": ObsTermCfg(func=f"{_SELF}:kick_dir", scale=1.0, noise=0.0)}
    if dribble:
        terms["kick_mode"] = ObsTermCfg(func=f"{_SELF}:kick_mode_onehot", scale=1.0, noise=0.0)
    return terms


_TEACHER_GROUPS = ("actor_obs", "critic_obs")
# In the distill presets actor_obs is the student (it gets kick_dir through its command).
_DISTILL_GROUPS = ("critic_obs", "teacher_obs")


def _with_teacher_obs(cfg: ExperimentConfig, dribble: bool, group_names: tuple[str, ...]) -> ExperimentConfig:
    groups = dict(cfg.observation.groups)
    for name in group_names:
        if name in groups:
            g = groups[name]
            groups[name] = replace(g, terms={**g.terms, **_teacher_obs_terms(dribble)})
    return replace(cfg, observation=replace(cfg.observation, groups=groups))


@dataclass(frozen=True)
class DribbleRewardCfg:
    """Dribble reward starting values (estimates, tune after the first GPU runs).

    Change them here for a new default, or per run on the CLI, e.g.
    ``--reward.terms.ball-near-foot.weight 1.0`` and
    ``--reward.terms.ball-near-foot.params '{"sigma": 0.2}'``;
    ``training_runs/dribble_sweep.py`` prints a grid of such commands.
    """

    near_foot_weight: float = 0.5
    near_foot_sigma: float = 0.15  # m
    along_heading_weight: float = 0.5
    along_heading_sigma: float = 0.3  # m/s
    along_heading_lateral_weight: float = 0.5

    def terms(self) -> dict[str, RewardTermCfg]:
        return {
            "ball_near_foot": RewardTermCfg(
                func=f"{_SELF}:ball_near_foot", weight=self.near_foot_weight, params={"sigma": self.near_foot_sigma}
            ),
            "ball_along_heading": RewardTermCfg(
                func=f"{_SELF}:ball_along_heading",
                weight=self.along_heading_weight,
                params={"sigma": self.along_heading_sigma, "lateral_weight": self.along_heading_lateral_weight},
            ),
        }


DRIBBLE_REWARDS = DribbleRewardCfg()


def _with_rewards(cfg: ExperimentConfig, dribble: bool, rewards: DribbleRewardCfg | None = None) -> ExperimentConfig:
    terms = dict(cfg.reward.terms)
    old = terms["ball_kick_direction"]
    terms["ball_kick_direction"] = replace(old, func=f"{_SELF}:ball_kick_direction_dir")
    if dribble:
        old_u = terms["ball_untouched"]
        terms["ball_untouched"] = replace(old_u, func=f"{_SELF}:BallUntouchedUnlessDribble")
        terms.update((rewards or DRIBBLE_REWARDS).terms())
    return replace(cfg, reward=replace(cfg.reward, terms=terms))


def build_phase6_teacher(
    teacher: ExperimentConfig, *, dribble: bool, rewards: DribbleRewardCfg | None = None
) -> ExperimentConfig:
    """Phase-3 teacher + kick_dir (and dribble mode) inputs and rewards."""
    cfg = _with_rewards(_with_teacher_obs(_with_command(teacher), dribble, _TEACHER_GROUPS), dribble, rewards)
    name = "lupus_dribble_teacher" if dribble else "lupus_kickdir_teacher"
    return replace(cfg, training=replace(cfg.training, name=name))


def build_phase6_distill(
    student: ExperimentConfig, *, dribble: bool, rewards: DribbleRewardCfg | None = None
) -> ExperimentConfig:
    """Phase-4 student + the same changes; student command gets kick_dir (and the 3 modes)."""
    from wbt_training.config_values.ball_distillation import with_phase6_command

    cfg = _with_rewards(_with_teacher_obs(_with_command(student), dribble, _DISTILL_GROUPS), dribble, rewards)
    cfg = replace(cfg, observation=with_phase6_command(cfg.observation, kick_dir=True, dribble=dribble))
    actor = cfg.observation.groups["actor_obs"]
    params = {**actor.terms["command"].params, "kick_dir_max": KICK_DIR_MAX}
    actor = replace(actor, terms={**actor.terms, "command": replace(actor.terms["command"], params=params)})
    cfg = replace(cfg, observation=replace(cfg.observation, groups={**cfg.observation.groups, "actor_obs": actor}))
    name = "lupus_dribble_distill_finetune" if dribble else "lupus_kickdir_distill_finetune"
    return replace(cfg, training=replace(cfg.training, name=name))


__all__ = [
    "BallUntouchedUnlessDribble",
    "DRIBBLE_REWARDS",
    "DribbleRewardCfg",
    "KICK_DIR_MAX",
    "LupusPhase6MotionCommand",
    "ball_along_heading",
    "ball_kick_direction_dir",
    "ball_near_foot",
    "build_phase6_distill",
    "build_phase6_teacher",
    "kick_dir",
    "kick_mode_onehot",
    "load_phase6_fields",
]
