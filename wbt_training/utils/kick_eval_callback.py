"""Holosoma eval callback that records kick metrics in IsaacSim (plan, Phase 5).

Attach with ``eval_student.py --kick-metrics.enabled True`` (or
``eval_teacher.py``); writes a JSON report with the summary, the PHP-style grid
and every episode record (see :mod:`wbt_training.utils.kick_metrics`).

Per step it reads the ball and robot state from the simulator, the commanded
(v, omega) and ``kick_phase`` from the motion command, and robot-ball contacts
from Phase 3's :class:`~wbt_training.config_values.lupus_kick.BallContactDetector`.
Episodes are closed on ``dones``; ``terminated`` = done and not timeout.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
from holosoma.agents.callbacks.base_callback import RLEvalCallback
from loguru import logger

from wbt_training.config_values.kick_command_obs import frame_field
from wbt_training.utils.kick_metrics import (
    KickEpisodeTracker,
    KickEvalGrid,
    KickMetricsCfg,
    summarize,
    table_to_markdown,
    write_report,
)


@dataclass(frozen=True)
class KickMetricsConfig:
    enabled: bool = False
    """Record kick metrics during evaluation."""
    output_path: str = "kick_metrics.json"
    """Report path (relative paths go into the eval log dir)."""
    window_s: float = 0.5
    min_kick_speed: float = 0.5
    max_dir_err_deg: float = 15.0
    approach: tuple[float, ...] = KickEvalGrid().approach
    lateral: tuple[float, ...] = KickEvalGrid().lateral
    speed: tuple[float, ...] = KickEvalGrid().speed
    episodes_per_cell: int = 100
    foot_offset_y: float = 0.133
    """Half track width of LUPUS (m); the lateral grid axis is measured to the nearer foot."""


@dataclass(frozen=True)
class KickEvalArgs:
    """CLI wrapper: ``--kick-metrics.enabled True --kick-metrics.output-path ...``."""

    kick_metrics: KickMetricsConfig = KickMetricsConfig()


def _yaw(q_xyzw: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q_xyzw.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _up_z(q_xyzw: torch.Tensor) -> torch.Tensor:
    x, y, _, _ = q_xyzw.unbind(-1)
    return 1 - 2 * (x * x + y * y)


def collect_step(env: Any, contact: torch.Tensor) -> dict[str, torch.Tensor]:
    """One :class:`KickEpisodeTracker` step from a holosoma LUPUS kick env."""
    from wbt_training.config_values.lupus_kick import FOOT_BODY_NAMES, ball_state_w

    sim = env.simulator
    root = sim.robot_root_states
    q = root[:, 3:7]
    yaw = _yaw(q)
    heading = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
    ball = ball_state_w(env)
    cmd = env.command_manager.get_state("motion_command")
    n, dev = env.num_envs, env.device
    vel_cmd = frame_field(cmd, "vel_cmd", n, dev, width=2)
    target = getattr(cmd, "contact_heading", torch.full((n,), float("nan"), device=dev))
    target = target + frame_field(cmd, "kick_dir", n, dev)
    out = {
        "ball_pos_w": ball[:, :3],
        "ball_vel_w": ball[:, 7:10],
        "base_pos_w": root[:, :3],
        "base_yaw": yaw,
        "base_up_z": _up_z(q),
        "base_fwd_vel": (root[:, 7:9] * heading).sum(-1),
        "base_yaw_rate": root[:, 12],
        "cmd_v": vel_cmd[:, 0],
        "cmd_w": vel_cmd[:, 1],
        "kick_phase": frame_field(cmd, "kick_phase", n, dev),
        "contact": contact,
        "target_heading": target,
        # Phase 6 dribble metrics (kick_mode is 0 for commands without the field)
        "kick_mode": frame_field(cmd, "kick_mode", n, dev),
    }
    feet = _feet_pos_w(env, FOOT_BODY_NAMES)
    if feet is not None:
        out["feet_pos_w"] = feet
    return out


def _feet_pos_w(env: Any, names: list[str]) -> torch.Tensor | None:
    sim = env.simulator
    body_names = getattr(sim, "body_names", None)
    pos = getattr(sim, "_rigid_body_pos", None)
    if body_names is None or pos is None:
        return None
    body_names = list(body_names)
    return pos[:, [body_names.index(nm) for nm in names], :3]


class KickMetricsEvalCallback(RLEvalCallback):
    def __init__(self, config: KickMetricsConfig, training_loop: Any):
        super().__init__(config, training_loop)
        env = training_loop.env
        self.env = env
        self.cfg = KickMetricsCfg(
            dt=float(env.dt), window_s=config.window_s, min_kick_speed=config.min_kick_speed, max_dir_err_deg=config.max_dir_err_deg,
            foot_offset_y=config.foot_offset_y,
        )
        self.grid = KickEvalGrid(config.approach, config.lateral, config.speed, config.episodes_per_cell)
        self.tracker = KickEpisodeTracker(env.num_envs, self.cfg, env.device)
        from wbt_training.config_values.lupus_kick import BallContactDetector

        self.contacts = BallContactDetector(env)
        path = config.output_path
        if not os.path.isabs(path):
            path = os.path.join(str(getattr(training_loop, "log_dir", ".")), path)
        self.output_path = path

    def on_post_eval_env_step(self, actor_state: dict[str, Any]) -> dict[str, Any]:
        dones = actor_state.get("dones")
        contact = self.contacts.update()
        if dones is not None:
            done = torch.as_tensor(dones).bool().flatten()
            ids = torch.nonzero(done, as_tuple=False).flatten()
            if ids.numel():
                timeouts = actor_state.get("extras", {}).get("time_outs")
                timeout = torch.as_tensor(timeouts).bool().flatten()[ids] if timeouts is not None else torch.zeros_like(ids, dtype=torch.bool)
                self.tracker.finish(ids, ~timeout.cpu())
                self.contacts.reset(ids)
                contact = contact.clone()
                contact[ids] = False
        self.tracker.step(collect_step(self.env, contact))
        return actor_state

    def on_post_evaluate_policy(self) -> None:
        recs = self.tracker.records
        report = write_report(self.output_path, recs, self.grid)
        logger.info(f"Kick metrics ({len(recs)} episodes) -> {self.output_path}")
        logger.info(f"Kick summary: {report['summary']}")
        print(table_to_markdown(report["grid"]), flush=True)


def attach_kick_metrics(algo: Any, config: KickMetricsConfig) -> KickMetricsEvalCallback | None:
    """Append the callback to ``algo.eval_callbacks`` if ``config.enabled``."""
    if not config.enabled:
        return None
    cb = KickMetricsEvalCallback(config, algo)
    algo.eval_callbacks.append(cb)
    return cb


__all__ = ["KickEvalArgs", "KickMetricsConfig", "KickMetricsEvalCallback", "attach_kick_metrics", "collect_step", "summarize"]
