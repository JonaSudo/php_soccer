"""Realistic behavior-stack command jitter for the Phase-5 jitter test.

The behavior on the robot does not send a clean (v, omega, kick_enable): its
estimate of where to go is noisy, it updates at its own rate and messages can
arrive late. :class:`CommandJitter` reproduces that on top of the command the
student observes, either from a parametric model or by replaying residuals
recorded from the real behavior stack:

* Ornstein-Uhlenbeck noise on v and omega (``sigma_*``, correlation time ``tau_s``)
* behavior update rate ``update_hz``: between updates the last value is held
* stale messages: with ``stale_prob`` per update the update is skipped
* ``kick_enable`` glitches: single-step flips with ``kick_glitch_prob``
* replay (``log_file``): an ``.npz`` with ``v`` and ``omega`` arrays sampled at the
  policy rate; the residual against a moving average (``log_smooth_steps``) is
  added from a random offset per env, so the noise spectrum is the real one

Used by :class:`~wbt_training.config_values.kick_command_obs.StudentCommandObsTerm`
when its params contain ``jitter`` (a dict of :class:`CommandJitterCfg` fields),
and by the MuJoCo kick Sim2Sim. Training does not use it by default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class CommandJitterCfg:
    sigma_v: float = 0.05  # m/s, stationary std of the OU noise
    sigma_omega: float = 0.10  # rad/s
    tau_s: float = 0.3  # correlation time
    update_hz: float = 20.0  # behavior rate; >= policy rate means every step
    stale_prob: float = 0.05
    kick_glitch_prob: float = 0.0
    log_file: str | None = None
    log_smooth_steps: int = 25
    dt: float = 0.02

    @classmethod
    def from_params(cls, params: dict[str, Any] | None) -> "CommandJitterCfg":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (params or {}).items() if k in names})


def load_residuals(path: str, smooth_steps: int) -> torch.Tensor:
    """``[T, 2]`` residual of a recorded (v, omega) log against its moving average."""
    with np.load(path) as data:
        x = np.stack([np.asarray(data["v"], float), np.asarray(data["omega"], float)], axis=-1)
    k = max(1, int(smooth_steps))
    kernel = np.ones(k) / k
    smooth = np.stack([np.convolve(x[:, i], kernel, mode="same") for i in range(2)], axis=-1)
    return torch.as_tensor(x - smooth, dtype=torch.float32)


class CommandJitter:
    """Stateful per-env jitter; :meth:`apply` maps a clean command to the jittered one."""

    def __init__(
        self,
        num_envs: int,
        cfg: CommandJitterCfg,
        device: str | torch.device = "cpu",
        generator: torch.Generator | None = None,
    ):
        self.cfg = cfg
        self.n = num_envs
        self.device = torch.device(device)
        self.g = generator
        self.ou = torch.zeros(num_envs, 2, device=self.device)
        self.held = torch.zeros(num_envs, 2, device=self.device)
        self.held_kick = torch.zeros(num_envs, device=self.device)
        self.phase = torch.zeros(num_envs, device=self.device)
        self.fresh = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        self.residuals = load_residuals(cfg.log_file, cfg.log_smooth_steps).to(self.device) if cfg.log_file else None
        self.offset = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.reset(None)

    def reset(self, env_ids: torch.Tensor | None) -> None:
        ids = torch.arange(self.n, device=self.device) if env_ids is None else torch.as_tensor(env_ids, device=self.device).long()
        self.ou[ids] = 0.0
        self.fresh[ids] = True
        self.phase[ids] = torch.rand(ids.numel(), generator=self.g, device=self.device)
        if self.residuals is not None:
            self.offset[ids] = (torch.rand(ids.numel(), generator=self.g, device=self.device) * len(self.residuals)).long()

    def apply(self, vel: torch.Tensor, kick: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``vel`` [N, 2] = (v, omega), ``kick`` [N] in {0, 1} -> jittered copies."""
        c = self.cfg
        a = math.exp(-c.dt / max(c.tau_s, 1e-6))
        sig = torch.tensor([c.sigma_v, c.sigma_omega], device=self.device)
        noise = torch.randn(self.n, 2, generator=self.g, device=self.device)
        self.ou = a * self.ou + math.sqrt(1 - a * a) * sig * noise
        target = vel + self.ou
        if self.residuals is not None:
            idx = self.offset % len(self.residuals)
            target = vel + self.residuals[idx]
            self.offset += 1

        self.phase += c.update_hz * c.dt
        update = (self.phase >= 1.0) | self.fresh
        self.phase = torch.where(self.phase >= 1.0, self.phase - 1.0, self.phase)
        stale = torch.rand(self.n, generator=self.g, device=self.device) < c.stale_prob
        update &= ~stale | self.fresh
        self.held = torch.where(update[:, None], target, self.held)
        self.held_kick = torch.where(update, kick.float(), self.held_kick)
        self.fresh[:] = False

        out_kick = self.held_kick.clone()
        if c.kick_glitch_prob > 0:
            glitch = torch.rand(self.n, generator=self.g, device=self.device) < c.kick_glitch_prob
            out_kick = torch.where(glitch, 1.0 - out_kick, out_kick)
        return self.held.clone(), out_kick


__all__ = ["CommandJitter", "CommandJitterCfg", "load_residuals"]
