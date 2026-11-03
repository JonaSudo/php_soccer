"""Student command observation for the LUPUS kick policy (plan, Phase 4 and 6).

On the robot the behavior sends ``(v, omega, kick_enable)`` and, from Phase 6,
``kick_dir`` and a three-valued kick mode. The motion reference carries the same
values per frame (``vel_cmd``, ``kick_enable``, ``kick_phase``, ``kick_dir``,
``kick_mode`` in the motion NPZ, see ``motion_matching/io/kick_io.py``). This
term turns them into what the student observes and makes the policy robust to a
noisy behavior stack, without ever changing the reference it tracks:

* **Noise** on ``v`` and ``omega`` (Gaussian, every step).
* **Commit phase.** From ``kick_phase`` 1 (run-up / commit) until the kick is
  over (``kick_phase`` back to 0), the observed command and kick flag are, with
  probability ``commit_replace_prob`` per kick, replaced by random values from a
  random onset on. The reference keeps kicking, so the policy learns not to
  abort a kick it has started.
* **Flicker.** Outside the commit phase, while the reference has
  ``kick_enable`` = 0, single short spikes to 1 (``flicker_prob`` per step,
  1..``flicker_max_steps`` long, at least ``flicker_gap_steps`` apart) teach the policy that one wrong value from the
  behavior does not trigger a free kick (the trigger in the stand is the rising
  edge of a *sustained* ``kick_enable``).

* **Jitter test** (Phase 5, eval only): ``params["jitter"]`` adds the
  behavior-stack model of :mod:`wbt_training.utils.command_jitter` on top of the
  reference command (set ``commit_replace_prob`` and ``flicker_prob`` to 0 then).

Output layout (concatenated in this order):

* ``mode="enable"`` (Phase 4): ``[v, omega, kick_enable]``
* ``mode="mode3"`` (Phase 6 dribble): ``[v, omega, ignore, kick, dribble]``
  (one-hot of ``kick_mode``)
* ``include_kick_dir=True`` (Phase 6 kick direction) inserts ``kick_dir`` (rad,
  relative to the walking direction) after ``omega``.

The command term is read through :func:`frame_field`, which accepts either a
per-env attribute on the command (``cmd.kick_enable``) or a per-frame array on
its motion loader (``cmd.motion._kick_enable``), and falls back to zeros so the
term also runs against motions that have no kick fields yet.
"""

from __future__ import annotations

from typing import Any

import torch

from holosoma.managers.observation.base import ObservationTermBase

KICK_MODE_IGNORE, KICK_MODE_KICK, KICK_MODE_DRIBBLE = 0, 1, 2


def frame_field(cmd: Any, name: str, num_envs: int, device: Any, width: int | None = None) -> torch.Tensor:
    """Current-frame value of a per-frame motion field for every env.

    Looks for ``cmd.<name>`` (a tensor already indexed by the env's timestep)
    first, then ``cmd.motion._<name>`` / ``cmd.motion.<name>`` indexed by
    ``cmd.time_steps``. Missing fields give zeros of shape ``[num_envs]`` (or
    ``[num_envs, width]``).
    """
    value = None
    try:
        value = getattr(cmd, name, None)
    except (RuntimeError, AttributeError):  # e.g. PhpMotionCommand.vel_cmd without the field
        value = None
    if value is None:
        motion = getattr(cmd, "motion", None)
        per_frame = getattr(motion, f"_{name}", None)
        if per_frame is None:
            per_frame = getattr(motion, name, None)
        if per_frame is not None and hasattr(cmd, "time_steps"):
            value = per_frame[cmd.time_steps]
    if value is None:
        shape = (num_envs,) if width is None else (num_envs, width)
        return torch.zeros(shape, device=device)
    value = torch.as_tensor(value, device=device).float()
    if width is None:
        return value.reshape(num_envs)
    return value.reshape(num_envs, width)


class StudentCommandObsTerm(ObservationTermBase):
    """Noisy, commit-robust command observation for the student (see module doc)."""

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        p = dict(cfg.params or {})
        self.num_envs = env.num_envs
        self.device = env.device
        self.command_name = p.get("command_name", "motion_command")
        self.mode = p.get("mode", "enable")
        if self.mode not in ("enable", "mode3"):
            raise ValueError(f"mode must be 'enable' or 'mode3', got {self.mode!r}")
        self.include_kick_dir = bool(p.get("include_kick_dir", False))
        self.v_noise_std = float(p.get("v_noise_std", 0.03))
        self.omega_noise_std = float(p.get("omega_noise_std", 0.05))
        self.kick_dir_noise_std = float(p.get("kick_dir_noise_std", 0.02))
        self.v_range = tuple(p.get("v_range", (0.0, 1.0)))
        self.omega_max = float(p.get("omega_max", 1.0))
        self.kick_dir_max = float(p.get("kick_dir_max", 0.5236))  # 30 deg
        self.commit_replace_prob = float(p.get("commit_replace_prob", 0.5))
        self.commit_onset_max_steps = int(p.get("commit_onset_max_steps", 25))
        self.flicker_prob = float(p.get("flicker_prob", 0.01))
        self.flicker_max_steps = int(p.get("flicker_max_steps", 2))
        seed = p.get("seed")
        self.generator = None
        if seed is not None:
            self.generator = torch.Generator(device=self.device).manual_seed(int(seed))

        n = self.num_envs
        self.in_commit = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.corrupt = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.commit_steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self.onset = torch.zeros(n, dtype=torch.long, device=self.device)
        self.repl_vel = torch.zeros(n, 2, device=self.device)
        self.repl_dir = torch.zeros(n, device=self.device)
        self.repl_mode = torch.zeros(n, dtype=torch.long, device=self.device)
        self.flicker_left = torch.zeros(n, dtype=torch.long, device=self.device)
        # Steps since the last flicker ended; a new one needs flicker_gap_steps.
        self.flicker_gap_steps = int(p.get("flicker_gap_steps", 10))
        self.since_flicker = torch.full((n,), self.flicker_gap_steps, dtype=torch.long, device=self.device)
        self._cached: torch.Tensor | None = None
        # Phase-5 jitter test: behavior-stack noise on top of the reference.
        self.jitter = None
        # Either a dict under "jitter" or flat "jitter_<field>" keys with
        # "jitter_enabled" (flat keys can be overridden from the CLI).
        jitter_params = dict(p.get("jitter") or {})
        if p.get("jitter_enabled"):
            jitter_params.update({k[len("jitter_") :]: v for k, v in p.items() if k.startswith("jitter_") and k != "jitter_enabled"})
        if p.get("jitter") or p.get("jitter_enabled"):
            from wbt_training.utils.command_jitter import CommandJitter, CommandJitterCfg

            jcfg = CommandJitterCfg.from_params({"dt": float(getattr(env, "dt", 0.02)), **jitter_params})
            self.jitter = CommandJitter(n, jcfg, device=self.device, generator=self.generator)

    # ------------------------------------------------------------------ utils

    def _rand(self, *shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=self.generator, device=self.device)

    def _randn(self, *shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=self.generator, device=self.device)

    @property
    def dim(self) -> int:
        return 2 + int(self.include_kick_dir) + (3 if self.mode == "mode3" else 1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.in_commit[env_ids] = False
        self.corrupt[env_ids] = False
        self.commit_steps[env_ids] = 0
        self.flicker_left[env_ids] = 0
        self.since_flicker[env_ids] = self.flicker_gap_steps
        if self.jitter is not None:
            ids = None if isinstance(env_ids, slice) else env_ids
            self.jitter.reset(ids)

    # ---------------------------------------------------------------- compute

    def reference(self, env: Any) -> dict[str, torch.Tensor]:
        """Command values of the reference frame (what the behavior would send)."""
        cmd = env.command_manager.get_state(self.command_name)
        n, dev = self.num_envs, self.device
        kick_enable = frame_field(cmd, "kick_enable", n, dev)
        mode = frame_field(cmd, "kick_mode", n, dev)
        # Without an explicit kick_mode, kick_enable decides kick vs ignore.
        has_mode = getattr(cmd, "kick_mode", None) is not None or getattr(getattr(cmd, "motion", None), "_kick_mode", None) is not None
        if not has_mode:
            mode = kick_enable
        return {
            "vel": frame_field(cmd, "vel_cmd", n, dev, width=2),
            "kick_enable": kick_enable,
            "kick_phase": frame_field(cmd, "kick_phase", n, dev),
            "kick_dir": frame_field(cmd, "kick_dir", n, dev),
            "kick_mode": mode.round().long().clamp(0, 2),
        }

    def __call__(self, env: Any, *, modify_history: bool = True, **kwargs: Any) -> torch.Tensor:
        if not modify_history and self._cached is not None:
            return self._cached.clone()
        ref = self.reference(env)
        n = self.num_envs
        if self.jitter is not None:
            kick = (ref["kick_mode"] == KICK_MODE_KICK).float()
            ref["vel"], kick_j = self.jitter.apply(ref["vel"], kick)
            if self.mode == "enable":
                ref["kick_mode"] = kick_j.round().long()

        # --- commit phase bookkeeping -----------------------------------
        commit_now = ref["kick_phase"] >= 1
        entering = commit_now & ~self.in_commit
        if entering.any():
            k = int(entering.sum())
            self.corrupt[entering] = self._rand(k) < self.commit_replace_prob
            self.onset[entering] = (self._rand(k) * (self.commit_onset_max_steps + 1)).long()
            self.commit_steps[entering] = 0
            lo, hi = self.v_range
            self.repl_vel[entering, 0] = lo + (hi - lo) * self._rand(k)
            self.repl_vel[entering, 1] = (2 * self._rand(k) - 1) * self.omega_max
            self.repl_dir[entering] = (2 * self._rand(k) - 1) * self.kick_dir_max
            n_modes = 3 if self.mode == "mode3" else 2
            self.repl_mode[entering] = (self._rand(k) * n_modes).long().clamp(max=n_modes - 1)
        self.in_commit = commit_now
        self.commit_steps = torch.where(commit_now, self.commit_steps + 1, torch.zeros_like(self.commit_steps))
        self.corrupt &= commit_now
        replaced = self.corrupt & (self.commit_steps > self.onset)

        # --- flicker of the kick flag before a trigger ----------------------
        idle = ~commit_now & (ref["kick_mode"] == KICK_MODE_IGNORE)
        ready = (self.flicker_left == 0) & (self.since_flicker >= self.flicker_gap_steps)
        start = idle & ready & (self._rand(n) < self.flicker_prob)
        if start.any():
            length = 1 + (self._rand(int(start.sum())) * self.flicker_max_steps).long().clamp(max=self.flicker_max_steps - 1)
            self.flicker_left[start] = length
        flicker = idle & (self.flicker_left > 0)
        self.flicker_left = torch.where(flicker, self.flicker_left - 1, torch.zeros_like(self.flicker_left))
        self.since_flicker = torch.where(flicker, torch.zeros_like(self.since_flicker), self.since_flicker + 1)

        # --- assemble -------------------------------------------------------
        vel = ref["vel"].clone()
        vel[:, 0] += self.v_noise_std * self._randn(n)
        vel[:, 1] += self.omega_noise_std * self._randn(n)
        vel = torch.where(replaced[:, None], self.repl_vel, vel)

        mode = torch.where(replaced, self.repl_mode, ref["kick_mode"])
        mode = torch.where(flicker, torch.full_like(mode, KICK_MODE_KICK), mode)

        parts = [vel]
        if self.include_kick_dir:
            kd = ref["kick_dir"] + self.kick_dir_noise_std * self._randn(n)
            parts.append(torch.where(replaced, self.repl_dir, kd)[:, None])
        if self.mode == "mode3":
            parts.append(torch.nn.functional.one_hot(mode, 3).float())
        else:
            parts.append((mode == KICK_MODE_KICK).float()[:, None])
        out = torch.cat(parts, dim=-1)
        self._cached = out.clone()
        return out


__all__ = [
    "KICK_MODE_DRIBBLE",
    "KICK_MODE_IGNORE",
    "KICK_MODE_KICK",
    "StudentCommandObsTerm",
    "frame_field",
]
