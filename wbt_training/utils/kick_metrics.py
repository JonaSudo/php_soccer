"""Kick evaluation metrics and the PHP-style evaluation grid (plan, Phase 5).

Simulator-agnostic: :class:`KickEpisodeTracker` takes per-step tensors (from
IsaacSim through :class:`KickMetricsEvalCallback`, or from the MuJoCo Sim2Sim
runner) and writes one record per finished episode. :func:`summarize` turns
records into the metrics of the plan; :class:`KickEvalGrid` bins them like
PHP's Table I (run-up distance x lateral ball offset x speed).

Per episode:

* ``intended``: the reference kicks in this episode (``kick_phase`` reaches 1)
* ``kicked``: first robot-ball contact while ``kick_phase`` >= 1
* ``ball_speed``: peak horizontal ball speed within ``window_s`` after that contact
* ``dir_err_deg``: signed angle between the ball direction at that peak and
  the target direction (heading at contact, plus ``kick_dir`` in Phase 6)
* ``success``: ``intended`` and ``kicked`` and speed >= ``min_kick_speed`` and
  ``|dir_err_deg|`` <= ``max_dir_err_deg`` (plan: +-15 deg)
* ``miskick``: robot-ball contact while no kick is under way (``kick_phase`` 0
  before the kick; covers stand-before-trigger, off-path and no-ball)
* ``fell``: base below ``fall_height`` or tilted more than 60 deg at some step;
  ``terminated``: the episode ended by a termination other than timeout
* ``v_err`` / ``w_err``: RMS error of base forward speed and yaw rate against
  the commanded (v, omega) while walking (``kick_phase`` 0)
* ``approach``, ``lateral``, ``speed``: ball x / |y| in the robot heading frame
  at the episode start (|y| to the nearer foot if ``foot_offset_y`` > 0) and the mean commanded v before the kick (grid axes)
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

STEP_KEYS = (
    "ball_pos_w",  # [N, 3]
    "ball_vel_w",  # [N, 3]
    "base_pos_w",  # [N, 3]
    "base_yaw",  # [N]
    "base_up_z",  # [N] z of the base up-axis in world (1 = upright)
    "base_fwd_vel",  # [N] base velocity along the heading
    "base_yaw_rate",  # [N]
    "cmd_v",  # [N]
    "cmd_w",  # [N]
    "kick_phase",  # [N]
    "contact",  # [N] bool robot-ball contact this step
    "target_heading",  # [N] direction the kick should go (rad), NaN if unknown
)


@dataclass(frozen=True)
class KickMetricsCfg:
    dt: float = 0.02
    window_s: float = 0.5
    min_kick_speed: float = 0.5
    max_dir_err_deg: float = 15.0
    fall_height: float = 0.30
    fall_up_z: float = 0.5  # cos(60 deg)
    # Lateral ball offset is measured to the nearer kick foot at +-foot_offset_y
    # (LUPUS: 0.133 m); 0 measures it to the base centre line.
    foot_offset_y: float = 0.0


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


class KickEpisodeTracker:
    """Accumulates per-env episode statistics; ``records`` holds finished episodes."""

    def __init__(self, num_envs: int, cfg: KickMetricsCfg | None = None, device: str | torch.device = "cpu"):
        self.cfg = cfg or KickMetricsCfg()
        self.n = num_envs
        self.device = torch.device(device)
        self.records: list[dict[str, Any]] = []
        self._alloc()

    def _alloc(self) -> None:
        n, d = self.n, self.device
        z = lambda dtype=torch.float32: torch.zeros(n, dtype=dtype, device=d)  # noqa: E731
        self.started = z(torch.bool)
        self.steps = z(torch.long)
        self.approach, self.lateral = z(), z()
        self.intended, self.kicked, self.miskick, self.fell = z(torch.bool), z(torch.bool), z(torch.bool), z(torch.bool)
        self.after_kick = z(torch.bool)  # kick contact seen
        self.kick_started = z(torch.bool)  # kick_phase >= 1 seen
        self.steps_since_kick = z(torch.long)
        self.peak_speed, self.peak_dir, self.target = z(), z(), torch.full((n,), float("nan"), device=d)
        self.v_sum, self.v_sq, self.w_sq, self.walk_steps = z(), z(), z(), z()
        self.miskick_count = z(torch.long)
        self.pre_steps = z()

    def _clear(self, ids: torch.Tensor) -> None:
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor) and value.shape[:1] == (self.n,):
                value[ids] = float("nan") if name == "target" else 0

    def step(self, s: dict[str, torch.Tensor]) -> None:
        """Feed one control step (state *after* the physics step, before resets)."""
        c = self.cfg
        f = lambda k: torch.as_tensor(s[k], device=self.device)  # noqa: E731
        ball_p, ball_v, base_p = f("ball_pos_w").float(), f("ball_vel_w").float(), f("base_pos_w").float()
        yaw = f("base_yaw").float()
        phase = f("kick_phase").long()
        contact = f("contact").bool()

        new = ~self.started
        if new.any():
            d = ball_p[:, :2] - base_p[:, :2]
            cy, sy = torch.cos(yaw), torch.sin(yaw)
            x = cy * d[:, 0] + sy * d[:, 1]
            y = -sy * d[:, 0] + cy * d[:, 1]
            self.approach = torch.where(new, x, self.approach)
            fo = self.cfg.foot_offset_y
            lat = torch.minimum((y - fo).abs(), (y + fo).abs()) if fo > 0 else y.abs()
            self.lateral = torch.where(new, lat, self.lateral)
            self.started |= new
        self.steps += 1

        kicking = phase >= 1
        self.intended |= kicking
        self.kick_started |= kicking
        idle = (phase == 0) & ~self.kick_started
        self.miskick_count += (contact & idle).long()
        self.miskick |= contact & idle

        first_kick = contact & kicking & ~self.after_kick
        if first_kick.any():
            th = f("target_heading").float()
            th = torch.where(torch.isnan(th), yaw, th)
            self.target = torch.where(first_kick, th, self.target)
            self.after_kick |= first_kick
            self.kicked |= first_kick
        in_window = self.after_kick & (self.steps_since_kick.float() * c.dt <= c.window_s)
        speed = ball_v[:, :2].norm(dim=-1)
        better = in_window & (speed > self.peak_speed)
        self.peak_speed = torch.where(better, speed, self.peak_speed)
        self.peak_dir = torch.where(better, torch.atan2(ball_v[:, 1], ball_v[:, 0]), self.peak_dir)
        self.steps_since_kick += self.after_kick.long()

        self.fell |= (base_p[:, 2] < c.fall_height) | (f("base_up_z").float() < c.fall_up_z)

        walk = phase == 0
        ev = f("base_fwd_vel").float() - f("cmd_v").float()
        ew = f("base_yaw_rate").float() - f("cmd_w").float()
        self.v_sq += torch.where(walk, ev * ev, torch.zeros_like(ev))
        self.w_sq += torch.where(walk, ew * ew, torch.zeros_like(ew))
        pre = walk & ~self.kick_started
        self.v_sum += torch.where(pre, f("cmd_v").float(), torch.zeros_like(ev))
        self.walk_steps += walk.long()
        self.pre_steps += pre.float()

    def finish(self, env_ids: torch.Tensor | Sequence[int], terminated: torch.Tensor | Sequence[bool] | None = None) -> None:
        """Close the episodes of ``env_ids`` (after their done flag) and clear them."""
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        if ids.numel() == 0:
            return
        term = (
            torch.zeros(ids.numel(), dtype=torch.bool)
            if terminated is None
            else torch.as_tensor(terminated, dtype=torch.bool).flatten().cpu()
        )
        c = self.cfg
        pre = self.pre_steps
        for k, i in enumerate(ids.tolist()):
            if not bool(self.started[i]):
                continue
            kicked = bool(self.kicked[i])
            err = float(torch.rad2deg(_wrap(self.peak_dir[i] - self.target[i]))) if kicked else float("nan")
            speed = float(self.peak_speed[i]) if kicked else 0.0
            walk = max(int(self.walk_steps[i]), 1)
            rec = {
                "env_id": i,
                "steps": int(self.steps[i]),
                "approach": float(self.approach[i]),
                "lateral": float(self.lateral[i]),
                "speed": float(self.v_sum[i] / max(float(pre[i]), 1.0)),
                "intended": bool(self.intended[i]),
                "kicked": kicked,
                "ball_speed": speed,
                "dir_err_deg": err,
                "success": bool(self.intended[i])
                and kicked
                and speed >= c.min_kick_speed
                and abs(err) <= c.max_dir_err_deg,
                "miskick": bool(self.miskick[i]),
                "miskick_contacts": int(self.miskick_count[i]),
                "fell": bool(self.fell[i]),
                "terminated": bool(term[k]),
                "v_err": math.sqrt(float(self.v_sq[i]) / walk),
                "w_err": math.sqrt(float(self.w_sq[i]) / walk),
            }
            self.records.append(rec)
        self._clear(ids)


def _mean(xs: Iterable[float]) -> float:
    xs = [x for x in xs if x == x]  # drop NaN
    return float(sum(xs) / len(xs)) if xs else float("nan")


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Plan metrics over a set of episode records."""
    kick = [r for r in records if r["intended"]]
    no_kick = [r for r in records if not r["intended"]]
    kicked = [r for r in kick if r["kicked"]]
    return {
        "episodes": float(len(records)),
        "kick_episodes": float(len(kick)),
        "kick_success_rate": _mean(float(r["success"]) for r in kick),
        "kick_rate": _mean(float(r["kicked"]) for r in kick),
        "dir_err_deg_mean_abs": _mean(abs(r["dir_err_deg"]) for r in kicked),
        "dir_err_deg_mean": _mean(r["dir_err_deg"] for r in kicked),
        "ball_speed_mean": _mean(r["ball_speed"] for r in kicked),
        "miskick_rate": _mean(float(r["miskick"]) for r in records),
        "miskick_rate_no_kick_episodes": _mean(float(r["miskick"]) for r in no_kick),
        "fall_rate": _mean(float(r["fell"]) for r in records),
        "termination_rate": _mean(float(r["terminated"]) for r in records),
        "v_err_rms": _mean(r["v_err"] for r in records),
        "w_err_rms": _mean(r["w_err"] for r in records),
    }


@dataclass(frozen=True)
class KickEvalGrid:
    """PHP Table I style grid: run-up distance x lateral offset x speed, N episodes per cell.

    Bin edges are the midpoints between the listed values; a record falls into
    the nearest value on every axis.
    """

    approach: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0)  # m, ball ahead at episode start
    lateral: tuple[float, ...] = (0.0, 0.05, 0.10, 0.25, 0.5)  # m, |y| of the ball to the nearer foot
    speed: tuple[float, ...] = (0.0, 0.3, 0.6, 0.9)  # m/s, commanded v before the kick
    episodes_per_cell: int = 100

    @staticmethod
    def _nearest(values: Sequence[float], x: float) -> float:
        return min(values, key=lambda v: abs(v - x))

    def cell_of(self, rec: dict[str, Any]) -> tuple[float, float, float]:
        return (
            self._nearest(self.approach, rec["approach"]),
            self._nearest(self.lateral, rec["lateral"]),
            self._nearest(self.speed, rec["speed"]),
        )

    def cells(self) -> list[tuple[float, float, float]]:
        return [(a, l, s) for a in self.approach for l in self.lateral for s in self.speed]

    def scenario_specs(self) -> list[dict[str, float]]:
        """One spec per cell for generating a dedicated eval motion set (Phase 2 generator)."""
        return [
            {"approach_distance": a, "lateral_offset": l, "speed": s, "episodes": self.episodes_per_cell}
            for a, l, s in self.cells()
        ]

    def table(self, records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        groups: dict[tuple[float, float, float], list[dict[str, Any]]] = {}
        for r in records:
            groups.setdefault(self.cell_of(r), []).append(r)
        rows = []
        for cell in self.cells():
            recs = groups.get(cell, [])
            row = {"approach": cell[0], "lateral": cell[1], "speed": cell[2], "n": len(recs)}
            row.update({k: v for k, v in summarize(recs).items() if k != "episodes"} if recs else {})
            row["complete"] = len(recs) >= self.episodes_per_cell
            rows.append(row)
        return rows


def table_to_markdown(rows: Sequence[dict[str, Any]], cols: Sequence[str] = ("kick_success_rate", "dir_err_deg_mean_abs", "ball_speed_mean", "miskick_rate", "fall_rate")) -> str:
    head = ["approach", "lateral", "speed", "n", *cols]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        if r["n"] == 0:
            continue
        cells = [f"{r['approach']:.2f}", f"{r['lateral']:.2f}", f"{r['speed']:.2f}", str(r["n"])]
        cells += [f"{r.get(c, float('nan')):.3f}" for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(path: str, records: Sequence[dict[str, Any]], grid: KickEvalGrid | None = None, extra: dict | None = None) -> dict:
    grid = grid or KickEvalGrid()
    report = {
        "summary": summarize(records),
        "grid": grid.table(records),
        "grid_cfg": asdict(grid),
        "records": list(records),
        **(extra or {}),
    }
    with open(path, "w") as fh:
        json.dump(report, fh, indent=1, default=float)
    return report


__all__ = [
    "KickEpisodeTracker",
    "KickEvalGrid",
    "KickMetricsCfg",
    "STEP_KEYS",
    "summarize",
    "table_to_markdown",
    "write_report",
]
