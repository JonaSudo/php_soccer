"""Phase-2 acceptance checks for a soccer training bundle.

The plan's acceptance criterion for Phase 2 reads: *in the visualizer, the
ball lies at the foot in the contact frame for random samples, and the
run-ups cover both standing legs and different step phases.* This module
turns that into numbers over a whole bundle (every motion, not just samples):

* **Ball at the kick foot**: horizontal distance between the ball centre and
  the kick foot body in the contact frame (``max_ball_foot_distance``), the
  ball rests on the ground (z = radius) and does not move before contact.
* **Both kick feet / standing legs**: per kick scenario, each foot kicks in
  at least ``min_side_share`` of the motions. The standing leg is the other
  foot; it is reported as well, measured as the lower foot at kick entry.
* **Step phases at kick entry**: gait phase (0..1, one full stride) at the
  first commit frame (``kick_phase`` 1) of every run-up kick, from the
  fore-aft offset of the two feet in the heading frame. The phases must fill
  at least ``min_phase_bins`` of ``phase_bins`` bins.
* **Negatives**: no contact frame, off-path balls 0.25-1.0 m from the path,
  parked balls >= 5 m away, ``outside_zone`` balls outside the kick zone.
* **Flat terrain**: every ``*_terrain.npy`` has zero boxes and every
  ``*_terrain.obj`` is only the ground plane (top at z = 0).
* **Kick fields** pass :func:`motion_matching.io.kick_io.validate_kick_npz`.

Use :func:`check_bundle` (returns a report dict) or the ``check``
sub-command of :mod:`motion_matching.build_soccer_bundle`.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from motion_matching.io.kick_io import KICK_PHASE_COMMIT, load_ball_radius, validate_kick_npz
from motion_matching.scenarios import soccer as S

#: Kick scenarios (have a ball at the foot and exactly one contact).
KICK_SCENARIOS = ("walk_kick", "stand_kick", "walk_kick_dir", "stand_kick_dir")
#: Run-up kicks, where the gait phase at entry matters.
RUN_UP_SCENARIOS = ("walk_kick", "walk_kick_dir")
SIDE_NAMES = {0: "right", 1: "left"}


@dataclass
class AcceptanceConfig:
    """Thresholds; all placeholders for a robot of Booster K1 size."""

    # Ball centre to kick-foot body origin (ankle / foot link), horizontal.
    # The foot origin sits behind the toe, so this is foot length/2 + ball radius-ish.
    max_ball_foot_distance: float = 0.25
    min_side_share: float = 0.3
    phase_bins: int = 4
    min_phase_bins: int = 3
    # Below this many run-up kicks the phase spread is reported but not enforced.
    min_run_ups_for_phase_check: int = 8
    position_tol: float = 1e-4
    path_tol: float = 0.05


@dataclass
class AcceptanceReport:
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    kick_sides: dict[str, dict[str, int]] = field(default_factory=dict)
    stance_legs: dict[str, dict[str, int]] = field(default_factory=dict)
    ball_foot_distance: dict[str, float] = field(default_factory=dict)
    entry_phase_hist: list[int] = field(default_factory=list)
    entry_phases: list[float] = field(default_factory=list)
    flat_terrain: bool = True
    motions_checked: int = 0

    def fail(self, msg: str) -> None:
        self.passed = False
        self.errors.append(msg)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["errors"] = d["errors"][:200]
        return d

    def summary(self) -> str:
        lines = [f"Phase-2 acceptance: {'PASSED' if self.passed else 'FAILED'} ({self.motions_checked} motions)"]
        lines.append("  counts: " + ", ".join(f"{k}={v}" for k, v in sorted(self.counts.items())))
        for sc, sides in sorted(self.kick_sides.items()):
            stance = self.stance_legs.get(sc, {})
            lines.append(f"  {sc}: kick foot {dict(sides)}, stance leg at entry {dict(stance)}")
        if self.ball_foot_distance:
            b = self.ball_foot_distance
            lines.append(f"  ball-foot distance at contact: median {b['median']:.3f} m, max {b['max']:.3f} m")
        if self.entry_phase_hist:
            lines.append(f"  step phase at kick entry (bins of 1/{len(self.entry_phase_hist)} stride): {self.entry_phase_hist}")
        lines.append(f"  flat terrain: {self.flat_terrain}")
        lines += [f"  WARNING {w}" for w in self.warnings[:20]]
        lines += [f"  ERROR {e}" for e in self.errors[:20]]
        if len(self.errors) > 20:
            lines.append(f"  ... {len(self.errors) - 20} more errors")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------


def foot_body_index(body_names: Any, side: str) -> int:
    """Index of the ``side`` foot body (``*_foot_link`` for LUPUS, ``*_ankle_roll_link`` for G1)."""
    names = [str(n) for n in body_names]
    for wanted in (f"{side}_foot_link", f"{side}_ankle_roll_link"):
        if wanted in names:
            return names.index(wanted)
    for i, n in enumerate(names):
        if n.startswith(side) and ("foot" in n or "ankle" in n) and "contact_point" not in n:
            return i
    raise KeyError(f"no {side} foot body in {names}")


def _yaw(quat_wxyz: np.ndarray) -> np.ndarray:
    return S._yaw(quat_wxyz)


def foot_forward_offset(body_pos_w: np.ndarray, root_quat_wxyz: np.ndarray, left: int, right: int) -> np.ndarray:
    """Per frame: (left foot - right foot) along the root heading, in m."""
    yaw = _yaw(root_quat_wxyz)
    fwd = np.stack([np.cos(yaw), np.sin(yaw)], axis=-1)
    d = body_pos_w[:, left, :2] - body_pos_w[:, right, :2]
    return np.sum(d * fwd, axis=-1)


def gait_phase_at(offset: np.ndarray, k: int, min_amplitude: float = 0.02) -> Optional[float]:
    """Gait phase in [0, 1) at frame ``k`` from the fore-aft foot offset signal.

    0 = the left foot just passed the right one (offset crosses 0 upwards),
    0.5 = the right foot just passed the left one. Linear between the two
    sign changes around ``k``, the half period taken from the previous
    crossings. ``None`` when the robot was not walking before ``k`` (fewer
    than two crossings, or tiny steps), as for a kick from standing.
    """
    x = np.asarray(offset[: k + 1], dtype=np.float64)
    if len(x) < 3 or np.ptp(x) < 2 * min_amplitude:
        return None
    # Hysteresis: only count a sign change once |x| exceeded min_amplitude.
    sign = np.zeros(len(x), dtype=np.int8)
    state = 0
    for i, v in enumerate(x):
        if v > min_amplitude:
            state = 1
        elif v < -min_amplitude:
            state = -1
        sign[i] = state
    changes = [i for i in range(1, len(sign)) if sign[i] != sign[i - 1] and sign[i - 1] != 0]
    if len(changes) < 2:
        return None
    half = float(np.median(np.diff(changes[-4:])))
    if half <= 0:
        return None
    frac = min((k - changes[-1]) / half, 0.999)
    return float((0.0 if sign[-1] > 0 else 0.5) + 0.5 * frac)


# --------------------------------------------------------------------------
# Terrain
# --------------------------------------------------------------------------


def terrain_is_flat(npy: Optional[Path], obj: Optional[Path], tol: float = 1e-4) -> tuple[bool, str]:
    if npy is not None:
        boxes = np.load(npy, allow_pickle=False)
        if boxes.ndim != 3 or boxes.shape[0] != 0:
            return False, f"{npy.name}: {boxes.shape[0] if boxes.ndim else '?'} terrain boxes, expected 0"
    if obj is not None:
        top = -np.inf
        with open(obj) as fh:
            for line in fh:
                if line.startswith("v "):
                    top = max(top, float(line.split()[3]))
        if top > tol:
            return False, f"{obj.name}: geometry above the ground plane (max z {top:.3f} m)"
    return True, ""


# --------------------------------------------------------------------------
# Bundle check
# --------------------------------------------------------------------------


def _scenario_of(path: Path, data: dict, manifest_entry: Optional[dict]) -> str:
    if manifest_entry is not None:
        return manifest_entry["scenario"]
    if "scenario" in data:
        return str(data["scenario"])
    return path.parent.name


def check_motion(
    path: Path,
    report: AcceptanceReport,
    cfg: AcceptanceConfig,
    entry: Optional[dict],
    ball_radius: float,
    collect: dict,
) -> None:
    with np.load(path, allow_pickle=False) as npz:
        data = {k: npz[k] for k in npz.files}
    name = path.name
    scenario = _scenario_of(path, data, entry)
    report.counts[scenario] = report.counts.get(scenario, 0) + 1
    try:
        validate_kick_npz(data)
    except ValueError as e:
        report.fail(f"{name}: {e}")
        return
    contacts = np.flatnonzero(data["contact_frame"])
    ball = data["object_pos_w"]
    body_pos = data["body_pos_w"]
    root_quat = data["body_quat_w"][:, 0]
    root_xy = body_pos[:, 0, :2]
    body_names = data["body_names"]
    li, ri = foot_body_index(body_names, "left"), foot_body_index(body_names, "right")

    if scenario in KICK_SCENARIOS:
        if len(contacts) != 1:
            report.fail(f"{name}: {len(contacts)} contact frames, expected 1")
            return
        c = int(contacts[0])
        side_code = int(np.asarray(data["kick_side"]).reshape(-1)[0]) if "kick_side" in data else -1
        side = SIDE_NAMES.get(side_code) or (entry or {}).get("kick_side")
        if side not in ("left", "right"):
            report.fail(f"{name}: unknown kick side {side_code}")
            return
        collect["sides"].setdefault(scenario, Counter())[side] += 1
        foot = li if side == "left" else ri
        dist = float(np.linalg.norm(body_pos[c, foot, :2] - ball[c, :2]))
        collect["dist"].append(dist)
        if dist > cfg.max_ball_foot_distance:
            report.fail(f"{name}: ball {dist:.3f} m from the {side} foot at contact (max {cfg.max_ball_foot_distance})")
        if abs(float(ball[0, 2]) - ball_radius) > cfg.position_tol:
            report.fail(f"{name}: ball z {float(ball[0, 2]):.4f} != radius {ball_radius:.4f}")
        if np.abs(ball[: c + 1] - ball[0]).max() > cfg.position_tol:
            report.fail(f"{name}: ball moves before contact")
        commit = np.flatnonzero(data["kick_phase"] == KICK_PHASE_COMMIT)
        if not len(commit):
            report.fail(f"{name}: no commit phase before contact")
            return
        k = int(commit[0])
        stance = "left" if body_pos[k, li, 2] < body_pos[k, ri, 2] else "right"
        collect["stance"].setdefault(scenario, Counter())[stance] += 1
        if scenario in RUN_UP_SCENARIOS:
            offset = foot_forward_offset(body_pos, root_quat, li, ri)
            phase = gait_phase_at(offset, k)
            if phase is None:
                report.warnings.append(f"{name}: no gait before kick entry (frame {k})")
            else:
                collect["phases"].append(phase)
        if scenario.startswith("stand_kick"):
            ke = data["kick_enable"]
            if ke[0] > 0.5:
                report.fail(f"{name}: stand kick starts with kick_enable 1 (no rising edge)")
            elif k < int(np.argmax(ke > 0.5)) - 1:
                report.fail(f"{name}: commit at frame {k} before the kick_enable edge")
    else:
        if len(contacts):
            report.fail(f"{name}: negative scenario {scenario} has a contact frame")
        if np.ptp(ball, axis=0).max() > cfg.position_tol:
            report.fail(f"{name}: ball moves in a negative scenario")
        d = float(S.path_distance(ball[0, :2], root_xy)[0])
        mode = (entry or {}).get("ball_mode")
        if scenario == "walk_ball_offpath" or mode == "offpath":
            if not (S.IGNORE_LATERAL_MIN - cfg.path_tol <= d <= S.IGNORE_LATERAL_MAX + cfg.path_tol):
                report.fail(f"{name}: off-path ball {d:.3f} m from the path")
        elif scenario == "walk_no_ball" or mode == "far":
            if d < S.FAR_BALL_MIN_DISTANCE - cfg.path_tol:
                report.fail(f"{name}: parked ball only {d:.2f} m from the path")
        elif scenario == "stand_ball_outside_zone" or mode == "outside_zone":
            for side, foot in (("left", li), ("right", ri)):
                if S.ball_in_kick_zone(ball[-1, :2], body_pos[-1, foot, :2], root_quat[-1]):
                    report.fail(f"{name}: ball inside the {side} foot's kick zone")
        elif scenario == "walk_dribble":
            if d > 0.1 + cfg.path_tol:
                report.fail(f"{name}: dribble ball {d:.3f} m off the path")


def check_bundle(
    bundle: Path, cfg: Optional[AcceptanceConfig] = None, ball_radius: Optional[float] = None
) -> AcceptanceReport:
    """Run every Phase-2 acceptance check over ``bundle`` (a directory of ``*_motion.npz``)."""
    cfg = cfg or AcceptanceConfig()
    bundle = Path(bundle)
    radius = load_ball_radius() if ball_radius is None else float(ball_radius)
    report = AcceptanceReport()
    manifest_path = bundle / "manifest.json"
    entries: dict[str, dict] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        entries = {m["file"]: m for m in manifest.get("motions", [])}
        radius = float(manifest.get("ball_radius", radius)) if ball_radius is None else radius
    motions = sorted(bundle.rglob("*_motion.npz"))
    if not motions:
        report.fail(f"no *_motion.npz in {bundle}")
        return report
    collect: dict[str, Any] = {"sides": {}, "stance": {}, "dist": [], "phases": []}
    for path in motions:
        rel = path.relative_to(bundle).as_posix()
        check_motion(path, report, cfg, entries.get(rel), radius, collect)
        report.motions_checked += 1
        npy = path.with_name(path.name.replace("_motion.npz", "_terrain.npy"))
        obj = npy.with_suffix(".obj")
        flat, why = terrain_is_flat(npy if npy.is_file() else None, obj if obj.is_file() else None)
        if not flat:
            report.flat_terrain = False
            report.fail(why)

    for scenario, sides in collect["sides"].items():
        report.kick_sides[scenario] = dict(sides)
        total = sum(sides.values())
        for side in ("left", "right"):
            share = sides.get(side, 0) / total
            if total >= 2 and share < cfg.min_side_share:
                report.fail(f"{scenario}: {side} foot kicks in only {share:.0%} of {total} motions")
    for scenario, stance in collect["stance"].items():
        report.stance_legs[scenario] = dict(stance)
    if collect["dist"]:
        dist = np.asarray(collect["dist"])
        report.ball_foot_distance = {
            "median": float(np.median(dist)), "max": float(dist.max()), "min": float(dist.min())
        }
    phases = np.asarray(collect["phases"])
    report.entry_phases = [round(float(p), 3) for p in phases]
    if len(phases):
        hist, _ = np.histogram(phases, bins=cfg.phase_bins, range=(0.0, 1.0))
        report.entry_phase_hist = hist.astype(int).tolist()
        filled = int((hist > 0).sum())
        if len(phases) >= cfg.min_run_ups_for_phase_check and filled < cfg.min_phase_bins:
            report.fail(
                f"run-up kicks enter in only {filled} of {cfg.phase_bins} step phase bins ({report.entry_phase_hist})"
            )
        elif len(phases) < cfg.min_run_ups_for_phase_check:
            report.warnings.append(f"only {len(phases)} run-up kicks with gait; phase spread not enforced")
    return report


__all__ = [
    "AcceptanceConfig",
    "AcceptanceReport",
    "check_bundle",
    "foot_body_index",
    "foot_forward_offset",
    "gait_phase_at",
    "terrain_is_flat",
]
