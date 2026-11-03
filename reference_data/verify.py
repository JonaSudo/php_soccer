"""Acceptance check of exported clips (Phase 1 "Abnahme").

    python -m reference_data.verify reference_out/           # all clips in the folder
    python -m reference_data.verify reference_out/stand_kick/x_0.7x.npz

Per clip:

1. **Ground**: no robot collision geom lower than ``--max-penetration``
   (default 1 cm) in any frame, replayed in MuJoCo with ``lupus.xml``.
2. **Kick**: at the annotated contact frame the kick foot box touches the
   annotated ball (gap between ``--min-gap`` and ``--max-gap``, default
   -2..+2 cm), using the ball the npz carries (``object_pos_w``).
3. **Format**: qpos is (T, 29), unit root quaternions, finite values, the
   DoFs within the joint stops.

Exit code 1 if any clip fails, so it can gate a batch run.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from reference_data.clip_ops import LupusClip
from reference_data.lupus_kin import box_point_distance, kinematics


@dataclass
class Result:
    name: str
    ok: bool
    lowest: float
    contact_gap: float | None
    problems: list[str] = field(default_factory=list)


def verify_clip(
    npz: Path | str,
    *,
    max_penetration: float = 0.01,
    min_gap: float = -0.02,
    max_gap: float = 0.02,
    stop_margin: float = 0.02,
) -> Result:
    npz = Path(npz)
    clip = LupusClip.load(npz)
    kin = kinematics()
    problems = []

    q = clip.qpos
    if not np.all(np.isfinite(q)):
        problems.append("non-finite qpos")
    if not np.allclose(np.linalg.norm(q[:, :4], axis=1), 1.0, atol=1e-3):
        problems.append("root quaternion not normalised")
    exc = np.maximum(kin.lower - q[:, 7:], q[:, 7:] - kin.upper).max()
    if exc > stop_margin:
        problems.append(f"joint beyond its stop by {exc:.3f} rad")

    fk = kin.clip_fk(q)
    lowest = float(fk["lowest"].min())
    if lowest < -max_penetration:
        frame = int(np.argmin(fk["lowest"]))
        problems.append(f"ground penetration {-lowest * 100:.1f} cm at frame {frame}")

    gap = None
    js = npz.with_suffix(".json")
    ann = json.loads(js.read_text()).get("annotation") if js.exists() else None
    if ann is not None:
        if clip.ball_pos is None:
            problems.append("kick clip without object_pos_w")
        else:
            t = int(ann["contact_frame"])
            side = ann["kick_foot"]
            ball = clip.ball_pos[t]
            gap = box_point_distance(
                fk[f"{side}_center"][t], fk[f"{side}_rot"][t], kin.feet[side].half, ball
            ) - float(ann["ball_radius"])
            if not (min_gap <= gap <= max_gap):
                problems.append(f"kick foot misses the ball by {gap * 100:.1f} cm at frame {t}")
            if not np.allclose(ball, ann["ball_pos_contact"], atol=1e-3):
                problems.append("object_pos_w at contact differs from the annotated ball point")
    return Result(npz.stem, not problems, lowest, gap, problems)


def collect(paths: list[Path]) -> list[Path]:
    out = []
    for p in paths:
        out.extend(sorted(p.rglob("*.npz")) if p.is_dir() else [p])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", type=Path, nargs="+")
    ap.add_argument("--max-penetration", type=float, default=0.01)
    ap.add_argument("--min-gap", type=float, default=-0.02)
    ap.add_argument("--max-gap", type=float, default=0.02)
    ap.add_argument("--json", type=Path, help="write the results here")
    args = ap.parse_args(argv)
    results = [
        verify_clip(p, max_penetration=args.max_penetration, min_gap=args.min_gap, max_gap=args.max_gap)
        for p in collect(args.paths)
    ]
    for r in results:
        gap = "" if r.contact_gap is None else f" gap {r.contact_gap * 100:+.1f} cm"
        print(f"{'OK  ' if r.ok else 'FAIL'} {r.name}: lowest {r.lowest * 100:+.1f} cm{gap}"
              + ("" if r.ok else "  <- " + "; ".join(r.problems)))
    if args.json:
        args.json.write_text(json.dumps([r.__dict__ for r in results], indent=2))
    n_bad = sum(not r.ok for r in results)
    print(f"{len(results) - n_bad}/{len(results)} clips pass")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
