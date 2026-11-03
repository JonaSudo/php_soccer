"""Retargeted LUPUS clip -> PHP source clips with annotations (Phase 1, steps 3-7).

    python -m reference_data.postprocess clip.lupus_raw.npz --group stand_kick \\
        --speed auto --mirror --out reference_out/

Per clip: trim, put on the ground, time scaling (fixed factor or the fastest
feasible one), feasibility check, mirroring, kick annotation and export.
Every variant becomes

    <out>/<group>/<id>[_mirror]_<speed>x.npz    qpos (T, 29) PHP order, fps,
                                                object_pos_w (kicks), ball_radius
    <out>/<group>/<id>[_mirror]_<speed>x.json   annotation + feasibility + provenance

The npz is the motion-matching source format (``generate_database --robot
lupus --fps <fps>``); qpos is ``[root_quat_wxyz, root_pos, dofs]`` in URDF
joint order. The speed tag follows PHP's naming (``dash_vault_76_0.75x``).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from reference_data import feasibility
from reference_data.annotate import KickParams, annotate, ball_track, mirror_fields, scale_frames
from reference_data.clip_ops import LupusClip

GROUPS = ("walk", "walk_kick", "stand_kick")
KICK_GROUPS = ("walk_kick", "stand_kick")
# Plan, step 3: a 1.75 m human kicks too fast for a K1-sized robot.
DEFAULT_SPEEDS = (0.8, 0.75, 0.7, 0.65, 0.6)


@dataclass
class ClipSpec:
    id: str
    group: str
    trim: tuple[int, int | None] | None = None   # frames of the retargeted clip
    speed: float | str = "auto"                   # factor or "auto"
    speeds: tuple[float, ...] = DEFAULT_SPEEDS    # candidates for "auto"
    mirror: bool = True
    out_fps: float | None = None                  # default: keep
    annotation: dict = field(default_factory=dict)  # overrides, frames of the trimmed unscaled clip
    notes: str = ""

    def __post_init__(self) -> None:
        if self.group not in GROUPS:
            raise ValueError(f"group must be one of {GROUPS}, got {self.group!r}")


@dataclass
class Exported:
    name: str
    npz: Path
    json: Path
    accepted: bool
    reasons: list[str]


def _tag(speed: float) -> str:
    return f"{speed:.2f}".rstrip("0").rstrip(".") + "x"


def process(
    raw: LupusClip,
    spec: ClipSpec,
    out_dir: Path | str,
    *,
    thresholds: feasibility.Thresholds | None = None,
    kick_params: KickParams | None = None,
) -> list[Exported]:
    out_dir = Path(out_dir) / spec.group
    out_dir.mkdir(parents=True, exist_ok=True)
    clip = raw
    if spec.trim:
        clip = clip.trim(*spec.trim)
    clip = clip.place_on_ground()

    reasons: list[str] = []
    if spec.speed == "auto":
        speed, report = feasibility.slowest_feasible_scale(clip, thresholds, tuple(spec.speeds))
        if speed is None:
            speed = min(spec.speeds)
            reasons.append(f"infeasible even at {speed}x: {', '.join(report.failing)}")
    else:
        speed = float(spec.speed)
        report = feasibility.check(clip.time_scale(speed), thresholds)
        if not report.ok:
            reasons.append(f"infeasible at {speed}x: {', '.join(report.failing)}")
    scaled = clip.time_scale(speed, spec.out_fps)
    fps_ratio = scaled.fps / clip.fps

    variants = [(False, scaled)]
    if spec.mirror:
        variants.append((True, scaled.mirror()))

    exported = []
    for mirrored, var in variants:
        name = f"{spec.id}{'_mirror' if mirrored else ''}_{_tag(speed)}"
        var_reasons = list(reasons)
        record: dict = {
            "name": name,
            "id": spec.id,
            "group": spec.group,
            "speed": speed,
            "mirrored": mirrored,
            "fps": var.fps,
            "num_frames": var.num_frames,
            "source": raw.meta.get("source", ""),
            "retarget_meta": raw.meta,
            "trim": list(spec.trim) if spec.trim else None,
            "feasibility": report.to_dict(),
            "notes": spec.notes,
        }
        extra = {}
        if spec.group in KICK_GROUPS:
            # hand annotations refer to the trimmed clip before time scaling
            ov = scale_frames(spec.annotation, speed / fps_ratio)
            if mirrored:
                ov = mirror_fields(ov)
            ann = annotate(var, kick_params, ov)
            record["annotation"] = ann.to_dict()
            if ann.group != spec.group:
                var_reasons.append(f"annotated as {ann.group} ({ann.reason or 'detector'}), catalog says {spec.group}")
            var.ball_pos = ball_track(var, ann)
            var.ball_radius = ann.ball_radius
            extra["contact_frame"] = ann.contact_frame
        else:
            var.ball_pos = None
        record["accepted"] = not var_reasons
        record["reasons"] = var_reasons
        npz = var.save(out_dir / f"{name}.npz", extra)
        js = out_dir / f"{name}.json"
        js.write_text(json.dumps(record, indent=2, default=_json_default))
        exported.append(Exported(name, npz, js, not var_reasons, var_reasons))
    return exported


def _json_default(o):
    import numpy as np

    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def write_manifest(out_dir: Path | str, exported: list[Exported]) -> Path:
    out_dir = Path(out_dir)
    path = out_dir / "manifest.json"
    rows = [
        {"name": e.name, "npz": str(e.npz.relative_to(out_dir)), "json": str(e.json.relative_to(out_dir)),
         "accepted": e.accepted, "reasons": e.reasons}
        for e in exported
    ]
    path.write_text(json.dumps(rows, indent=2))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw", type=Path, help="*.lupus_raw.npz from retarget_lupus")
    ap.add_argument("--id", help="clip id (default: file stem)")
    ap.add_argument("--group", choices=GROUPS, required=True)
    ap.add_argument("--speed", default="auto", help="factor (0.7) or 'auto'")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--trim", type=int, nargs=2)
    ap.add_argument("--out-fps", type=float)
    ap.add_argument("--annotation", type=json.loads, default={}, help='overrides, e.g. \'{"contact_frame": 52}\'')
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    raw = LupusClip.from_retarget(args.raw)
    spec = ClipSpec(
        id=args.id or raw.name,
        group=args.group,
        trim=tuple(args.trim) if args.trim else None,
        speed=args.speed if args.speed == "auto" else float(args.speed),
        mirror=not args.no_mirror,
        out_fps=args.out_fps,
        annotation=args.annotation,
    )
    for e in process(raw, spec, args.out):
        status = "ok" if e.accepted else "REJECT: " + "; ".join(e.reasons)
        print(f"{e.npz}  {status}")


if __name__ == "__main__":
    main()
