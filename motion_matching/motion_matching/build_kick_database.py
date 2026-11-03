"""Build a kick skill database plus its ``.kicks.json`` sidecar from annotated clips.

Usage::

    python -m motion_matching.build_kick_database --robot lupus --family kick_run \\
        path/to/kicks/            # every *.kick.json in it, or explicit json files

Writes ``database_<robot>_<family>.bin`` and ``database_<robot>_<family>.kicks.json``
under ``PHP_MOTION_DATABASE_DIR``. Each clip is baked unmirrored and mirrored (the
mirrored copy kicks with the other foot), so one set of right-foot clips serves
both ``kick_right`` and ``kick_left``. See :mod:`motion_matching.io.kick_io` for
the annotation format.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Sequence

from motion_matching.generate_database import build_database
from motion_matching.io.kick_io import (
    KickClipAnnotation,
    KickSidecar,
    kick_ranges_from_entries,
    load_ball_radius,
    load_kick_annotations,
    sidecar_path_for,
)
from motion_matching.kinematics.robot_spec import DEFAULT_ROBOT, ROBOT_SPECS, get_robot_spec
from motion_matching.utils import db_path


def collect_annotations(paths: Sequence[os.PathLike]) -> list[tuple[Path, KickClipAnnotation]]:
    out: list[tuple[Path, KickClipAnnotation]] = []
    for p in paths:
        p = Path(p)
        files = sorted(p.glob("*.kick.json")) if p.is_dir() else [p]
        for f in files:
            out.extend(load_kick_annotations(f))
    if not out:
        raise FileNotFoundError(f"no kick annotations found in {list(map(str, paths))}")
    return out


def build_kick_database(
    annotations: Sequence[tuple[Path, KickClipAnnotation]],
    *,
    robot=None,
    output: str,
    mirror: bool = True,
    ball_radius: Optional[float] = None,
) -> KickSidecar:
    """Bake ``annotations`` into ``output`` (.bin) and write the sidecar next to it."""
    spec = get_robot_spec(robot)
    for clip_path, _ in annotations:
        if not clip_path.is_file():
            raise FileNotFoundError(f"kick clip not found: {clip_path}")
    files = [[str(clip), mirror, ann.speed_up, ann.fps, None] for clip, ann in annotations]
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    entries = build_database(files, robot=spec, output=output)
    radius = load_ball_radius() if ball_radius is None else float(ball_radius)
    sidecar = KickSidecar(
        robot=spec.name,
        ranges=kick_ranges_from_entries(annotations, entries, ball_radius=radius),
        ball_radius=radius,
    )
    sidecar.save(sidecar_path_for(output))
    return sidecar


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("annotations", nargs="+", help="*.kick.json files or directories containing them")
    parser.add_argument("--robot", default=DEFAULT_ROBOT, choices=sorted(ROBOT_SPECS))
    parser.add_argument("--family", default="kick_run", help="database family, e.g. kick_run or kick_stand")
    parser.add_argument("--no-mirror", action="store_true", help="do not add mirrored copies")
    args = parser.parse_args(argv)

    output = db_path(f"database_{args.robot}_{args.family}.bin")
    sidecar = build_kick_database(
        collect_annotations(args.annotations), robot=args.robot, output=output, mirror=not args.no_mirror
    )
    print(f"Wrote {output} with {len(sidecar.ranges)} kick ranges")


if __name__ == "__main__":
    main()
