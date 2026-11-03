"""Synthetic placeholder clips until the Phase 1 reference data exists.

The motions are kinematic stand-ins, not physically sensible: sinusoidal leg
swings on a root that glides over flat ground. They exist so the soccer
pipeline (kick databases, kick skills, scenarios, npz export) runs end to end
today and can be tested without GPU or downloads. Real retargeted clips plug
into exactly the same path: an npz with ``qpos`` plus a ``*.kick.json``
annotation, then ``build_kick_database``.

Usage::

    python -m motion_matching.placeholder_clips --robot g1 --out /tmp/clips --build
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

from motion_matching.kinematics import quat
from motion_matching.kinematics.robot_spec import DEFAULT_ROBOT, ROBOT_SPECS, RobotSpec, get_robot_spec

FPS = 30.0
GAIT_HZ = 1.6
WALK_SPEED = 0.5
HIP_AMPLITUDE = 0.35
KNEE_AMPLITUDE = 0.5
KICK_SWING = 0.9


def _joint(spec: RobotSpec, side: str, part: str) -> int:
    for i, name in enumerate(spec.skeleton.joint_names):
        if name.startswith(side) and part in name:
            return i
    raise KeyError(f"{spec.name}: no {side} {part} joint")


def standing_root_height(spec: RobotSpec) -> float:
    """Root height that puts the lower foot bone on the ground at the zero pose."""
    skel = spec.skeleton
    n = spec.num_dofs
    positions = np.concatenate([np.zeros((1, 3)), skel.joint_bone_vectors], axis=0)[None]
    rotations = quat.from_angle_axis(np.zeros((1, n)), spec.joint_axes.astype(np.float64))
    rotations = np.concatenate([np.array([[[1.0, 0.0, 0.0, 0.0]]]), rotations], axis=1)
    parents = np.concatenate([[-1], skel.joint_parents + 1])
    _, gpos = quat.fk(lrot=rotations, lpos=positions, parents=parents)
    names = [skel.base_link] + list(skel.joint_names)
    feet = [names.index(spec.left_foot_joint), names.index(spec.right_foot_joint)]
    return float(-gpos[0, feet, 2].min()) + 0.05


def _qpos(root_xy: np.ndarray, yaw: np.ndarray, height: float, dof: np.ndarray) -> np.ndarray:
    T = len(dof)
    root_quat = np.stack([np.cos(yaw / 2), np.zeros(T), np.zeros(T), np.sin(yaw / 2)], axis=1)
    root_pos = np.concatenate([root_xy, np.full((T, 1), height)], axis=1)
    return np.concatenate([root_quat, root_pos, dof], axis=1).astype(np.float32)


def _gait(spec: RobotSpec, dof: np.ndarray, t: np.ndarray, amplitude: np.ndarray) -> None:
    phase = 2 * np.pi * GAIT_HZ * t
    for side, sign in (("left", 1.0), ("right", -1.0)):
        dof[:, _joint(spec, side, "hip_pitch")] += amplitude * sign * HIP_AMPLITUDE * np.sin(phase)
        dof[:, _joint(spec, side, "knee")] += amplitude * KNEE_AMPLITUDE * np.maximum(0.0, sign * np.sin(phase))


def _integrate(speed: np.ndarray, yaw_rate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dt = 1.0 / FPS
    yaw = np.cumsum(yaw_rate) * dt
    xy = np.cumsum(np.stack([np.cos(yaw), np.sin(yaw)], axis=1) * speed[:, None], axis=0) * dt
    return xy, yaw


def locomotion_clip(spec: RobotSpec) -> np.ndarray:
    """Stand, walk straight, turn left, turn right, faster walk, stand (about 24 s)."""
    pieces = [(3.0, 0.0, 0.0), (5.0, WALK_SPEED, 0.0), (4.0, WALK_SPEED, 0.6), (4.0, WALK_SPEED, -0.6),
              (4.0, 1.5 * WALK_SPEED, 0.0), (4.0, 0.0, 0.0)]
    speed = np.concatenate([np.full(int(d * FPS), v) for d, v, _ in pieces])
    yaw_rate = np.concatenate([np.full(int(d * FPS), w) for d, _, w in pieces])
    t = np.arange(len(speed)) / FPS
    dof = np.zeros((len(speed), spec.num_dofs))
    _gait(spec, dof, t, np.clip(speed / WALK_SPEED, 0.0, 1.5))
    xy, yaw = _integrate(speed, yaw_rate)
    return _qpos(xy, yaw, standing_root_height(spec), dof)


def kick_clip(spec: RobotSpec, kind: str) -> tuple[np.ndarray, dict]:
    """Right-foot kick, ``kind`` "run" (walk run-up) or "stand". Returns qpos and annotation frames."""
    if kind == "run":
        pre, swing, post = 3.0, 0.8, 2.0
        v_pre, v_post = WALK_SPEED, WALK_SPEED
    else:
        pre, swing, post = 2.0, 0.8, 2.0
        v_pre, v_post = 0.0, 0.0
    n_pre, n_swing, n_post = int(pre * FPS), int(swing * FPS), int(post * FPS)
    T = n_pre + n_swing + n_post
    t = np.arange(T) / FPS
    # Stand kicks still shift the body a little into the kick (and the
    # feature normalisation needs some root motion in every database).
    v_swing = max(0.5 * (v_pre + v_post), 0.15)
    speed = np.concatenate([np.full(n_pre, v_pre), np.full(n_swing, v_swing), np.full(n_post, v_post)])
    dof = np.zeros((T, spec.num_dofs))
    gait_amp = (speed > 0).astype(np.float64)
    gait_amp[n_pre : n_pre + n_swing] = 0.0
    _gait(spec, dof, t, gait_amp)
    # Swing: back, then forward through the ball, then settle.
    s = np.linspace(0.0, 1.0, n_swing)
    hip = _joint(spec, "right", "hip_pitch")
    knee = _joint(spec, "right", "knee")
    dof[n_pre : n_pre + n_swing, hip] += KICK_SWING * (0.5 * np.sin(2 * np.pi * s) - 0.5 * np.sin(np.pi * s))
    dof[n_pre : n_pre + n_swing, knee] += KICK_SWING * np.sin(np.pi * s) ** 2
    xy, yaw = _integrate(speed, np.zeros(T))
    contact = n_pre + int(0.6 * n_swing)
    frames = {
        "entry_start": int((1.2 if kind == "run" else 0.3) * FPS),
        "skill_start": n_pre - int(0.3 * FPS),
        "contact_frame": contact,
        "skill_end": T - int(0.5 * FPS),
    }
    return _qpos(xy, yaw, standing_root_height(spec), dof), frames


def write_placeholder_clips(robot=None, out_dir: os.PathLike = ".") -> dict[str, Path]:
    """Write ``locomotion.npz``, ``kick_run.npz`` / ``kick_stand.npz`` and their ``*.kick.json``."""
    spec = get_robot_spec(robot)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"locomotion": out / "placeholder_locomotion.npz"}
    np.savez(paths["locomotion"], qpos=locomotion_clip(spec))
    for kind in ("run", "stand"):
        qpos, frames = kick_clip(spec, kind)
        clip = out / f"placeholder_kick_{kind}.npz"
        np.savez(clip, qpos=qpos)
        ann = {"clip": clip.name, "fps": FPS, "speed_up": 1.0, "kind": kind, "kick_foot": "right",
               "ball_offset_foot": [0.08, 0.0, 0.0], **frames}
        ann_path = out / f"placeholder_kick_{kind}.kick.json"
        ann_path.write_text(json.dumps(ann, indent=2))
        paths[f"kick_{kind}"] = ann_path
    return paths


def build_placeholder_databases(robot=None, out_dir: os.PathLike = ".", db_dir: Optional[os.PathLike] = None) -> dict:
    """Write placeholder clips and bake them into locomotion / kick_run / kick_stand databases.

    Databases land in ``db_dir`` (default ``PHP_MOTION_DATABASE_DIR``), named as
    ``SKILL_REGISTRY`` expects (``database_<robot>_locomotion.bin`` ...).
    Returns the locomotion frame count, needed for its search range.
    """
    from motion_matching.build_kick_database import build_kick_database
    from motion_matching.generate_database import build_database
    from motion_matching.io.kick_io import load_kick_annotations
    from motion_matching.utils import DATABASE_DIR

    spec = get_robot_spec(robot)
    db_dir = Path(db_dir) if db_dir is not None else DATABASE_DIR
    db_dir.mkdir(parents=True, exist_ok=True)
    paths = write_placeholder_clips(spec, out_dir)
    entries = build_database(
        [[str(paths["locomotion"]), True, 1.0, FPS, None]],
        robot=spec,
        output=str(db_dir / f"database_{spec.name}_locomotion.bin"),
    )
    for kind in ("run", "stand"):
        build_kick_database(
            load_kick_annotations(paths[f"kick_{kind}"]),
            robot=spec,
            output=str(db_dir / f"database_{spec.name}_kick_{kind}.bin"),
        )
    return {"locomotion_frames": entries[0].db_stop - entries[0].db_start, "db_dir": db_dir}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--robot", default=DEFAULT_ROBOT, choices=sorted(ROBOT_SPECS))
    parser.add_argument("--out", default="placeholder_clips")
    parser.add_argument("--build", action="store_true", help="also bake the databases into PHP_MOTION_DATABASE_DIR")
    args = parser.parse_args(argv)
    if args.build:
        info = build_placeholder_databases(args.robot, args.out)
        print(f"Built placeholder databases in {info['db_dir']} (locomotion: {info['locomotion_frames']} frames)")
    else:
        print(write_placeholder_clips(args.robot, args.out))


if __name__ == "__main__":
    main()
