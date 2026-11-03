"""LUPUS clip container and the edits of Phase 1: time scaling, mirroring, ground fix.

A :class:`LupusClip` holds PHP qpos (see :mod:`reference_data.lupus_kin`) at a
fixed frame rate, plus the ball centre when the clip has one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

from reference_data.lupus_kin import kinematics, mirror_rules


@dataclass
class LupusClip:
    qpos: np.ndarray                  # (T, 29) PHP order
    fps: float
    ball_pos: np.ndarray | None = None  # (T, 3)
    ball_radius: float = 0.0
    name: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.qpos = np.asarray(self.qpos, dtype=np.float64)
        if self.qpos.ndim != 2 or self.qpos.shape[1] != 7 + len(kinematics().urdf_names):
            raise ValueError(f"qpos must be (T, 29), got {self.qpos.shape}")
        if self.ball_pos is not None:
            self.ball_pos = np.asarray(self.ball_pos, dtype=np.float64)
            if self.ball_pos.shape != (len(self.qpos), 3):
                raise ValueError("ball_pos must match qpos frames")

    @property
    def num_frames(self) -> int:
        return len(self.qpos)

    @property
    def duration(self) -> float:
        return (self.num_frames - 1) / self.fps

    @property
    def root_quat(self) -> np.ndarray:
        return self.qpos[:, 0:4]

    @property
    def root_pos(self) -> np.ndarray:
        return self.qpos[:, 4:7]

    @property
    def dofs(self) -> np.ndarray:
        return self.qpos[:, 7:]

    # ---- io -----------------------------------------------------------------

    @classmethod
    def from_retarget(cls, path: Path | str, name: str | None = None) -> LupusClip:
        """Load a ``*.lupus_raw.npz`` written by :mod:`reference_data.retarget_lupus`."""
        path = Path(path)
        kin = kinematics()
        with np.load(path, allow_pickle=False) as d:
            names = [str(n) for n in d["joint_names"]]
            if tuple(names) != kin.mj_joint_names:
                raise ValueError(f"{path}: joint order {names} differs from lupus.xml")
            qpos = kin.mj_to_php(d["qpos_mj"])
            ball = np.asarray(d["ball_pos"]) if "ball_pos" in d else None
            meta = json.loads(str(d["meta"])) if "meta" in d else {}
            return cls(qpos, float(d["fps"]), ball, float(d["ball_radius"]), name or path.stem.split(".")[0], meta)

    def save(self, path: Path | str, extra: dict | None = None) -> Path:
        """PHP source npz: ``qpos`` plus fps and (optional) ``object_pos_w``."""
        path = Path(path)
        data = {"qpos": self.qpos.astype(np.float32), "fps": np.float32(self.fps)}
        if self.ball_pos is not None:
            data["object_pos_w"] = self.ball_pos.astype(np.float32)
            data["ball_radius"] = np.float32(self.ball_radius)
        data.update(extra or {})
        np.savez(path, **data)
        return path

    @classmethod
    def load(cls, path: Path | str) -> LupusClip:
        path = Path(path)
        with np.load(path, allow_pickle=False) as d:
            ball = np.asarray(d["object_pos_w"], dtype=np.float64) if "object_pos_w" in d else None
            r = float(d["ball_radius"]) if "ball_radius" in d else 0.0
            return cls(np.asarray(d["qpos"], dtype=np.float64), float(d["fps"]), ball, r, path.stem)

    # ---- edits ----------------------------------------------------------------

    def trim(self, start: int, stop: int | None = None) -> LupusClip:
        sl = slice(start, stop)
        return replace(
            self, qpos=self.qpos[sl].copy(), ball_pos=None if self.ball_pos is None else self.ball_pos[sl].copy()
        )

    def time_scale(self, factor: float, fps: float | None = None) -> LupusClip:
        """Play the clip at ``factor`` x speed (0.7 = 30 % slower, longer clip).

        Joints and root position use a cubic spline, the root orientation
        slerp; the output is sampled at ``fps`` (default: unchanged).
        """
        if factor <= 0:
            raise ValueError("factor must be positive")
        fps = fps or self.fps
        t_src = np.arange(self.num_frames) / self.fps / factor  # new time of each source frame
        n = int(np.floor(t_src[-1] * fps + 1e-9)) + 1
        t_dst = np.arange(n) / fps
        lin = CubicSpline(t_src, np.concatenate([self.root_pos, self.dofs], axis=1), axis=0)(t_dst)
        quats = _unroll(self.root_quat)
        rot = Rotation.from_quat(quats[:, [1, 2, 3, 0]])
        q = Slerp(t_src, rot)(t_dst).as_quat()[:, [3, 0, 1, 2]]
        qpos = np.concatenate([q, lin[:, :3], lin[:, 3:]], axis=1)
        ball = None
        if self.ball_pos is not None:
            # Linear: the ball rests, then moves ballistically; a spline would
            # overshoot at the kick.
            ball = np.stack([np.interp(t_dst, t_src, self.ball_pos[:, k]) for k in range(3)], axis=1)
        meta = {**self.meta, "time_scale": self.meta.get("time_scale", 1.0) * factor}
        return replace(self, qpos=qpos, fps=fps, ball_pos=ball, meta=meta)

    def mirror(self) -> LupusClip:
        """Mirror across the x-z plane (left <-> right), ball included."""
        from motion_matching.utils.data_utils import animation_mirror_xz

        neg, lp, rp = mirror_rules()
        names = list(kinematics().urdf_names)
        dof, pos, quat = animation_mirror_xz(
            self.dofs.copy(), self.root_pos.copy(), self.root_quat.copy(), names, neg, lp, rp
        )
        ball = None
        if self.ball_pos is not None:
            ball = self.ball_pos.copy()
            ball[:, 1] *= -1
        meta = {**self.meta, "mirrored": not self.meta.get("mirrored", False)}
        return replace(
            self, qpos=np.concatenate([quat, pos, dof], axis=1), ball_pos=ball, name=self.name, meta=meta
        )

    def ground_offset(self, percentile: float = 50.0) -> float:
        """Vertical shift that puts the robot's support on the ground.

        Uses the median of the per-frame lowest robot point: in walking and
        kicking some foot is on the ground most of the time, so the median is
        the sole height, while a few frames of penetration (a toe dipping into
        the ground) or flight do not move the whole clip. Those frames stay
        visible to the feasibility check.
        """
        lowest = kinematics().clip_fk(self.qpos)["lowest"]
        return -float(np.percentile(lowest, percentile))

    def shift_z(self, dz: float) -> LupusClip:
        qpos = self.qpos.copy()
        qpos[:, 6] += dz
        ball = None
        if self.ball_pos is not None:
            ball = self.ball_pos.copy()  # the ball rests on the ground; do not move it
        return replace(self, qpos=qpos, ball_pos=ball, meta={**self.meta, "ground_shift": dz})

    def place_on_ground(self, percentile: float = 50.0) -> LupusClip:
        return self.shift_z(self.ground_offset(percentile))


def _unroll(q: np.ndarray) -> np.ndarray:
    q = np.array(q, dtype=np.float64)
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q
