"""Kick clip annotations, kick database sidecars and the kick fields of motion npz files.

Three layers, from source data to training data:

1. **Clip annotation** (``*.kick.json`` next to a retargeted clip, written in
   Phase 1). Frames are source-clip frames at the clip's own fps::

       {
         "clip": "kick_run_01.npz",       # relative to the json, qpos (T, 7 + N_dof)
         "fps": 30,
         "speed_up": 0.7,                  # < 1 slows the clip down (0.6x to 0.8x for kicks)
         "kind": "run",                    # "run" (Anlauf) or "stand" (aus dem Stand)
         "kick_foot": "right",
         "entry_start": 40,                # start of the entry window (run-up steps / calm standing pose)
         "skill_start": 70,                # last frame motion matching may enter the clip
         "contact_frame": 112,             # foot meets ball
         "skill_end": 160,                 # hand back to locomotion
         "ball_offset_foot": [0.06, 0.0, 0.0]   # ball centre relative to the kick foot at contact,
                                                 # in the robot's heading frame (x forward, y left)
         # or instead: "ball_pos": [x, y, z] in the clip's world frame
       }

   A file may also hold a list of such objects.

2. **Database sidecar** (``database_<robot>_<family>.kicks.json`` next to the
   ``.bin``), written by ``build_kick_database``. Everything is mapped onto
   database frames (60 fps, mirrored copies included)::

       {"robot": "lupus", "ball_radius": 0.075, "ranges": [
          {"search": [entry_start, skill_start, skill_end], "contact_frame": c,
           "kick_foot": "right", "kind": "run", "ball_pos": [x, y, z],
           "clip": "...", "mirrored": false}, ...]}

   ``search`` is exactly the ``(start_search, end_search, end_of_motion)``
   triple of ``SKILL_REGISTRY["..."]["search_ranges"]``; mirrored ranges have
   the other ``kick_foot``.

3. **Motion npz fields** written by ``process_scenario`` for soccer scenarios,
   agreed with the teacher (Phase 3), see :data:`KICK_NPZ_FIELDS`.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

# --------------------------------------------------------------------------
# Ball
# --------------------------------------------------------------------------

#: Fallback ball radius (m) until ``robots/ball/ball.yaml`` exists. 0.15 m
#: diameter is a placeholder until the league rules are settled.
DEFAULT_BALL_RADIUS = 0.075

#: Repository-local ball spec (Phase 3). ``BALL_SPEC`` overrides the path.
BALL_SPEC_PATH = Path(
    os.environ.get("BALL_SPEC", Path(__file__).resolve().parents[3] / "robots" / "ball" / "ball.yaml")
)


def load_ball_radius(path: Optional[os.PathLike] = None) -> float:
    """Ball radius in metres from ``ball.yaml`` (``diameter: {value: ...}``).

    Accepts ``diameter`` or ``radius``, each either a number or a
    ``{value, status}`` mapping. Falls back to :data:`DEFAULT_BALL_RADIUS` when
    the file is missing or has neither key.
    """
    path = Path(path) if path is not None else BALL_SPEC_PATH
    if not path.is_file():
        return DEFAULT_BALL_RADIUS
    text = path.read_text()
    try:
        import yaml

        spec = yaml.safe_load(text) or {}
    except ImportError:  # motion_matching does not depend on pyyaml
        spec = _parse_simple_yaml_numbers(text)

    def _value(entry: Any) -> Optional[float]:
        if isinstance(entry, dict):
            entry = entry.get("value")
        return None if entry is None else float(entry)

    radius = _value(spec.get("radius"))
    if radius is None:
        diameter = _value(spec.get("diameter"))
        radius = None if diameter is None else 0.5 * diameter
    return DEFAULT_BALL_RADIUS if radius is None else radius


def _parse_simple_yaml_numbers(text: str) -> dict[str, Any]:
    """Read top-level ``key: number`` / ``key: {value: number}`` / nested ``value:`` entries."""
    import re

    spec: dict[str, Any] = {}
    current = None
    for line in text.splitlines():
        line = line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        top = re.match(r"^([A-Za-z_][\w]*)\s*:\s*(.*)$", line)
        if top:
            current, rest = top.group(1), top.group(2).strip()
            inline = re.search(r"value\s*:\s*([-+0-9.eE]+)", rest)
            if inline:
                spec[current] = float(inline.group(1))
            elif re.fullmatch(r"[-+0-9.eE]+", rest):
                spec[current] = float(rest)
            continue
        nested = re.match(r"^\s+value\s*:\s*([-+0-9.eE]+)", line)
        if nested and current is not None:
            spec[current] = float(nested.group(1))
    return spec


# --------------------------------------------------------------------------
# Clip annotations (Phase 1 output)
# --------------------------------------------------------------------------

KICK_FEET = ("right", "left")
KICK_KINDS = ("run", "stand")


@dataclass
class KickClipAnnotation:
    clip: str
    fps: float
    kick_foot: str
    entry_start: int
    skill_start: int
    contact_frame: int
    skill_end: int
    kind: str = "run"
    speed_up: float = 1.0
    ball_offset_foot: Optional[list[float]] = None
    ball_pos: Optional[list[float]] = None

    def validate(self) -> None:
        if self.kick_foot not in KICK_FEET:
            raise ValueError(f"{self.clip}: kick_foot must be one of {KICK_FEET}, got {self.kick_foot!r}")
        if self.kind not in KICK_KINDS:
            raise ValueError(f"{self.clip}: kind must be one of {KICK_KINDS}, got {self.kind!r}")
        if not (0 <= self.entry_start <= self.skill_start <= self.contact_frame <= self.skill_end):
            raise ValueError(
                f"{self.clip}: need 0 <= entry_start <= skill_start <= contact_frame <= skill_end, got "
                f"{self.entry_start}, {self.skill_start}, {self.contact_frame}, {self.skill_end}"
            )
        if (self.ball_offset_foot is None) == (self.ball_pos is None):
            raise ValueError(f"{self.clip}: give exactly one of ball_offset_foot / ball_pos")
        if not (self.fps > 0 and self.speed_up > 0):
            raise ValueError(f"{self.clip}: fps and speed_up must be positive")


def load_kick_annotations(path: os.PathLike) -> list[tuple[Path, KickClipAnnotation]]:
    """Read one ``*.kick.json`` (object or list). Returns ``(clip_path, annotation)`` pairs."""
    path = Path(path)
    with open(path) as f:
        raw = json.load(f)
    items = raw if isinstance(raw, list) else [raw]
    out = []
    for item in items:
        ann = KickClipAnnotation(**item)
        ann.validate()
        clip_path = Path(ann.clip)
        if not clip_path.is_absolute():
            clip_path = path.parent / clip_path
        out.append((clip_path, ann))
    return out


def _yaw_rotate(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate ``vec`` by the yaw part of ``quat_wxyz``."""
    w, x, y, z = quat_wxyz
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1], vec[2]], dtype=np.float64)


# --------------------------------------------------------------------------
# Database sidecar
# --------------------------------------------------------------------------


@dataclass
class KickRange:
    search: tuple[int, int, int]
    contact_frame: int
    kick_foot: str
    kind: str
    ball_pos: tuple[float, float, float]
    clip: str = ""
    mirrored: bool = False

    def contains(self, frame: int) -> bool:
        return self.search[0] <= frame < self.search[2]


@dataclass
class KickSidecar:
    robot: str
    ranges: list[KickRange] = field(default_factory=list)
    ball_radius: float = DEFAULT_BALL_RADIUS

    def select(self, *, kick_foot: Optional[str] = None, kind: Optional[str] = None) -> list[KickRange]:
        return [
            r
            for r in self.ranges
            if (kick_foot is None or r.kick_foot == kick_foot) and (kind is None or r.kind == kind)
        ]

    def save(self, path: os.PathLike) -> None:
        payload = {"robot": self.robot, "ball_radius": self.ball_radius, "ranges": [asdict(r) for r in self.ranges]}
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: os.PathLike) -> "KickSidecar":
        with open(path) as f:
            raw = json.load(f)
        ranges = [
            KickRange(
                search=tuple(int(v) for v in r["search"]),
                contact_frame=int(r["contact_frame"]),
                kick_foot=r["kick_foot"],
                kind=r.get("kind", "run"),
                ball_pos=tuple(float(v) for v in r["ball_pos"]),
                clip=r.get("clip", ""),
                mirrored=bool(r.get("mirrored", False)),
            )
            for r in raw["ranges"]
        ]
        return cls(robot=raw["robot"], ranges=ranges, ball_radius=float(raw.get("ball_radius", DEFAULT_BALL_RADIUS)))


def sidecar_path_for(bin_path: os.PathLike) -> str:
    """``database_x.bin`` -> ``database_x.kicks.json``."""
    root, _ = os.path.splitext(str(bin_path))
    return root + ".kicks.json"


def kick_ranges_from_entries(
    annotations: Sequence[tuple[os.PathLike, KickClipAnnotation]], entries: Sequence[Any], *, ball_radius: float
) -> list[KickRange]:
    """Map clip annotations onto database frames.

    Args:
        annotations: ``(clip_path, annotation)`` pairs as returned by
            :func:`load_kick_annotations`.
        entries: :class:`~motion_matching.generate_database.DatabaseEntry`
            objects of the built database. Each clip owns its unmirrored and
            (if baked) mirrored entry, matched by ``filename == str(clip_path)``.
        ball_radius: Resting height of the ball centre.
    """
    ranges: list[KickRange] = []
    pending = list(entries)
    for clip_path, ann in annotations:
        owned = [e for e in pending if e.filename == str(clip_path)]
        if not owned:
            raise ValueError(f"no database entry for clip {ann.clip}")
        for e in owned:
            pending.remove(e)
            frames = [e.source_to_db_frame(f) for f in (ann.entry_start, ann.skill_start, ann.skill_end)]
            contact = e.source_to_db_frame(ann.contact_frame)
            foot = ann.kick_foot
            if e.mirrored:
                foot = "left" if foot == "right" else "right"
            foot_idx = 0 if foot == "left" else 1
            local = contact - e.db_start
            if ann.ball_offset_foot is not None:
                offset = np.asarray(ann.ball_offset_foot, dtype=np.float64)
                if e.mirrored:
                    offset = offset * np.array([1.0, -1.0, 1.0])
                ball = e.foot_positions[local, foot_idx].astype(np.float64) + _yaw_rotate(
                    e.sim_rotations[local].astype(np.float64), offset
                )
            else:
                ball = np.asarray(ann.ball_pos, dtype=np.float64).copy()
                if e.mirrored:
                    ball[1] = -ball[1]
            ball[2] = ball_radius  # the ball rests on flat ground
            # skill_end is exclusive in set_custom_search_ranges; keep it inside the clip
            frames[2] = min(frames[2], e.db_stop - 1)
            ranges.append(
                KickRange(
                    search=(frames[0], frames[1], frames[2]),
                    contact_frame=contact,
                    kick_foot=foot,
                    kind=ann.kind,
                    ball_pos=tuple(float(v) for v in ball),
                    clip=os.path.basename(ann.clip),
                    mirrored=e.mirrored,
                )
            )
    return ranges


# --------------------------------------------------------------------------
# Motion npz fields (agreed with the Phase 3 teacher loader)
# --------------------------------------------------------------------------

KICK_PHASE_NONE = 0  # walking / standing, ball must not be touched
KICK_PHASE_COMMIT = 1  # run-up / commit until contact
KICK_PHASE_AFTER = 2  # after contact

#: Per-frame fields and their dtypes / trailing shapes.
KICK_NPZ_FIELDS: dict[str, tuple[np.dtype, tuple[int, ...]]] = {
    "object_pos_w": (np.dtype(np.float32), (3,)),
    "object_quat_w": (np.dtype(np.float32), (4,)),
    "object_lin_vel_w": (np.dtype(np.float32), (3,)),
    "kick_enable": (np.dtype(np.float32), ()),
    "kick_phase": (np.dtype(np.int8), ()),
    "contact_frame": (np.dtype(bool), ()),
    # Phase 6 placeholders (agreed with the student thread): kick direction
    # relative to the heading in rad, and mode 0 ignore / 1 kick / 2 dribble.
    "kick_dir": (np.dtype(np.float32), ()),
    "kick_mode": (np.dtype(np.int8), ()),
}

KICK_MODE_IGNORE = 0
KICK_MODE_KICK = 1
KICK_MODE_DRIBBLE = 2

#: Scalar metadata: ``kick_side`` 0 right, 1 left, -1 no kick; ``scenario`` name.
KICK_SIDE = {"right": 0, "left": 1, None: -1}

#: Continuous command layout of ``vel_cmd`` in soccer motions.
VEL_CMD_VW_DIM = 2  # [v (m/s), omega (rad/s)]


def resample_hold(values: np.ndarray, n_out: int) -> np.ndarray:
    """Zero-order-hold resample of per-frame ``values`` onto ``n_out`` frames.

    Same sampling grid as ``resample_vel_cmd`` / ``adjust_vel_cmd_speed``
    (``linspace(0, n-1, n_out)``, value of the previous source frame).
    """
    values = np.asarray(values)
    n = len(values)
    if n == 0 or n_out == n:
        return values.copy()
    t = np.linspace(0, n - 1, n_out)
    idx = np.clip(np.floor(t + 1e-9).astype(np.int64), 0, n - 1)
    return values[idx]


def map_frame(index: int, n_src: int, n_out: int) -> int:
    """Nearest output frame of source frame ``index`` on a ``linspace`` grid."""
    if n_src <= 1 or n_out <= 1:
        return 0
    return int(np.clip(round(index * (n_out - 1) / (n_src - 1)), 0, n_out - 1))


def build_kick_fields(
    *,
    num_frames: int,
    kick_enable: np.ndarray,
    commit_start: Optional[int],
    contact: Optional[int],
    ball_pos: np.ndarray,
) -> dict[str, np.ndarray]:
    """Assemble the per-frame kick fields for one motion at output fps.

    Args:
        num_frames: Output frame count ``T``.
        kick_enable: ``(T,)`` 0/1.
        commit_start: First frame of kick phase 1 (skill trigger), or ``None``.
        contact: Contact frame, or ``None`` when the motion has no kick.
        ball_pos: ``(3,)`` resting ball position (constant over the motion), or
            ``(T, 3)``.
    """
    T = int(num_frames)
    phase = np.zeros(T, dtype=np.int8)
    contact_mask = np.zeros(T, dtype=bool)
    if contact is not None:
        start = contact if commit_start is None else min(commit_start, contact)
        phase[start:contact] = KICK_PHASE_COMMIT
        phase[contact:] = KICK_PHASE_AFTER
        contact_mask[contact] = True
    ball_pos = np.asarray(ball_pos, dtype=np.float32)
    if ball_pos.ndim == 1:
        ball_pos = np.repeat(ball_pos[None], T, axis=0)
    quat = np.zeros((T, 4), dtype=np.float32)
    quat[:, 0] = 1.0
    kick_enable = np.asarray(kick_enable, dtype=np.float32).reshape(T)
    return {
        "object_pos_w": ball_pos.astype(np.float32),
        "object_quat_w": quat,
        "object_lin_vel_w": np.zeros((T, 3), dtype=np.float32),
        "kick_enable": kick_enable,
        "kick_phase": phase,
        "contact_frame": contact_mask,
        "kick_dir": np.zeros(T, dtype=np.float32),
        "kick_mode": np.where(kick_enable > 0.5, KICK_MODE_KICK, KICK_MODE_IGNORE).astype(np.int8),
    }


def validate_kick_npz(data: Any) -> None:
    """Raise ``ValueError`` if a loaded motion npz lacks or misshapes kick fields."""
    files = set(getattr(data, "files", data.keys()))
    T = None
    for key, (dtype, shape) in KICK_NPZ_FIELDS.items():
        if key not in files:
            raise ValueError(f"missing kick field {key!r}")
        arr = np.asarray(data[key])
        if arr.dtype != dtype:
            raise ValueError(f"{key}: dtype {arr.dtype}, expected {dtype}")
        if arr.shape[1:] != shape:
            raise ValueError(f"{key}: shape {arr.shape}, expected (T, *{shape})")
        T = len(arr) if T is None else T
        if len(arr) != T:
            raise ValueError(f"{key}: {len(arr)} frames, expected {T}")
    phase = np.asarray(data["kick_phase"])
    if not set(np.unique(phase)) <= {0, 1, 2}:
        raise ValueError("kick_phase must be in {0, 1, 2}")
    if np.any(np.diff(phase.astype(np.int64)) < 0):
        raise ValueError("kick_phase must not decrease")
    contacts = np.flatnonzero(np.asarray(data["contact_frame"]))
    if len(contacts) > 1:
        raise ValueError("at most one contact frame")
    if len(contacts) == 1 and phase[contacts[0]] != KICK_PHASE_AFTER:
        raise ValueError("contact frame must start kick_phase 2")
    if len(contacts) == 0 and np.any(phase == KICK_PHASE_AFTER):
        raise ValueError("kick_phase 2 without contact frame")
    if "vel_cmd" in files:
        vel = np.asarray(data["vel_cmd"])
        if vel.shape != (T, VEL_CMD_VW_DIM):
            raise ValueError(f"vel_cmd: shape {vel.shape}, expected ({T}, {VEL_CMD_VW_DIM})")
