"""Neutral human-clip format and loaders for the Phase 1 sources.

A :class:`HumanClip` is what the retargeter consumes: world joint positions,
z-up, metres, one naming convention (``lafan``, ``smplh`` or ``smplx``; see
``joint_mappings`` in ``lupus_retarget.yaml``), plus an optional ball
trajectory. Each source dataset gets a small loader that produces it:

=================  =====================================================
Source             Loader
=================  =====================================================
LAFAN1 BVH         :func:`from_bvh` (``fmt="lafan"``)
OmniContact        :func:`from_omnicontact` (``motion_actor.bvh`` + ball
                   from the processed npz ``object_pos_w``)
AMASS SMPL-X       :func:`from_amass_prepped` (output of holosoma
(CMU, ACCAD)       ``data_utils/prep_amass_smplx_for_rt.py``)
holosoma demo      :func:`from_intermimic_pt` (OMOMO, used by the tests)
=================  =====================================================

Saved as ``<name>.human.npz`` with ``python -m reference_data.human_clip``.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Naming convention -> joints needed for heading / height estimation.
PELVIS = {"lafan": "Hips", "smplh": "Pelvis", "smplx": "Pelvis"}
HIPS = {"lafan": ("LeftUpLeg", "RightUpLeg"), "smplh": ("L_Hip", "R_Hip"), "smplx": ("L_Hip", "R_Hip")}
HEAD = {"lafan": "Head", "smplh": "Head", "smplx": "Head"}

# BVH joint-name aliases seen in mocap exports (Motive, Xsens, Vicon, Rokoko)
# mapped to the LAFAN names. Matching is case-insensitive and ignores
# separators and a leading skeleton prefix ("Skeleton1_", "mixamorig:").
BVH_ALIASES: dict[str, tuple[str, ...]] = {
    "Hips": ("hips", "pelvis", "hip", "root"),
    "Spine": ("spine", "spine0", "ab", "lowerback"),
    "Spine1": ("spine1", "chest", "spine2lower"),
    "Spine2": ("spine2", "upperchest", "chest2", "spine3"),
    "Neck": ("neck", "neck1"),
    "Head": ("head",),
    "LeftShoulder": ("leftshoulder", "lshoulder", "leftcollar", "lcollar", "lclavicle", "leftclavicle"),
    "RightShoulder": ("rightshoulder", "rshoulder", "rightcollar", "rcollar", "rclavicle", "rightclavicle"),
    "LeftArm": ("leftarm", "leftupperarm", "luparm", "luarm", "lupperarm", "lhumerus"),
    "RightArm": ("rightarm", "rightupperarm", "ruparm", "ruarm", "rupperarm", "rhumerus"),
    "LeftForeArm": ("leftforearm", "leftlowerarm", "lfarm", "lforearm", "llowerarm", "lradius"),
    "RightForeArm": ("rightforearm", "rightlowerarm", "rfarm", "rforearm", "rlowerarm", "rradius"),
    "LeftHand": ("lefthand", "lhand", "lwrist", "leftwrist"),
    "RightHand": ("righthand", "rhand", "rwrist", "rightwrist"),
    "LeftUpLeg": ("leftupleg", "leftthigh", "lthigh", "leftupperleg", "lupleg", "lupperleg", "lfemur", "lefthip"),
    "RightUpLeg": (
        "rightupleg", "rightthigh", "rthigh", "rightupperleg", "rupleg", "rupperleg", "rfemur", "righthip",
    ),
    "LeftLeg": ("leftleg", "leftshin", "lshin", "leftlowerleg", "lleg", "llowerleg", "ltibia", "leftknee"),
    "RightLeg": ("rightleg", "rightshin", "rshin", "rightlowerleg", "rleg", "rlowerleg", "rtibia", "rightknee"),
    "LeftFoot": ("leftfoot", "lfoot", "leftankle", "lankle"),
    "RightFoot": ("rightfoot", "rfoot", "rightankle", "rankle"),
    "LeftToeBase": ("lefttoebase", "lefttoe", "ltoe", "ltoes", "lefttoes", "ltoebase"),
    "RightToeBase": ("righttoebase", "righttoe", "rtoe", "rtoes", "righttoes", "rtoebase"),
}


@dataclass
class HumanClip:
    joints: np.ndarray            # (T, J, 3) world positions, z-up, metres
    joint_names: list[str]
    fps: float
    fmt: str                      # naming convention: lafan | smplh | smplx
    human_height: float           # m, for robot/human scaling
    object_pos: np.ndarray | None = None   # (T, 3) ball centre
    object_quat: np.ndarray | None = None  # (T, 4) wxyz
    ball_diameter: float | None = None     # m, of the ball in the capture
    source: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.joints = np.asarray(self.joints, dtype=np.float64)
        if self.joints.ndim != 3 or self.joints.shape[2] != 3:
            raise ValueError(f"joints must be (T, J, 3), got {self.joints.shape}")
        if len(self.joint_names) != self.joints.shape[1]:
            raise ValueError("joint_names does not match joints")
        if self.fmt not in PELVIS:
            raise ValueError(f"unknown naming convention {self.fmt!r}")
        if self.object_pos is not None:
            self.object_pos = np.asarray(self.object_pos, dtype=np.float64)
            if self.object_pos.shape != (self.num_frames, 3):
                raise ValueError(f"object_pos must be ({self.num_frames}, 3), got {self.object_pos.shape}")
            if self.object_quat is None:
                self.object_quat = np.tile([1.0, 0.0, 0.0, 0.0], (self.num_frames, 1))
            self.object_quat = np.asarray(self.object_quat, dtype=np.float64)

    @property
    def num_frames(self) -> int:
        return self.joints.shape[0]

    @property
    def has_ball(self) -> bool:
        return self.object_pos is not None

    def joint(self, name: str) -> np.ndarray:
        return self.joints[:, self.joint_names.index(name)]

    # ---- io -----------------------------------------------------------------

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        data = dict(
            joints=self.joints,
            joint_names=np.array(self.joint_names),
            fps=self.fps,
            fmt=self.fmt,
            human_height=self.human_height,
            source=self.source,
            meta=json.dumps(self.meta),
        )
        if self.has_ball:
            data.update(object_pos=self.object_pos, object_quat=self.object_quat)
        if self.ball_diameter is not None:
            data["ball_diameter"] = self.ball_diameter
        np.savez(path, **data)
        return path

    @classmethod
    def load(cls, path: Path | str) -> HumanClip:
        with np.load(path, allow_pickle=False) as d:
            return cls(
                joints=d["joints"],
                joint_names=[str(n) for n in d["joint_names"]],
                fps=float(d["fps"]),
                fmt=str(d["fmt"]),
                human_height=float(d["human_height"]),
                object_pos=d["object_pos"] if "object_pos" in d else None,
                object_quat=d["object_quat"] if "object_quat" in d else None,
                ball_diameter=float(d["ball_diameter"]) if "ball_diameter" in d else None,
                source=str(d["source"]),
                meta=json.loads(str(d["meta"])),
            )

    # ---- editing --------------------------------------------------------------

    def resample(self, fps: float) -> HumanClip:
        """Linear resampling of all positions (the ball too) to ``fps``."""
        if abs(fps - self.fps) < 1e-9:
            return self
        duration = (self.num_frames - 1) / self.fps
        n = int(round(duration * fps)) + 1
        t_src = np.arange(self.num_frames) / self.fps
        t_dst = np.arange(n) / fps
        joints = _interp(t_src, self.joints.reshape(self.num_frames, -1), t_dst).reshape(n, -1, 3)
        obj_pos = obj_quat = None
        if self.has_ball:
            obj_pos = _interp(t_src, self.object_pos, t_dst)
            obj_quat = _interp(t_src, _unroll(self.object_quat), t_dst)
            obj_quat /= np.linalg.norm(obj_quat, axis=1, keepdims=True)
        return HumanClip(
            joints, self.joint_names, fps, self.fmt, self.human_height, obj_pos, obj_quat,
            self.ball_diameter, self.source, dict(self.meta),
        )

    def trim(self, start: int, stop: int | None = None) -> HumanClip:
        sl = slice(start, stop)
        return HumanClip(
            self.joints[sl], self.joint_names, self.fps, self.fmt, self.human_height,
            None if self.object_pos is None else self.object_pos[sl],
            None if self.object_quat is None else self.object_quat[sl],
            self.ball_diameter, self.source, {**self.meta, "trim": [start, stop]},
        )

    def heading_quat(self, frame: int = 0) -> np.ndarray:
        """Yaw-only wxyz quaternion of the pelvis heading (x forward, y left)."""
        lh, rh = HIPS[self.fmt]
        left = self.joint(lh)[frame] - self.joint(rh)[frame]
        left[2] = 0.0
        left /= np.linalg.norm(left)
        fwd = np.cross(left, [0.0, 0.0, 1.0])
        yaw = np.arctan2(fwd[1], fwd[0])
        return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _interp(t_src: np.ndarray, x: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(t_dst, t_src, x[:, k]) for k in range(x.shape[1])], axis=1)


def _unroll(q: np.ndarray) -> np.ndarray:
    q = q.copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def y_up_to_z_up(p: np.ndarray) -> np.ndarray:
    """Proper rotation (+90 deg about x): (x, y, z)_yup -> (x, -z, y)_zup.

    holosoma's ``transform_y_up_to_z_up`` swaps y and z, which is a
    reflection and silently turns left into right; this keeps handedness.
    """
    p = np.asarray(p, dtype=np.float64)
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], axis=-1)


def estimate_height(joints: np.ndarray, names: list[str], fmt: str) -> float:
    """Standing height estimate: median head height above the lowest foot point + 0.1 m crown."""
    head = joints[:, names.index(HEAD[fmt]), 2]
    floor = np.min(joints[:, :, 2], axis=1)
    return float(np.median(head - floor) + 0.10)


# ---------------------------------------------------------------------------
# BVH
# ---------------------------------------------------------------------------


@dataclass
class Bvh:
    names: list[str]
    parents: np.ndarray        # (J,)
    offsets: np.ndarray        # (J, 3)
    channels: list[list[str]]  # per joint
    frame_time: float
    motion: np.ndarray         # (T, C)
    end_sites: dict[str, np.ndarray] = field(default_factory=dict)  # parent name -> offset


def parse_bvh(text: str) -> Bvh:
    tokens = re.findall(r"[^\s{}]+|[{}]", text)
    i = 0
    names: list[str] = []
    parents: list[int] = []
    offsets: list[list[float]] = []
    channels: list[list[str]] = []
    end_sites: dict[str, np.ndarray] = {}
    stack: list[int] = []
    pending_end = False

    while i < len(tokens):
        tok = tokens[i]
        up = tok.upper()
        if up in ("ROOT", "JOINT"):
            names.append(tokens[i + 1])
            parents.append(stack[-1] if stack else -1)
            offsets.append([0.0, 0.0, 0.0])
            channels.append([])
            i += 2
            continue
        if up == "END":  # "End Site"
            pending_end = True
            i += 2
            continue
        if tok == "{":
            stack.append(-2 if pending_end else len(names) - 1)
            i += 1
            continue
        if tok == "}":
            stack.pop()
            pending_end = False
            i += 1
            continue
        if up == "OFFSET":
            vals = [float(v) for v in tokens[i + 1 : i + 4]]
            if stack and stack[-1] == -2:
                end_sites[names[_last_real(stack)]] = np.array(vals)
            else:
                offsets[-1] = vals
            i += 4
            continue
        if up == "CHANNELS":
            n = int(tokens[i + 1])
            channels[-1] = tokens[i + 2 : i + 2 + n]
            i += 2 + n
            continue
        if up == "MOTION":
            break
        i += 1

    m = re.search(r"Frames:\s*(\d+)\s*Frame Time:\s*([0-9.eE+-]+)\s*(.*)", text, re.S | re.I)
    if not m:
        raise ValueError("BVH has no MOTION section")
    n_frames = int(m.group(1))
    frame_time = float(m.group(2))
    n_ch = sum(len(c) for c in channels)
    values = np.array(m.group(3).split(), dtype=np.float64)
    if values.size < n_frames * n_ch:
        raise ValueError(f"BVH motion has {values.size} values, expected {n_frames * n_ch}")
    motion = values[: n_frames * n_ch].reshape(n_frames, n_ch)
    return Bvh(names, np.array(parents), np.array(offsets), channels, frame_time, motion, end_sites)


def _last_real(stack: list[int]) -> int:
    for s in reversed(stack):
        if s >= 0:
            return s
    raise ValueError("End Site outside a joint")


def _axis_rot(axis: str, angle: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    o, z = np.ones_like(angle), np.zeros_like(angle)
    if axis == "X":
        m = [[o, z, z], [z, c, -s], [z, s, c]]
    elif axis == "Y":
        m = [[c, z, s], [z, o, z], [-s, z, c]]
    else:
        m = [[c, -s, z], [s, c, z], [z, z, o]]
    return np.moveaxis(np.array(m), [0, 1], [-2, -1])


def bvh_global_positions(bvh: Bvh) -> np.ndarray:
    """Forward kinematics, (T, J, 3) in the BVH's own units and up axis."""
    T, J = bvh.motion.shape[0], len(bvh.names)
    glob_rot = np.zeros((T, J, 3, 3))
    glob_pos = np.zeros((T, J, 3))
    col = 0
    for j in range(J):
        local_rot = np.broadcast_to(np.eye(3), (T, 3, 3)).copy()
        local_pos = np.broadcast_to(bvh.offsets[j], (T, 3)).copy()
        for ch in bvh.channels[j]:
            v = bvh.motion[:, col]
            col += 1
            kind, axis = ch[1:].lower(), ch[0].upper()
            if kind == "position":
                local_pos[:, "XYZ".index(axis)] = v
            elif kind == "rotation":
                local_rot = local_rot @ _axis_rot(axis, np.deg2rad(v))
            else:
                raise ValueError(f"unknown BVH channel {ch!r}")
        p = bvh.parents[j]
        if p < 0:
            glob_rot[:, j] = local_rot
            glob_pos[:, j] = local_pos
        else:
            glob_rot[:, j] = glob_rot[:, p] @ local_rot
            glob_pos[:, j] = glob_pos[:, p] + np.einsum("tij,tj->ti", glob_rot[:, p], local_pos)
    return glob_pos


def _canon(name: str) -> str:
    name = name.split(":")[-1]
    name = re.sub(r"^(skeleton\d*|skel\d*|bip\d*|char\d*)[_ ]", "", name, flags=re.I)
    return re.sub(r"[^a-z0-9]", "", name.lower())


def map_bvh_names(
    names: list[str], parents: np.ndarray | None = None, extra_aliases: dict[str, str] | None = None
) -> dict[str, int]:
    """LAFAN name -> BVH joint index.

    End joints (hips, head, hands, feet, toes) are matched by name with
    :data:`BVH_ALIASES`. Elbow/shoulder and knee/hip are then taken from the
    skeleton structure (parent of the hand is the elbow, and so on), because
    exporters disagree on what "Shoulder" means (Xsens: upper arm, LAFAN:
    clavicle). ``extra_aliases`` (LAFAN name -> BVH name) override everything.
    """
    canon = {_canon(n): i for i, n in enumerate(names)}
    out: dict[str, int] = {}
    for target, aliases in BVH_ALIASES.items():
        for a in (target.lower(),) + aliases:
            if a in canon:
                out[target] = canon[a]
                break
    if parents is not None:
        for side in ("Left", "Right"):
            hand, foot = out.get(f"{side}Hand"), out.get(f"{side}Foot")
            if hand is not None and parents[hand] >= 0:
                out[f"{side}ForeArm"] = int(parents[hand])
                if parents[parents[hand]] >= 0:
                    out[f"{side}Arm"] = int(parents[parents[hand]])
            if foot is not None and parents[foot] >= 0:
                out[f"{side}Leg"] = int(parents[foot])
                if parents[parents[foot]] >= 0:
                    out[f"{side}UpLeg"] = int(parents[parents[foot]])
            if f"{side}ToeBase" not in out and foot is not None:
                kids = [i for i, p in enumerate(parents) if p == foot]
                if len(kids) == 1:
                    out[f"{side}ToeBase"] = kids[0]
        for side in ("Left", "Right"):  # a clavicle alias must not alias the upper arm
            if out.get(f"{side}Shoulder") is not None and out.get(f"{side}Shoulder") == out.get(f"{side}Arm"):
                del out[f"{side}Shoulder"]
    for target, src in (extra_aliases or {}).items():
        out[target] = names.index(src)
    return out


REQUIRED_LAFAN = (
    "Hips", "LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg", "LeftFoot", "RightFoot",
    "LeftToeBase", "RightToeBase", "LeftArm", "RightArm", "LeftForeArm", "RightForeArm",
    "LeftHand", "RightHand", "Head",
)


def from_bvh(
    path: Path | str,
    *,
    unit_scale: float | None = None,
    up_axis: str = "y",
    aliases: dict[str, str] | None = None,
    human_height: float | None = None,
) -> HumanClip:
    """Load a BVH (LAFAN1, OmniContact ``motion_actor.bvh``, CMU BVH exports).

    ``unit_scale`` converts to metres; ``None`` guesses cm vs m from the leg
    length. Joints are renamed to the LAFAN convention.
    """
    path = Path(path)
    bvh = parse_bvh(path.read_text())
    pos = bvh_global_positions(bvh)
    idx = map_bvh_names(bvh.names, bvh.parents, aliases)
    missing = [n for n in REQUIRED_LAFAN if n not in idx]
    if missing:
        raise ValueError(
            f"{path.name}: cannot map BVH joints {missing}; BVH has {bvh.names}. "
            "Pass aliases={'LeftToeBase': '<bvh name>', ...}."
        )
    names = list(idx)
    joints = pos[:, [idx[n] for n in names]]
    if up_axis.lower() == "y":
        joints = y_up_to_z_up(joints)
    elif up_axis.lower() != "z":
        raise ValueError("up_axis must be 'y' or 'z'")
    if unit_scale is None:
        leg = np.linalg.norm(joints[0, names.index("LeftUpLeg")] - joints[0, names.index("LeftFoot")])
        unit_scale = 0.01 if leg > 5.0 else 1.0
    joints = joints * unit_scale
    height = human_height or estimate_height(joints, names, "lafan")
    return HumanClip(
        joints, names, 1.0 / bvh.frame_time, "lafan", height, source=str(path),
        meta={"bvh_joints": bvh.names, "unit_scale": unit_scale, "up_axis": up_axis},
    )


# ---------------------------------------------------------------------------
# Other sources
# ---------------------------------------------------------------------------


def from_omnicontact(
    bvh_path: Path | str,
    npz_path: Path | str,
    *,
    ball_diameter: float | None = None,
    up_axis: str = "y",
    npz_up_axis: str = "z",
    aliases: dict[str, str] | None = None,
) -> HumanClip:
    """OmniContact soccer clip: actor BVH + ball centre from the processed npz.

    The processed npz (``object_pos_w``, ``fps``) is on the G1 retarget time
    axis; it is resampled onto the BVH frames by time. Both are assumed to
    start at the same instant (checked on the first clips by
    ``verify.py --overlay``; see README).
    """
    clip = from_bvh(bvh_path, up_axis=up_axis, aliases=aliases)
    with np.load(npz_path, allow_pickle=False) as d:
        obj = np.asarray(d["object_pos_w"], dtype=np.float64)
        npz_fps = float(np.ravel(d["fps"])[0])
        quat = np.asarray(d["object_quat_w"], dtype=np.float64) if "object_quat_w" in d else None
    if npz_up_axis == "y":
        obj = y_up_to_z_up(obj)
    t_obj = np.arange(len(obj)) / npz_fps
    t_bvh = np.arange(clip.num_frames) / clip.fps
    clip.object_pos = _interp(t_obj, obj, t_bvh)
    clip.object_quat = None if quat is None else _interp(t_obj, _unroll(quat), t_bvh)
    if clip.object_quat is not None:
        clip.object_quat /= np.linalg.norm(clip.object_quat, axis=1, keepdims=True)
    clip.ball_diameter = ball_diameter
    clip.meta["object_source"] = str(npz_path)
    clip.__post_init__()
    return clip


SMPLX_JOINTS = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2", "L_Ankle", "R_Ankle", "Spine3",
    "L_Foot", "R_Foot", "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow",
    "R_Elbow", "L_Wrist", "R_Wrist",
]


def from_amass_prepped(path: Path | str, fps: float = 30.0) -> HumanClip:
    """AMASS SMPL-X after holosoma ``prep_amass_smplx_for_rt.py`` (30 fps, z-up)."""
    with np.load(path, allow_pickle=False) as d:
        joints = np.asarray(d["global_joint_positions"], dtype=np.float64)
        height = float(d["height"])
    names = SMPLX_JOINTS[: joints.shape[1]]
    return HumanClip(joints, names, fps, "smplx", height, source=str(path))


SMPLH_JOINTS = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe", "Torso", "Spine",
    "Chest", "Neck", "Head", "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist",
    *[f"L_{f}{k}" for f in ("Index", "Middle", "Pinky", "Ring", "Thumb") for k in (1, 2, 3)],
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist",
    *[f"R_{f}{k}" for f in ("Index", "Middle", "Pinky", "Ring", "Thumb") for k in (1, 2, 3)],
]


def from_intermimic_pt(path: Path | str, human_height: float | None = None, with_object: bool = False) -> HumanClip:
    """holosoma demo data (OMOMO via InterMimic ``.pt``, 30 fps, z-up). Needs torch."""
    import torch

    data = torch.load(str(path), map_location="cpu").detach().numpy().astype(np.float64)
    joints = data[:, 162 : 162 + 52 * 3].reshape(-1, 52, 3)
    obj = data[:, 318:325][:, [6, 3, 4, 5, 0, 1, 2]]  # wxyz + xyz
    height = human_height or estimate_height(joints, SMPLH_JOINTS, "smplh")
    return HumanClip(
        joints, list(SMPLH_JOINTS), 30.0, "smplh", height,
        object_pos=obj[:, 4:] if with_object else None,
        object_quat=obj[:, :4] if with_object else None,
        source=str(path),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert a source clip into the neutral human-clip npz.")
    ap.add_argument("kind", choices=["bvh", "omnicontact", "amass", "intermimic"])
    ap.add_argument("path", type=Path)
    ap.add_argument("--object-npz", type=Path, help="omnicontact: processed npz with object_pos_w")
    ap.add_argument("--ball-diameter", type=float)
    ap.add_argument("--up-axis", default="y")
    ap.add_argument("--trim", type=int, nargs=2, metavar=("START", "STOP"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.kind == "bvh":
        clip = from_bvh(args.path, up_axis=args.up_axis)
    elif args.kind == "omnicontact":
        if args.object_npz is None:
            ap.error("omnicontact needs --object-npz")
        clip = from_omnicontact(args.path, args.object_npz, ball_diameter=args.ball_diameter, up_axis=args.up_axis)
    elif args.kind == "amass":
        clip = from_amass_prepped(args.path)
    else:
        clip = from_intermimic_pt(args.path)
    if args.trim:
        clip = clip.trim(*args.trim)
    clip.save(args.out)
    print(f"{args.out}: {clip.num_frames} frames @ {clip.fps:g} fps, fmt={clip.fmt}, height={clip.human_height:.2f} m")


if __name__ == "__main__":
    main()
