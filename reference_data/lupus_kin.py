"""MuJoCo kinematics helpers for LUPUS clips (FK, feet, ground, limits).

Conventions used throughout ``reference_data``:

- **PHP qpos** ``(T, 7 + 22)`` = ``[root_quat_wxyz, root_pos, dofs]`` with the
  DoFs in URDF order (``LUPUS_SPEC.joint_names``). This is the motion-matching
  source format (``generate_database --robot lupus``).
- **MuJoCo qpos** ``(T, 7 + 22)`` = ``[root_pos, root_quat_wxyz, dofs]`` with
  the DoFs in MJCF order (legs, arms, neck). This is what the retargeter
  writes and what MuJoCo plays.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
LUPUS_DIR = REPO / "robots" / "lupus"
MJCF = LUPUS_DIR / "lupus.xml"
JOINTS_YAML = LUPUS_DIR / "lupus_joints.yaml"

SIDES = ("left", "right")


@functools.lru_cache(maxsize=None)
def urdf_joint_names() -> tuple[str, ...]:
    """DoF order of PHP qpos (= ``LUPUS_SPEC.joint_names``)."""
    import sys

    mm = REPO / "motion_matching"
    if str(mm) not in sys.path:
        sys.path.insert(0, str(mm))
    from motion_matching.kinematics.robot_spec import LUPUS_SPEC

    return tuple(LUPUS_SPEC.joint_names)


@functools.lru_cache(maxsize=None)
def mirror_rules() -> tuple[tuple[str, ...], str, str]:
    urdf_joint_names()  # makes motion_matching importable
    from motion_matching.kinematics.robot_spec import LUPUS_SPEC

    return LUPUS_SPEC.mirror_negate_joints, LUPUS_SPEC.left_prefix, LUPUS_SPEC.right_prefix


@functools.lru_cache(maxsize=None)
def joint_table() -> dict[str, dict]:
    return yaml.safe_load(JOINTS_YAML.read_text())["joints"]


def _model_without_meshes() -> mujoco.MjModel:
    """lupus.xml with the visual meshes dropped (no meshes/ folder needed)."""
    import xml.etree.ElementTree as ET

    root = ET.parse(MJCF).getroot()
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "geom" and child.get("class") == "visual":
                parent.remove(child)
    asset = root.find("asset")
    for mesh in list(asset.findall("mesh")):
        asset.remove(mesh)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


@dataclass
class FootGeom:
    body: int
    geom: int
    half: np.ndarray  # (3,) half sizes in geom frame


class LupusKinematics:
    def __init__(self) -> None:
        self.model = _model_without_meshes()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.mj_joint_names = tuple(m.joint(j).name for j in range(1, m.njnt))
        names = urdf_joint_names()
        if sorted(names) != sorted(self.mj_joint_names):
            raise ValueError("MJCF and URDF joint sets differ")
        self.urdf_names = names
        # php dof index -> mj dof index
        self.perm_php_to_mj = np.array([names.index(n) for n in self.mj_joint_names])
        self.perm_mj_to_php = np.array([self.mj_joint_names.index(n) for n in names])
        self.feet = {}
        for s in SIDES:
            g = m.geom(f"{s}_foot_link_collision").id
            self.feet[s] = FootGeom(m.geom_bodyid[g], g, m.geom_size[g].copy())
        self.collision_geoms = [
            g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and m.geom_bodyid[g] != 0
        ]
        tab = joint_table()
        self.lower = np.array([tab[n]["lower"] for n in names])
        self.upper = np.array([tab[n]["upper"] for n in names])
        self.vel_limit = np.array([tab[n]["velocity"] for n in names])
        self.effort = np.array([tab[n]["effort"] for n in names])

    # ---- format conversion -------------------------------------------------

    def mj_to_php(self, qpos_mj: np.ndarray) -> np.ndarray:
        q = np.asarray(qpos_mj, dtype=np.float64)
        return np.concatenate([q[:, 3:7], q[:, 0:3], q[:, 7:][:, self.perm_mj_to_php]], axis=1)

    def php_to_mj(self, qpos_php: np.ndarray) -> np.ndarray:
        q = np.asarray(qpos_php, dtype=np.float64)
        return np.concatenate([q[:, 4:7], q[:, 0:4], q[:, 7:][:, self.perm_php_to_mj]], axis=1)

    # ---- forward kinematics --------------------------------------------------

    def forward(self, qpos_php_frame: np.ndarray) -> mujoco.MjData:
        q = self.php_to_mj(np.asarray(qpos_php_frame)[None])[0]
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        return self.data

    def foot_box_corners(self, side: str) -> np.ndarray:
        """(8, 3) world corners of the foot box (after :meth:`forward`)."""
        f = self.feet[side]
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        rot = self.data.geom_xmat[f.geom].reshape(3, 3)
        return self.data.geom_xpos[f.geom] + (signs * f.half) @ rot.T

    def foot_frame(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """(centre, R) of the foot box with R columns = forward, left, up of the sole."""
        f = self.feet[side]
        rot = self.data.geom_xmat[f.geom].reshape(3, 3)
        return self.data.geom_xpos[f.geom].copy(), rot.copy()

    def lowest_point(self) -> float:
        """Lowest world z over all robot collision geoms (geom AABB corners)."""
        m, d = self.model, self.data
        zmin = np.inf
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        for g in self.collision_geoms:
            aabb = m.geom_aabb[g]  # centre(3), half(3) in geom frame
            pts = aabb[:3] + signs * aabb[3:]
            rot = d.geom_xmat[g].reshape(3, 3)
            z = (d.geom_xpos[g] + pts @ rot.T)[:, 2].min()
            zmin = min(zmin, z)
        return float(zmin)

    def clip_fk(self, qpos_php: np.ndarray) -> dict[str, np.ndarray]:
        """Per-frame foot boxes, foot frames, lowest point and pelvis pose."""
        T = len(qpos_php)
        out = {
            "lowest": np.zeros(T),
            "pelvis": np.zeros((T, 3)),
        }
        for s in SIDES:
            out[f"{s}_corners"] = np.zeros((T, 8, 3))
            out[f"{s}_center"] = np.zeros((T, 3))
            out[f"{s}_rot"] = np.zeros((T, 3, 3))
        for t in range(T):
            self.forward(qpos_php[t])
            out["lowest"][t] = self.lowest_point()
            out["pelvis"][t] = self.data.xpos[1]
            for s in SIDES:
                out[f"{s}_corners"][t] = self.foot_box_corners(s)
                c, r = self.foot_frame(s)
                out[f"{s}_center"][t] = c
                out[f"{s}_rot"][t] = r
        return out


def box_point_distance(center: np.ndarray, rot: np.ndarray, half: np.ndarray, p: np.ndarray) -> float:
    """Signed distance from point ``p`` to an oriented box (negative inside)."""
    local = rot.T @ (p - center)
    q = np.abs(local) - half
    outside = np.linalg.norm(np.maximum(q, 0.0))
    inside = min(np.max(q), 0.0)
    return float(outside + inside)


@functools.lru_cache(maxsize=1)
def kinematics() -> LupusKinematics:
    return LupusKinematics()
