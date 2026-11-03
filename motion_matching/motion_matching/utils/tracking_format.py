"""Convert qpos motion to the FAR-tracking npz schema.

The schema (matching ``tracking_motion_animator.py``):

    fps:            (1,) int64
    joint_pos:      (N, 29) float32
    joint_vel:      (N, 29) float32
    body_pos_w:     (N, 30, 3) float32  -- world body positions
    body_quat_w:    (N, 30, 4) float32  -- world body quats (wxyz)
    body_lin_vel_w: (N, 30, 3) float32
    body_ang_vel_w: (N, 30, 3) float32

Ported from ``tracking_utils/interaction_mesh_to_sim_data.py`` in the sibling
``motion_matching`` repo. Joint/body remapping constants are reproduced
verbatim so the output stays bit-compatible with the existing conversion
pipeline.

That layout is G1's (29 joints, 30 bodies, IsaacLab order). Other robots
(``robot="lupus"``) use their MJCF directly: joints and bodies in MJCF order
(= the holosoma robot config order), every body except ``world``, and the
file additionally carries ``joint_names`` / ``body_names`` so the holosoma
converter can work by name.
"""

from __future__ import annotations

import numpy as np
import mujoco as mj

from dataclasses import dataclass

from motion_matching.utils import G1_MJCF


# Reorderings matching tracking_utils/interaction_mesh_to_sim_data.py:190-198.
_JOINT_MAPPING = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24,
     18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int64,
)
_BODY_MAPPING = np.array(
    [0, 1, 7, 13, 2, 8, 14, 3, 9, 15, 4, 10, 16, 23, 5, 11, 17, 24, 6, 12, 18,
     25, 19, 26, 20, 27, 21, 28, 22, 29],
    dtype=np.int64,
)
_EXCLUDED_BODIES = ("world", "left_toe_link", "right_toe_link")
_EXPECTED_BODIES = 30
_EXPECTED_JOINTS = 29



@dataclass
class _Layout:
    """FK model plus the output ordering for one robot."""

    model: mj.MjModel
    body_filter: np.ndarray          # MuJoCo body ids written, before body_mapping
    num_joints: int
    qpos_to_mjcf: np.ndarray | None  # motion-matching DoF order -> MJCF qpos order
    joint_mapping: np.ndarray | None
    body_mapping: np.ndarray | None
    joint_names: list[str] | None    # written to the npz when set
    body_names: list[str] | None


_layouts: dict[str, _Layout] = {}


def _g1_layout() -> _Layout:
    model = mj.MjModel.from_xml_path(G1_MJCF)
    body_ids = [
        i for i in range(model.nbody)
        if mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) not in _EXCLUDED_BODIES
    ]
    if len(body_ids) != _EXPECTED_BODIES:
        raise RuntimeError(
            f"G1 MJCF produced {len(body_ids)} bodies after filtering, "
            f"expected {_EXPECTED_BODIES}"
        )
    return _Layout(model, np.array(body_ids, dtype=np.int64), _EXPECTED_JOINTS,
                   None, _JOINT_MAPPING, _BODY_MAPPING, None, None)


def _mjcf_layout(robot: str) -> _Layout:
    from motion_matching.kinematics.robot_spec import get_robot_spec

    spec = get_robot_spec(robot)
    if spec.mjcf_path is None:
        raise ValueError(f"robot {robot!r} has no MJCF for tracking export")
    model = mj.MjModel.from_xml_path(str(spec.mjcf_path))
    if model.jnt_type[0] != mj.mjtJoint.mjJNT_FREE:
        raise ValueError(f"{spec.mjcf_path}: first joint must be the free root joint")
    mj_joints = [model.joint(i).name for i in range(1, model.njnt)]
    if sorted(mj_joints) != sorted(spec.joint_names):
        raise ValueError(f"{spec.mjcf_path} joints {mj_joints} != RobotSpec joints {spec.joint_names}")
    qpos_to_mjcf = np.array([spec.joint_names.index(n) for n in mj_joints], dtype=np.int64)
    body_ids = np.arange(1, model.nbody, dtype=np.int64)
    body_names = [model.body(int(i)).name for i in body_ids]
    return _Layout(model, body_ids, len(mj_joints), qpos_to_mjcf, None, None, mj_joints, body_names)


def _get_layout(robot: str = "g1") -> _Layout:
    """Load (once per process) the FK model and ordering for ``robot``."""
    if robot not in _layouts:
        _layouts[robot] = _g1_layout() if robot == "g1" else _mjcf_layout(robot)
    return _layouts[robot]


def _angular_velocity(q1: np.ndarray, q2: np.ndarray, dt: float) -> np.ndarray:
    """Angular velocity from a pair of (wxyz) quaternions over ``dt``."""
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    q1_conj = np.array([q1[0], -q1[1], -q1[2], -q1[3]])
    w1, x1, y1, z1 = q2
    w2, x2, y2, z2 = q1_conj
    dq = np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])
    if abs(dq[0]) < 1e-6:
        return np.zeros(3)
    return (2.0 / dt) * dq[1:] / dq[0]


def qpos_to_tracking(qpos: np.ndarray, fps: int, robot: str = "g1") -> dict[str, np.ndarray]:
    """Convert ``(N, 7 + num_dofs)`` qpos to the FAR-tracking schema.

    ``qpos`` layout is ``[root_quat_wxyz, root_pos, dofs]`` (the
    motion-matching convention, DoFs in RobotSpec order). MuJoCo uses
    ``[root_pos, root_quat_wxyz, dofs]``, so the layout is swapped once before
    calling FK.
    """
    layout = _get_layout(robot)
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] != 7 + layout.num_joints:
        raise ValueError(
            f"qpos must be (N, {7 + layout.num_joints}), got {qpos.shape}"
        )

    model, body_filter = layout.model, layout.body_filter
    data = mj.MjData(model)
    num_frames = qpos.shape[0]
    num_bodies = body_filter.shape[0]

    dofs = qpos[:, 7:] if layout.qpos_to_mjcf is None else qpos[:, 7:][:, layout.qpos_to_mjcf]
    # [quat, pos, dof] -> MuJoCo's [pos, quat, dof].
    mj_qpos = np.concatenate([qpos[:, 4:7], qpos[:, :4], dofs], axis=1)

    joint_pos = dofs.copy()
    body_pos_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    body_quat_w = np.zeros((num_frames, num_bodies, 4), dtype=np.float32)

    for i in range(num_frames):
        data.qpos[:] = 0
        data.qpos[: mj_qpos.shape[1]] = mj_qpos[i]
        mj.mj_forward(model, data)
        body_pos_w[i] = data.xpos[body_filter]
        body_quat_w[i] = data.xquat[body_filter]

    dt = 1.0 / fps
    joint_vel = np.gradient(joint_pos, dt, axis=0).astype(np.float32)
    body_lin_vel_w = np.gradient(body_pos_w, dt, axis=0).astype(np.float32)

    body_ang_vel_w = np.zeros_like(body_lin_vel_w)
    for i in range(num_frames):
        if i == 0:
            a, b, step = body_quat_w[0], body_quat_w[1], dt
        elif i == num_frames - 1:
            a, b, step = body_quat_w[i - 1], body_quat_w[i], dt
        else:
            a, b, step = body_quat_w[i - 1], body_quat_w[i + 1], 2.0 * dt
        for j in range(num_bodies):
            body_ang_vel_w[i, j] = _angular_velocity(a[j], b[j], step)

    if layout.joint_mapping is None:
        return {
            "fps": np.array([fps], dtype=np.int64),
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "body_pos_w": body_pos_w,
            "body_quat_w": body_quat_w,
            "body_lin_vel_w": body_lin_vel_w,
            "body_ang_vel_w": body_ang_vel_w,
            "joint_names": np.array(layout.joint_names),
            "body_names": np.array(layout.body_names),
        }
    return {
        "fps": np.array([fps], dtype=np.int64),
        "joint_pos": joint_pos[:, layout.joint_mapping],
        "joint_vel": joint_vel[:, layout.joint_mapping],
        "body_pos_w": body_pos_w[:, layout.body_mapping],
        "body_quat_w": body_quat_w[:, layout.body_mapping],
        "body_lin_vel_w": body_lin_vel_w[:, layout.body_mapping],
        "body_ang_vel_w": body_ang_vel_w[:, layout.body_mapping],
    }
