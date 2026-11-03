"""Robot description for the motion-matching pipeline.

Everything that used to be hard-coded for the Unitree G1 (URDF path, foot and
hip bone indices, joint rotation axes, DoF count, mirror rules, the joints that
define the simulation bone) lives in a :class:`RobotSpec`. G1 stays the default
(:data:`G1_SPEC`); further robots are registered in :data:`ROBOT_SPECS` and
selected by name with :func:`get_robot_spec`.

Database bone layout (produced by ``generate_database``)::

    bone 0       Simulation (projected root trajectory)
    bone 1       URDF base link (the "hips")
    bone 2 + i   actuated joint ``i`` in URDF order (``kin.joint_names``)

so the last ``num_dofs`` bones are exactly the DoFs, and every joint-indexed
bone index is derived from the joint name instead of a magic number.

Joint axes
----------
``joint_axes_frame`` selects how per-joint rotation axes are expressed:

- ``"local"``: the raw ``<axis>`` of each URDF joint, ignoring the joint
  origin's ``rpy``. This is what the G1 databases were built with, and it is
  only accurate when joint frames are (nearly) aligned with the base frame.
- ``"zero_pose"``: the joint axis rotated into the base frame at the zero pose.
  The database models every bone frame as base-aligned at the zero pose, so
  this is the exact choice for URDFs whose joint frames are rotated (e.g. CAD
  exports where every axis is ``0 0 1``).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from motion_matching.utils import G1_MJCF, G1_URDF, LUPUS_MJCF, LUPUS_URDF

# Bone indices fixed by the database layout (see module docstring).
BONE_SIMULATION = 0
BONE_BASE = 1
FIRST_JOINT_BONE = 2

JointAxesFrame = Literal["local", "zero_pose"]


@dataclass(frozen=True)
class RobotSkeleton:
    """URDF-derived data for a :class:`RobotSpec`, computed once per process."""

    base_link: str
    joint_names: tuple[str, ...]
    # (num_dofs,) parent joint index, -1 for joints attached to the base link.
    joint_parents: np.ndarray
    # (num_dofs, 3) offset of each joint from its parent at the zero pose,
    # expressed in the base frame.
    joint_bone_vectors: np.ndarray
    # (num_dofs, 3) float32 rotation axis per joint (see ``joint_axes_frame``).
    joint_axes: np.ndarray


@dataclass(frozen=True)
class RobotSpec:
    """Static description of one robot for motion matching.

    Attributes:
        name: Short identifier, e.g. ``"g1"``.
        urdf_path: URDF used for kinematics and visualization.
        num_dofs: Number of actuated joints. qpos is ``(T, 7 + num_dofs)``
            ordered ``[root_quat_wxyz, root_pos, dofs]``.
        left_foot_joint / right_foot_joint: Actuated joints whose bones act as
            the feet (contacts, foot features, foot locking).
        sim_position_joints: Joints whose mean horizontal position defines the
            simulation bone (G1 uses the two shoulder-pitch joints).
        mirror_negate_joints: Joints whose angle flips sign when mirroring the
            motion across the sagittal (x-z) plane.
        left_prefix / right_prefix: Name prefixes used to pair left and right
            joints when mirroring.
        joint_axes_frame: ``"local"`` or ``"zero_pose"``; see module docstring.
        mjcf_path: Optional MJCF used by MuJoCo-based tools.
    """

    name: str
    urdf_path: str
    num_dofs: int
    left_foot_joint: str
    right_foot_joint: str
    sim_position_joints: tuple[str, ...]
    mirror_negate_joints: tuple[str, ...]
    left_prefix: str = "left"
    right_prefix: str = "right"
    joint_axes_frame: JointAxesFrame = "local"
    mjcf_path: str | None = None
    # Hidden cache key so two specs with the same URDF but different axes frame
    # never share a skeleton.
    _cache_key: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.joint_axes_frame not in ("local", "zero_pose"):
            raise ValueError(f"joint_axes_frame must be 'local' or 'zero_pose', got {self.joint_axes_frame!r}")
        object.__setattr__(self, "_cache_key", (self.urdf_path, self.joint_axes_frame))

    # ---- shapes -------------------------------------------------------------

    @property
    def qpos_dim(self) -> int:
        """Width of a qpos row: 4 (root quat) + 3 (root pos) + DoFs."""
        return 7 + self.num_dofs

    @property
    def num_bones(self) -> int:
        """Bones in a database built for this robot (simulation + base + DoFs)."""
        return FIRST_JOINT_BONE + self.num_dofs

    # ---- URDF-derived data ---------------------------------------------------

    @property
    def skeleton(self) -> RobotSkeleton:
        skel = _load_skeleton(*self._cache_key)
        if len(skel.joint_names) != self.num_dofs:
            raise ValueError(
                f"RobotSpec {self.name!r} declares {self.num_dofs} DoFs but {self.urdf_path} "
                f"has {len(skel.joint_names)} actuated joints"
            )
        return skel

    @property
    def joint_names(self) -> list[str]:
        return list(self.skeleton.joint_names)

    @property
    def joint_axes(self) -> np.ndarray:
        """``(num_dofs, 3)`` float32 rotation axes in actuated-joint order."""
        return self.skeleton.joint_axes

    def joint_bone(self, joint_name: str) -> int:
        """Database bone index of an actuated joint."""
        try:
            return FIRST_JOINT_BONE + self.skeleton.joint_names.index(joint_name)
        except ValueError:
            raise ValueError(f"{joint_name!r} is not an actuated joint of robot {self.name!r}") from None

    @property
    def bone_left_foot(self) -> int:
        return self.joint_bone(self.left_foot_joint)

    @property
    def bone_right_foot(self) -> int:
        return self.joint_bone(self.right_foot_joint)

    @property
    def bone_hips(self) -> int:
        return BONE_BASE

    def make_kinematics(self, *, load_meshes: bool = False, load_collision_meshes: bool = False, verbose: bool = False):
        """Construct a :class:`URDFKinematics` for this robot."""
        from motion_matching.kinematics.urdf_kinematics import URDFKinematics

        return URDFKinematics(
            urdf_path=self.urdf_path,
            load_meshes=load_meshes,
            load_collision_meshes=load_collision_meshes,
            verbose=verbose,
        )

    # ---- helpers ------------------------------------------------------------

    def dof_from_bone_rotations(self, bone_rotations: np.ndarray) -> np.ndarray:
        """Project per-bone local rotations (``(..., num_bones, 4)``) to joint angles."""
        from motion_matching.utils.math_utils import quat_to_scaled_angle_axis

        dof_aa = quat_to_scaled_angle_axis(bone_rotations[..., -self.num_dofs :, :])
        return np.sum(dof_aa * self.joint_axes, axis=-1)


@functools.lru_cache(maxsize=None)
def _load_skeleton(urdf_path: str, joint_axes_frame: str) -> RobotSkeleton:
    from motion_matching.kinematics.urdf_kinematics import URDFKinematics

    kin = URDFKinematics(urdf_path=urdf_path, load_meshes=False, load_collision_meshes=False)
    names = list(kin.joint_names)
    n = len(names)

    link_to_joint_idx = {kin.urdf.joint_map[name].child: i for i, name in enumerate(names)}
    parents = np.full(n, -1, dtype=np.int32)
    for i, name in enumerate(names):
        parent_link = kin.urdf.joint_map[name].parent
        if parent_link in link_to_joint_idx:
            parents[i] = link_to_joint_idx[parent_link]

    positions, orientations = kin.compute_joint_poses(
        root_position=np.zeros(3),
        root_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        joint_angles=np.zeros(n),
    )
    bone_vectors = np.zeros((n, 3))
    for i in range(n):
        p = parents[i]
        bone_vectors[i] = positions[i] - (positions[p] if p != -1 else 0.0)

    axes = np.zeros((n, 3))
    for i, name in enumerate(names):
        axis = getattr(kin.urdf.joint_map[name], "axis", None)
        if axis is not None:
            axes[i] = np.asarray(axis, dtype=np.float64)
    if joint_axes_frame == "zero_pose":
        from motion_matching.utils.math_utils import quat_mul_vec3

        axes = np.stack([quat_mul_vec3(np.asarray(orientations[i]), axes[i]) for i in range(n)])
    axes = axes.astype(np.float32)

    for arr in (parents, bone_vectors, axes):
        arr.setflags(write=False)
    return RobotSkeleton(
        base_link=kin.urdf.base_link,
        joint_names=tuple(names),
        joint_parents=parents,
        joint_bone_vectors=bone_vectors,
        joint_axes=axes,
    )


G1_SPEC = RobotSpec(
    name="g1",
    urdf_path=G1_URDF,
    mjcf_path=G1_MJCF,
    num_dofs=29,
    left_foot_joint="left_ankle_roll_joint",
    right_foot_joint="right_ankle_roll_joint",
    sim_position_joints=("left_shoulder_pitch_joint", "right_shoulder_pitch_joint"),
    mirror_negate_joints=(
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
        "left_wrist_roll_joint",
        "right_wrist_roll_joint",
        "left_wrist_yaw_joint",
        "right_wrist_yaw_joint",
    ),
    joint_axes_frame="local",
)

# LUPUS (robots/lupus). The CAD export gives every joint axis as local +z with
# rotated joint origins, so the axes must be taken in the base frame at the
# zero pose. In that frame left and right joints share the same axis (pitch +y,
# roll +x, yaw +z), so mirroring across the x-z plane flips every roll and yaw
# joint and keeps pitch, knee and elbow. The neck (head_yaw/head_pitch) is part
# of the skeleton (it moves the head mass) but not of the policy action space.
LUPUS_SPEC = RobotSpec(
    name="lupus",
    urdf_path=LUPUS_URDF,
    mjcf_path=LUPUS_MJCF,
    num_dofs=22,
    left_foot_joint="left_ankle_roll",
    right_foot_joint="right_ankle_roll",
    sim_position_joints=("left_shoulder_pitch", "right_shoulder_pitch"),
    mirror_negate_joints=(
        "left_hip_roll",
        "right_hip_roll",
        "left_hip_yaw",
        "right_hip_yaw",
        "left_ankle_roll",
        "right_ankle_roll",
        "left_shoulder_roll",
        "right_shoulder_roll",
        "left_shoulder_yaw",
        "right_shoulder_yaw",
        "head_yaw",
    ),
    joint_axes_frame="zero_pose",
)

ROBOT_SPECS: dict[str, RobotSpec] = {G1_SPEC.name: G1_SPEC, LUPUS_SPEC.name: LUPUS_SPEC}
DEFAULT_ROBOT = G1_SPEC.name


def get_robot_spec(robot: str | RobotSpec | None = None) -> RobotSpec:
    """Resolve a robot name (or pass a spec through). ``None`` means the default (G1)."""
    if isinstance(robot, RobotSpec):
        return robot
    name = DEFAULT_ROBOT if robot is None else robot
    try:
        return ROBOT_SPECS[name]
    except KeyError:
        raise ValueError(f"Unknown robot {name!r}; known: {sorted(ROBOT_SPECS)}") from None
