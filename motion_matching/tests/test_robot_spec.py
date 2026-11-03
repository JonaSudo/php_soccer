"""RobotSpec: G1 must reproduce the values that used to be hard-coded."""

from __future__ import annotations

import numpy as np
import pytest

from motion_matching.kinematics.robot_spec import (
    DEFAULT_ROBOT,
    G1_SPEC,
    RobotSpec,
    get_robot_spec,
)

# The former ``G1_JOINT_AXES`` constant from ``metadata/constants.py``.
_LEGACY_G1_JOINT_AXES = np.array(
    [
        [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0],
        [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0],
        [0, 0, 1], [1, 0, 0], [0, 1, 0],
        [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
    ],
    dtype=np.float32,
)


def test_default_is_g1() -> None:
    assert DEFAULT_ROBOT == "g1"
    assert get_robot_spec() is G1_SPEC
    assert get_robot_spec("g1") is G1_SPEC
    assert get_robot_spec(G1_SPEC) is G1_SPEC


def test_unknown_robot_raises() -> None:
    with pytest.raises(ValueError, match="Unknown robot"):
        get_robot_spec("nope")


def test_g1_matches_legacy_constants() -> None:
    # Former G1_BONE_LEFT_FOOT / G1_BONE_RIGHT_FOOT / G1_BONE_HIPS and the
    # hard-coded 29 DoF / (T, 36) qpos shape.
    assert G1_SPEC.bone_left_foot == 7
    assert G1_SPEC.bone_right_foot == 13
    assert G1_SPEC.bone_hips == 1
    assert G1_SPEC.num_dofs == 29
    assert G1_SPEC.qpos_dim == 36
    assert G1_SPEC.num_bones == 31


def test_g1_joint_axes_match_legacy_table_exactly() -> None:
    axes = G1_SPEC.joint_axes
    assert axes.dtype == np.float32
    assert axes.tobytes() == _LEGACY_G1_JOINT_AXES.tobytes()


def test_g1_joint_order() -> None:
    names = G1_SPEC.joint_names
    assert names[0] == "left_hip_pitch_joint"
    assert names[-1] == "right_wrist_yaw_joint"
    for n in G1_SPEC.mirror_negate_joints + G1_SPEC.sim_position_joints:
        assert n in names


def test_dof_from_bone_rotations_matches_legacy_formula() -> None:
    from motion_matching.utils import math_utils as math

    rng = np.random.default_rng(0)
    q = rng.normal(size=(G1_SPEC.num_bones, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    legacy = np.sum(math.quat_to_scaled_angle_axis(q[-29:]) * _LEGACY_G1_JOINT_AXES, axis=-1)
    assert G1_SPEC.dof_from_bone_rotations(q).tobytes() == legacy.tobytes()


def test_mirror_default_is_g1_set() -> None:
    from motion_matching.utils.data_utils import animation_mirror_xz

    rng = np.random.default_rng(1)
    dof = rng.normal(size=(5, 29))
    pos = rng.normal(size=(5, 3))
    rot = rng.normal(size=(5, 4))
    a = animation_mirror_xz(dof.copy(), pos.copy(), rot.copy(), G1_SPEC.joint_names)
    b = animation_mirror_xz(
        dof.copy(), pos.copy(), rot.copy(), G1_SPEC.joint_names, neg_joints=G1_SPEC.mirror_negate_joints
    )
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


_ROTATED_URDF = """<?xml version="1.0"?>
<robot name="toy">
  <link name="base"/>
  <link name="left_leg"/>
  <link name="right_leg"/>
  <joint name="left_hip" type="revolute">
    <parent link="base"/><child link="left_leg"/>
    <origin xyz="0 0.1 0" rpy="1.5707963267948966 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="right_hip" type="revolute">
    <parent link="base"/><child link="right_leg"/>
    <origin xyz="0 -0.1 0" rpy="0 1.5707963267948966 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
</robot>
"""


def _toy_spec(tmp_path, frame: str, num_dofs: int = 2) -> RobotSpec:
    urdf = tmp_path / f"toy_{frame}_{num_dofs}.urdf"
    urdf.write_text(_ROTATED_URDF)
    return RobotSpec(
        name="toy",
        urdf_path=str(urdf),
        num_dofs=num_dofs,
        left_foot_joint="left_hip",
        right_foot_joint="right_hip",
        sim_position_joints=("left_hip", "right_hip"),
        mirror_negate_joints=(),
        joint_axes_frame=frame,
    )


def test_axes_frames(tmp_path) -> None:
    local = _toy_spec(tmp_path, "local")
    np.testing.assert_allclose(local.joint_axes, [[0, 0, 1], [0, 0, 1]], atol=1e-6)
    zero = _toy_spec(tmp_path, "zero_pose")
    # rpy roll +90deg maps local z to world -y; pitch +90deg maps local z to world +x.
    np.testing.assert_allclose(zero.joint_axes, [[0, -1, 0], [1, 0, 0]], atol=1e-6)
    assert zero.bone_left_foot == 2 and zero.bone_right_foot == 3
    np.testing.assert_allclose(zero.skeleton.joint_bone_vectors, [[0, 0.1, 0], [0, -0.1, 0]], atol=1e-9)


def test_dof_count_mismatch_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="declares 3 DoFs"):
        _ = _toy_spec(tmp_path, "local", num_dofs=3).joint_axes


def test_bad_axes_frame_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="joint_axes_frame"):
        _toy_spec(tmp_path, "world")


_CHAIN_URDF = """<?xml version="1.0"?>
<robot name="chain">
  <link name="base"/><link name="l1"/><link name="l2"/><link name="l3"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="l1"/>
    <origin xyz="0 0.1 0" rpy="0.3 -0.7 1.1"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="l1"/><child link="l2"/>
    <origin xyz="0.05 0 -0.3" rpy="-1.2 0.4 0.2"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
  <joint name="j3" type="revolute">
    <parent link="l2"/><child link="l3"/>
    <origin xyz="0 0.02 -0.25" rpy="0.9 0 -0.5"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
</robot>
"""


def test_zero_pose_axes_reproduce_urdf_fk(tmp_path) -> None:
    """With rotated joint frames, the database bone model is exact in 'zero_pose' mode."""
    from motion_matching.kinematics import quat

    urdf = tmp_path / "chain.urdf"
    urdf.write_text(_CHAIN_URDF)
    spec = RobotSpec(
        name="chain",
        urdf_path=str(urdf),
        num_dofs=3,
        left_foot_joint="j3",
        right_foot_joint="j3",
        sim_position_joints=("j1",),
        mirror_negate_joints=(),
        joint_axes_frame="zero_pose",
    )
    skel = spec.skeleton
    q = np.array([0.4, -0.9, 1.3])

    # Database-style FK: base-aligned bone frames, offsets from the zero pose.
    lrot = quat.from_angle_axis(q, skel.joint_axes.astype(np.float64))
    grot = np.zeros((3, 4))
    gpos = np.zeros((3, 3))
    for i in range(3):
        p = skel.joint_parents[i]
        if p == -1:
            grot[i], gpos[i] = lrot[i], skel.joint_bone_vectors[i]
        else:
            grot[i] = quat.mul(grot[p], lrot[i])
            gpos[i] = gpos[p] + quat.mul_vec(grot[p], skel.joint_bone_vectors[i])
    # A point fixed in the last link, compared against the URDF's own FK.
    kin = spec.make_kinematics()
    upos, _ = kin.compute_joint_poses(np.zeros(3), np.array([1.0, 0, 0, 0]), q)
    np.testing.assert_allclose(gpos, upos, atol=1e-9)
    _, zero_rot = kin.compute_joint_poses(np.zeros(3), np.array([1.0, 0, 0, 0]), np.zeros(3))
    _, urot = kin.compute_joint_poses(np.zeros(3), np.array([1.0, 0, 0, 0]), q)
    # bone frame = link frame * inverse(link frame at zero pose)
    expected = quat.mul(urot, quat.inv(zero_rot))
    np.testing.assert_allclose(np.abs(np.sum(grot * expected, axis=-1)), 1.0, atol=1e-9)


# ---------------------------------------------------------------------------- LUPUS


def _db_fk(skel, q):
    """Database-style FK (base-aligned bone frames) for joint angles ``q``."""
    from motion_matching.kinematics import quat

    n = len(q)
    lrot = quat.from_angle_axis(q, skel.joint_axes.astype(np.float64))
    grot = np.zeros((n, 4))
    gpos = np.zeros((n, 3))
    for i in range(n):
        p = skel.joint_parents[i]
        if p == -1:
            grot[i], gpos[i] = lrot[i], skel.joint_bone_vectors[i]
        else:
            grot[i] = quat.mul(grot[p], lrot[i])
            gpos[i] = gpos[p] + quat.mul_vec(grot[p], skel.joint_bone_vectors[i])
    return gpos


def _lupus_random_q(spec, seed):
    kin = spec.make_kinematics()
    rng = np.random.default_rng(seed)
    lo = np.array([kin.urdf.joint_map[n].limit.lower for n in spec.joint_names])
    hi = np.array([kin.urdf.joint_map[n].limit.upper for n in spec.joint_names])
    return kin, rng.uniform(lo, hi)


def test_lupus_registered() -> None:
    from motion_matching.kinematics.robot_spec import LUPUS_SPEC

    spec = get_robot_spec("lupus")
    assert spec is LUPUS_SPEC
    assert spec.num_dofs == 22 and spec.qpos_dim == 29 and spec.num_bones == 24
    assert spec.joint_axes_frame == "zero_pose"
    names = spec.joint_names
    for n in spec.mirror_negate_joints + spec.sim_position_joints + (spec.left_foot_joint, spec.right_foot_joint):
        assert n in names
    assert spec.bone_left_foot != spec.bone_right_foot
    # every axis is a base axis after the zero-pose rotation (CAD joints are axis-aligned)
    axes = spec.joint_axes
    np.testing.assert_allclose(np.abs(axes).max(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(axes, axis=1), 1.0, atol=1e-6)


def test_lupus_bone_model_matches_urdf_fk() -> None:
    spec = get_robot_spec("lupus")
    kin, q = _lupus_random_q(spec, 3)
    upos, _ = kin.compute_joint_poses(np.zeros(3), np.array([1.0, 0, 0, 0]), q)
    np.testing.assert_allclose(_db_fk(spec.skeleton, q), upos, atol=1e-9)


def test_lupus_mirror_rules_mirror_the_pose() -> None:
    """Mirroring the joint angles must mirror every joint position across the x-z plane."""
    from motion_matching.utils.data_utils import animation_mirror_xz

    spec = get_robot_spec("lupus")
    kin, q = _lupus_random_q(spec, 4)
    names = spec.joint_names
    dof_m, _, _ = animation_mirror_xz(
        q[None].copy(), np.zeros((1, 3)), np.array([[1.0, 0, 0, 0]]), names,
        neg_joints=spec.mirror_negate_joints, left_prefix=spec.left_prefix, right_prefix=spec.right_prefix,
    )
    root = (np.zeros(3), np.array([1.0, 0, 0, 0]))
    pos, _ = kin.compute_joint_poses(*root, q)
    pos_m, _ = kin.compute_joint_poses(*root, dof_m[0])
    partner = [
        names.index(n.replace("left_", "right_", 1) if n.startswith("left_") else n.replace("right_", "left_", 1))
        for n in names
    ]
    expected = pos[partner] * np.array([1.0, -1.0, 1.0])
    np.testing.assert_allclose(pos_m, expected, atol=1e-6)


def test_lupus_limits_are_mirror_symmetric() -> None:
    spec = get_robot_spec("lupus")
    kin = spec.make_kinematics()
    lim = {n: (kin.urdf.joint_map[n].limit.lower, kin.urdf.joint_map[n].limit.upper) for n in spec.joint_names}
    for n in spec.joint_names:
        if not n.startswith("left_"):
            continue
        lo, hi = lim[n.replace("left_", "right_", 1)]
        expected = (-hi, -lo) if n in spec.mirror_negate_joints else (lo, hi)
        assert lim[n] == pytest.approx(expected), n
