"""qpos_to_tracking for LUPUS: MJCF order, names in the file, RobotSpec DoF order on input."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from motion_matching.kinematics.robot_spec import get_robot_spec
from motion_matching.utils import LUPUS_DIR
from motion_matching.utils.tracking_format import qpos_to_tracking

pytestmark = pytest.mark.skipif(
    not (LUPUS_DIR / "meshes").exists(), reason="LUPUS CAD meshes not present"
)


def _mjcf():
    return mujoco.MjModel.from_xml_path(str(get_robot_spec("lupus").mjcf_path))


def _standing_qpos(n: int) -> np.ndarray:
    spec = get_robot_spec("lupus")
    qpos = np.zeros((n, 7 + spec.num_dofs), dtype=np.float32)
    qpos[:, 0] = 1.0  # identity quat (wxyz)
    qpos[:, 4] = np.linspace(0.0, 0.2, n)  # walk forward 0.2 m
    qpos[:, 6] = 0.5745  # base at hip-pitch midpoint above the sole
    return qpos


def test_layout_is_mjcf_order_with_names():
    m = _mjcf()
    t = qpos_to_tracking(_standing_qpos(6), 50, robot="lupus")
    joints = [m.joint(i).name for i in range(1, m.njnt)]
    bodies = [m.body(i).name for i in range(1, m.nbody)]
    assert t["joint_names"].tolist() == joints
    assert t["body_names"].tolist() == bodies
    assert t["joint_pos"].shape == (6, 22) and t["joint_vel"].shape == (6, 22)
    for key in ("body_pos_w", "body_lin_vel_w", "body_ang_vel_w"):
        assert t[key].shape == (6, len(bodies), 3)
    assert t["body_quat_w"].shape == (6, len(bodies), 4)
    assert t["fps"].tolist() == [50]


def test_dofs_are_permuted_from_robotspec_order():
    spec = get_robot_spec("lupus")
    qpos = _standing_qpos(4)
    for k, name in enumerate(spec.joint_names):
        qpos[:, 7 + k] = 0.01 * (k + 1)
    t = qpos_to_tracking(qpos, 50, robot="lupus")
    for col, name in enumerate(t["joint_names"].tolist()):
        expected = 0.01 * (spec.joint_names.index(name) + 1)
        np.testing.assert_allclose(t["joint_pos"][:, col], expected, rtol=1e-6)


def test_fk_matches_mujoco_and_velocities():
    m = _mjcf()
    d = mujoco.MjData(m)
    spec = get_robot_spec("lupus")
    qpos = _standing_qpos(10)
    knee = spec.joint_names.index("left_knee")
    qpos[:, 7 + knee] = np.linspace(0.0, 0.5, 10)
    t = qpos_to_tracking(qpos, 50, robot="lupus")
    names = t["body_names"].tolist()

    # base body = root, contact points on the floor in the zero pose
    np.testing.assert_allclose(t["body_pos_w"][:, 0], qpos[:, 4:7], atol=1e-6)
    sole = t["body_pos_w"][0, names.index("right_foot_contact_point"), 2]
    assert sole == pytest.approx(0.0, abs=2e-3)
    np.testing.assert_allclose(t["body_lin_vel_w"][:, 0, 0], 0.2 / 9 * 50, rtol=1e-4)

    # last frame against a direct MuJoCo forward pass
    d.qpos[:3] = qpos[-1, 4:7]
    d.qpos[3:7] = qpos[-1, :4]
    d.qpos[m.jnt_qposadr[m.joint("left_knee").id]] = 0.5
    mujoco.mj_kinematics(m, d)
    shin = names.index("left_shin_link")
    np.testing.assert_allclose(t["body_pos_w"][-1, shin], d.xpos[m.body("left_shin_link").id], atol=1e-5)
    col = t["joint_names"].tolist().index("left_knee")
    np.testing.assert_allclose(t["joint_vel"][3, col], 0.5 / 9 * 50, rtol=1e-4)


def test_wrong_width_is_rejected():
    with pytest.raises(ValueError, match=r"\(N, 29\)"):
        qpos_to_tracking(np.zeros((3, 36), dtype=np.float32), 50, robot="lupus")


def test_save_motion_npz_passes_robot(tmp_path):
    from motion_matching.io.motion_io import save_motion_npz

    out = tmp_path / "m.npz"
    save_motion_npz(_standing_qpos(30), str(out), target_fps=50, source_fps=50, robot="lupus")
    with np.load(out) as f:
        assert f["joint_pos"].shape[1] == 22
        assert "body_names" in f.files
