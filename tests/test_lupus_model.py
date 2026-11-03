"""Contracts of the generated LUPUS model (robots/lupus)."""

from __future__ import annotations

import importlib.util
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

LUPUS = Path(__file__).resolve().parents[1] / "robots/lupus"
SPEC = importlib.util.spec_from_file_location("build_urdf", LUPUS / "build_urdf.py")
build_urdf = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_urdf)

HW = yaml.safe_load((LUPUS / "hardware.yaml").read_text())
URDF = ET.parse(LUPUS / "lupus.urdf").getroot()
JOINTS = yaml.safe_load((LUPUS / "lupus_joints.yaml").read_text())["joints"]


def movable():
    return {j.get("name"): j for j in URDF.findall("joint") if j.get("type") != "fixed"}


def test_no_placeholder_limits_left():
    for name, j in movable().items():
        lim = j.find("limit")
        assert float(lim.get("effort")) < 100, name
        assert float(lim.get("velocity")) != 10, name
        assert not (float(lim.get("lower")) == -math.pi and float(lim.get("upper")) == math.pi), name


def test_joint_table_matches_urdf():
    assert set(movable()) == set(JOINTS)
    for name, j in movable().items():
        lim = j.find("limit")
        assert float(lim.get("effort")) == pytest.approx(JOINTS[name]["effort"], rel=1e-4)
        assert JOINTS[name]["armature"] > 0


def test_ankle_motor_joints_fixed_and_neck_added():
    types = {j.get("name"): j.get("type") for j in URDF.findall("joint")}
    for name in HW["ankle_linkage"]["motor_joints"]:
        assert types[name] == "fixed"
    assert types["head_yaw"] == types["head_pitch"] == "revolute"
    assert len(movable()) == 22


def test_motor_derived_values():
    specs = build_urdf.joint_specs(HW)
    assert specs["left_knee"]["effort"] == 34.0
    assert specs["left_knee"]["armature"] == pytest.approx(0.30e-4 * 36**2)
    assert specs["left_elbow"]["velocity"] == pytest.approx(178 * 2 * math.pi / 60)
    lk = HW["ankle_linkage"]
    ratio = lk["roll_lever_w"] / lk["crank_radius_r"]
    assert specs["left_ankle_roll"]["effort"] == pytest.approx(2 * 7.0 * ratio)


def test_mass_target_matches_reference():
    # + 2 g for the two foot contact points added after the mass target
    assert build_urdf.total_mass(URDF) == pytest.approx(HW["mass_target"]["total"] + 0.002, abs=1e-6)


def test_every_joint_names_its_source():
    for name, j in HW["joints"].items():
        assert j["status"] in {"cad", "k1_reference", "estimate", "measured"}, name


def test_mesh_inertia_of_unit_cube():
    corners = [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]
    faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
             (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    import numpy as np

    tris = np.array([[corners[i] for i in f] for f in faces], dtype=float)
    vol, com, inertia = build_urdf.mesh_mass_properties(tris)
    assert vol == pytest.approx(1.0)
    assert com == pytest.approx([0.5, 0.5, 0.5])
    assert np.diag(inertia) == pytest.approx([1 / 6] * 3)
    assert inertia[0, 1] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.skipif(not (LUPUS / "meshes").exists(), reason="CAD meshes not present")
def test_generated_files_are_up_to_date(tmp_path):
    tree, _, _ = build_urdf.build(LUPUS / "source/robot_model.urdf", HW, LUPUS / "meshes")
    assert build_urdf.total_mass(tree.getroot()) == pytest.approx(build_urdf.total_mass(URDF), abs=1e-6)


def test_collisions_are_primitives_with_real_sole():
    links = {l.get("name"): l for l in URDF.findall("link")}
    assert not [c for c in URDF.iter("collision") if c.find("geometry/mesh") is not None]
    for foot in ("left_foot_link", "right_foot_link"):
        box = links[foot].find("collision/geometry/box")
        size = sorted(map(float, box.get("size").split()))
        # sole plate "footlow" from the CAD: 0.30 x 0.11 x 0.03 m
        assert size == pytest.approx([0.03, 0.11, 0.30], abs=2e-3), foot


def test_base_link_sits_at_hip_pitch_midpoint():
    w = build_urdf.world_transforms(URDF)
    joints = {j.get("name"): j for j in URDF.findall("joint")}
    hips = [(w[joints[n].find("parent").get("link")] @ build_urdf.origin_matrix(joints[n].find("origin")))[:3, 3]
            for n in ("left_hip_pitch", "right_hip_pitch")]
    import numpy as np

    assert (hips[0] + hips[1]) / 2 == pytest.approx(np.zeros(3), abs=1e-9)


mujoco = pytest.importorskip("mujoco") if (LUPUS / "meshes").exists() else None


@pytest.mark.skipif(mujoco is None, reason="CAD meshes or mujoco not present")
def test_mjcf_mass_com_and_standing():
    import numpy as np

    m = mujoco.MjModel.from_xml_path(str(LUPUS / "lupus.xml"))
    d = mujoco.MjData(m)
    assert m.nu == 22 and m.nbody == 26  # world + 23 segments + 2 foot contact points
    assert m.body_subtreemass[1] == pytest.approx(HW["mass_target"]["total"], abs=3e-3)
    mujoco.mj_forward(m, d)
    # center of mass inside the sole, above the ankles
    feet = [d.geom_xpos[m.geom(f"{s}_foot_link_collision").id] for s in ("left", "right")]
    com = d.subtree_com[1]
    assert abs(com[0] - feet[0][0]) < 0.15 and abs(com[1]) < 0.01

    # PD hold of the zero pose for 2 s: stays upright, only the feet touch the floor.
    # Ankle stiffness must exceed m*g*h (~50 N*m/rad per ankle) or LUPUS tips over.
    qadr = [m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)]
    vadr = [m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in range(m.nu)]
    names = [m.joint(m.actuator_trnid[i, 0]).name for i in range(m.nu)]
    kp = np.array([100.0 if any(k in n for k in ("hip", "knee", "ankle")) else 20.0 for n in names])
    for _ in range(int(2.0 / m.opt.timestep)):
        d.ctrl[:] = np.clip(-kp * d.qpos[qadr] - kp / 20 * d.qvel[vadr], m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
        mujoco.mj_step(m, d)
    assert d.qpos[2] == pytest.approx(m.body_pos[1][2], abs=0.01)
    assert abs(d.qpos[3]) > 0.999  # quaternion w: no tilt
    touching = {m.body(m.geom_bodyid[g]).name for c in d.contact[: d.ncon] for g in (c.geom1, c.geom2)} - {"world"}
    contact_points = [d.xpos[m.body(f"{s}_foot_contact_point").id] for s in ("left", "right")]
    assert max(abs(p[2]) for p in contact_points) < 0.005  # contact points sit on the sole
    assert touching == {"left_foot_link", "right_foot_link"}
