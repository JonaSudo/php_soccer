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
