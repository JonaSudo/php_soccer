#!/usr/bin/env python3
"""Build the simulation URDF for LUPUS from the CAD export and hardware.yaml.

    python robots/lupus/build_urdf.py [--mesh-dir robots/lupus/meshes]

Reads source/robot_model.urdf (unchanged CAD export) and writes
  lupus.urdf         motor limits, fixed ankle motor joints, neck, mass fixes
  lupus_joints.yaml  per-joint limits, effort, velocity and armature for the
                     holosoma robot config and the MJCF (URDF has no armature)
Meshes are needed to compute masses and inertias of parts that the export left
without material; they are referenced as meshes/<file> in the output.
"""

from __future__ import annotations

import argparse
import math
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
RPM = 2 * math.pi / 60


# --------------------------------------------------------------------------- geometry
def rpy_to_matrix(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def matrix_to_rpy(m: np.ndarray) -> tuple[float, float, float]:
    p = math.asin(max(-1.0, min(1.0, -m[2, 0])))
    if abs(math.cos(p)) > 1e-9:
        return math.atan2(m[2, 1], m[2, 2]), p, math.atan2(m[1, 0], m[0, 0])
    return math.atan2(-m[1, 2], m[1, 1]), p, 0.0


def origin_matrix(el: ET.Element | None) -> np.ndarray:
    t = np.eye(4)
    if el is not None:
        t[:3, :3] = rpy_to_matrix(*map(float, el.get("rpy", "0 0 0").split()))
        t[:3, 3] = list(map(float, el.get("xyz", "0 0 0").split()))
    return t


def load_stl(path: Path) -> np.ndarray:
    """Triangles (N, 3, 3) of a binary or ASCII STL."""
    data = path.read_bytes()
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if 84 + 50 * n == len(data):
            rec = np.frombuffer(data[84:], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]), count=n)
            return rec["v"].reshape(-1, 3, 3).astype(float)
    verts = [list(map(float, line.split()[1:4])) for line in data.decode(errors="ignore").splitlines() if line.strip().startswith("vertex")]
    return np.array(verts, dtype=float).reshape(-1, 3, 3)


def mesh_mass_properties(tris: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Volume, centroid and inertia about the centroid per unit density (closed mesh)."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    det = np.einsum("ij,ij->i", a, np.cross(b, c))
    vol = det.sum() / 6.0
    if vol < 0:  # inward normals
        det, vol = -det, -vol
    com = (det[:, None] * (a + b + c)).sum(0) / (24.0 * vol)
    # second moments  integral x_i x_j dV over each tetrahedron (origin, a, b, c)
    s = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            s[i, j] = (
                det
                * (
                    2 * (a[:, i] * a[:, j] + b[:, i] * b[:, j] + c[:, i] * c[:, j])
                    + a[:, i] * b[:, j] + a[:, j] * b[:, i]
                    + a[:, i] * c[:, j] + a[:, j] * c[:, i]
                    + b[:, i] * c[:, j] + b[:, j] * c[:, i]
                )
            ).sum() / 120.0
    s -= vol * np.outer(com, com)
    inertia = np.trace(s) * np.eye(3) - s
    return vol, com, inertia


def box_inertia(mass: float, size: list[float]) -> np.ndarray:
    x, y, z = size
    return mass / 12.0 * np.diag([y * y + z * z, x * x + z * z, x * x + y * y])


# --------------------------------------------------------------------------- urdf edits
def fmt(v: float) -> str:
    return repr(float(v))


def set_inertial(link: ET.Element, mass: float, com: np.ndarray, inertia: np.ndarray) -> None:
    old = link.find("inertial")
    if old is not None:
        link.remove(old)
    inertial = ET.Element("inertial")
    ET.SubElement(inertial, "origin", xyz=" ".join(map(fmt, com)), rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=fmt(mass))
    i = inertia
    ET.SubElement(
        inertial, "inertia",
        ixx=fmt(i[0, 0]), ixy=fmt(i[0, 1]), ixz=fmt(i[0, 2]), iyy=fmt(i[1, 1]), iyz=fmt(i[1, 2]), izz=fmt(i[2, 2]),
    )
    link.insert(0, inertial)


def apply_mass_fixes(robot: ET.Element, fixes: list[dict], mesh_dir: Path) -> list[str]:
    log = []
    for link in robot.findall("link"):
        visuals = link.findall("visual")
        if len(visuals) != 1 or visuals[0].find("geometry/mesh") is None:
            continue
        fname = visuals[0].find("geometry/mesh").get("filename")
        fix = next((f for f in fixes if f["mesh"] in fname), None)
        if fix is None:
            continue
        old_mass = float(link.find("inertial/mass").get("value"))
        if "scale" in fix:
            k = fix["scale"]
            link.find("inertial/mass").set("value", fmt(old_mass * k))
            el = link.find("inertial/inertia")
            for key in el.attrib:
                el.set(key, fmt(float(el.get(key)) * k))
            new_mass = old_mass * k
        else:
            vol, com, inertia_unit = mesh_mass_properties(load_stl(mesh_dir / fname))
            new_mass = fix["mass"] if "mass" in fix else fix["density"] * 1000.0 * vol
            t = origin_matrix(visuals[0].find("origin"))
            r = t[:3, :3]
            set_inertial(link, new_mass, r @ com + t[:3, 3], r @ (inertia_unit * new_mass / vol) @ r.T)
        log.append(f"{link.get('name')}: {old_mass:.4g} -> {new_mass:.4f} kg ({fix.get('note', '')})")
    return log


def joint_specs(hw: dict) -> dict[str, dict]:
    """Effort [N*m], velocity [rad/s], armature [kg*m^2] and limits per actuated joint."""
    specs = {}
    lk = hw["ankle_linkage"]
    for name, j in hw["joints"].items():
        if j["motor"] == "ankle_linkage":
            m = hw["motors"][lk["motor"]]
            ratio = (lk["pitch_lever_d"] if j["axis"] == "pitch" else lk["roll_lever_w"]) / lk["crank_radius_r"]
            effort = 2 * m["peak_torque"] * ratio
            velocity = m["no_load_speed_rpm"] * RPM / ratio
            armature = 2 * m["rotor_inertia"] * m["gear_ratio"] ** 2 * ratio**2
        else:
            m = hw["motors"][j["motor"]]
            effort = m["peak_torque"]
            velocity = m["no_load_speed_rpm"] * RPM
            armature = m["rotor_inertia"] * m["gear_ratio"] ** 2
        specs[name] = dict(lower=j["lower"], upper=j["upper"], effort=effort, velocity=velocity, armature=armature, status=j["status"])
    neck = hw["neck"]
    for name in ("head_yaw", "head_pitch"):
        n = neck[name]
        specs[name] = dict(lower=n["lower"], upper=n["upper"], effort=n["effort"], velocity=n["velocity"], armature=n["armature"], status=neck["status"])
    return specs


def world_transforms(robot: ET.Element) -> dict[str, np.ndarray]:
    by_child = {j.find("child").get("link"): j for j in robot.findall("joint")}
    cache: dict[str, np.ndarray] = {}

    def world(link: str) -> np.ndarray:
        if link not in cache:
            j = by_child.get(link)
            cache[link] = np.eye(4) if j is None else world(j.find("parent").get("link")) @ origin_matrix(j.find("origin"))
        return cache[link]

    return {l.get("name"): world(l.get("name")) for l in robot.findall("link")}


def add_neck(robot: ET.Element, neck: dict, specs: dict) -> None:
    t_parent = world_transforms(robot)[neck["parent_link"]]
    t_yaw = np.eye(4)
    t_yaw[:3, 3] = neck["yaw_origin_base"]
    rel = np.linalg.inv(t_parent) @ t_yaw  # joint frame aligned with the base frame

    def link(name: str, cfg: dict) -> None:
        el = ET.SubElement(robot, "link", name=name)
        set_inertial(el, cfg["mass"], np.array(cfg.get("com", [0, 0, cfg["size"][2] / 2])), box_inertia(cfg["mass"], cfg["size"]))
        for tag in ("visual", "collision"):
            g = ET.SubElement(el, tag)
            ET.SubElement(g, "origin", xyz=" ".join(map(fmt, cfg.get("com", [0, 0, cfg["size"][2] / 2]))), rpy="0 0 0")
            ET.SubElement(ET.SubElement(g, "geometry"), "box", size=" ".join(map(fmt, cfg["size"])))

    def joint(name: str, parent: str, child: str, xyz, rpy, axis: str) -> None:
        s = specs[name]
        el = ET.SubElement(robot, "joint", name=name, type="revolute")
        ET.SubElement(el, "parent", link=parent)
        ET.SubElement(el, "child", link=child)
        ET.SubElement(el, "origin", xyz=" ".join(map(fmt, xyz)), rpy=" ".join(map(fmt, rpy)))
        ET.SubElement(el, "axis", xyz=axis)
        ET.SubElement(el, "limit", lower=fmt(s["lower"]), upper=fmt(s["upper"]), effort=fmt(s["effort"]), velocity=fmt(s["velocity"]))

    link("neck_link", neck["neck_link"])
    link("head_link", neck["head_link"])
    joint("head_yaw", neck["parent_link"], "neck_link", rel[:3, 3], matrix_to_rpy(rel[:3, :3]), "0 0 1")
    joint("head_pitch", "neck_link", "head_link", neck["pitch_offset"], (0, 0, 0), "0 1 0")


def build(source: Path, hw: dict, mesh_dir: Path) -> tuple[ET.ElementTree, dict, list[str]]:
    tree = ET.parse(source)
    robot = tree.getroot()
    robot.set("name", "lupus")
    log = apply_mass_fixes(robot, hw["mass_fixes"], mesh_dir)
    specs = joint_specs(hw)

    fixed = set(hw["ankle_linkage"]["motor_joints"])
    for j in robot.findall("joint"):
        name = j.get("name")
        if name in fixed:
            j.set("type", "fixed")
            for tag in ("axis", "limit"):
                if j.find(tag) is not None:
                    j.remove(j.find(tag))
        elif j.get("type") != "fixed":
            if name not in specs:
                raise KeyError(f"joint {name} missing in hardware.yaml")
            s = specs[name]
            lim = j.find("limit")
            lim.attrib.update(lower=fmt(s["lower"]), upper=fmt(s["upper"]), effort=fmt(s["effort"]), velocity=fmt(s["velocity"]))
    add_neck(robot, hw["neck"], specs)

    for mesh in robot.iter("mesh"):
        fname = mesh.get("filename")
        if "/" not in fname:
            mesh.set("filename", f"meshes/{fname}")
    ET.indent(tree, "  ")
    return tree, specs, log


def total_mass(robot: ET.Element) -> float:
    return sum(float(m.get("value")) for m in robot.iter("mass"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh-dir", type=Path, default=HERE / "meshes")
    ap.add_argument("--hardware", type=Path, default=HERE / "hardware.yaml")
    ap.add_argument("--source", type=Path, default=HERE / "source" / "robot_model.urdf")
    ap.add_argument("--out-dir", type=Path, default=HERE)
    args = ap.parse_args()

    hw = yaml.safe_load(args.hardware.read_text())
    before = total_mass(ET.parse(args.source).getroot())
    tree, specs, log = build(args.source, hw, args.mesh_dir)
    after = total_mass(tree.getroot())

    out = args.out_dir / "lupus.urdf"
    header = "<!-- Generated by robots/lupus/build_urdf.py from source/robot_model.urdf and hardware.yaml. Do not edit. -->\n"
    out.write_text('<?xml version="1.0"?>\n' + header + ET.tostring(tree.getroot(), encoding="unicode") + "\n")
    joints_out = {
        "generated_by": "robots/lupus/build_urdf.py",
        "total_mass_kg": round(after, 3),
        "joints": {k: {kk: (round(v, 5) if isinstance(v, float) else v) for kk, v in s.items()} for k, s in specs.items()},
    }
    (args.out_dir / "lupus_joints.yaml").write_text(yaml.safe_dump(joints_out, sort_keys=False))

    print("\n".join(log))
    print(f"\nmass: export {before:.3f} kg -> model {after:.3f} kg")
    if hw.get("measured_mass"):
        print(f"scale {hw['measured_mass']:.3f} kg, gap {hw['measured_mass'] - after:+.3f} kg")
    print(f"wrote {out} and lupus_joints.yaml")


if __name__ == "__main__":
    main()
