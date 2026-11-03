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


def apply_mass_target(robot: ET.Element, target: dict | None) -> list[str]:
    """Close the gap between the CAD mass and the real (or reference) mass.

    method "scale": every link mass and inertia times total/current (keeps the
    mass distribution). method "lump": the gap as one box on `link`.
    """
    if not target or target.get("total") is None:
        return []
    current = total_mass(robot)
    gap = target["total"] - current
    if target.get("method", "scale") == "scale":
        k = target["total"] / current
        for link in robot.findall("link"):
            m = link.find("inertial/mass")
            if m is None:
                continue
            m.set("value", fmt(float(m.get("value")) * k))
            el = link.find("inertial/inertia")
            for key in el.attrib:
                el.set(key, fmt(float(el.get(key)) * k))
        return [f"mass_target: all links x{k:.4f} ({current:.3f} -> {target['total']:.3f} kg, {target.get('source', '')})"]
    if gap <= 0:
        raise ValueError(f"mass_target lump needs a positive gap, model already has {current:.3f} kg")
    lump = ET.SubElement(robot, "link", name="unmodelled_mass")
    set_inertial(lump, gap, np.zeros(3), box_inertia(gap, target["size"]))
    t = np.linalg.inv(world_transforms(robot)[target["link"]])
    t = t @ np.vstack([np.hstack([np.eye(3), np.array(target["position_base"])[:, None]]), [0, 0, 0, 1]])
    j = ET.SubElement(robot, "joint", name="unmodelled_mass_joint", type="fixed")
    ET.SubElement(j, "parent", link=target["link"])
    ET.SubElement(j, "child", link="unmodelled_mass")
    ET.SubElement(j, "origin", xyz=" ".join(map(fmt, t[:3, 3])), rpy=" ".join(map(fmt, matrix_to_rpy(t[:3, :3]))))
    return [f"mass_target: +{gap:.3f} kg lumped on {target['link']} ({target.get('source', '')})"]


# --------------------------------------------------------------------------- collisions
def segments(robot: ET.Element) -> dict[str, str]:
    """Rigid segment of every link: the child link of the nearest movable joint above it (or the root).

    A fixed joint marked ``dont_collapse="true"`` (foot contact points) also
    starts a segment, like in holosoma's T1 URDF."""
    by_child = {j.find("child").get("link"): j for j in robot.findall("joint")}
    out: dict[str, str] = {}

    def seg(link: str) -> str:
        if link not in out:
            j = by_child.get(link)
            keep = j is None or j.get("type") != "fixed" or j.get("dont_collapse") == "true"
            out[link] = link if keep else seg(j.find("parent").get("link"))
        return out[link]

    for l in robot.findall("link"):
        seg(l.get("name"))
    return out


def fit_box(v: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Axis-aligned box (in the frame of ``v``): rotation, center, size."""
    lo, hi = v.min(0), v.max(0)
    return np.eye(3), (lo + hi) / 2, list(hi - lo)


def fit_cylinder(v: np.ndarray, radius_quantile: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Cylinder along the main axis of ``v``: rotation (z = axis), center, radius, length."""
    c0 = v.mean(0)
    _, _, vt = np.linalg.svd(v - c0, full_matrices=False)
    a = vt[0] / np.linalg.norm(vt[0])
    t = (v - c0) @ a
    center = c0 + a * (t.min() + t.max()) / 2
    radial = np.linalg.norm((v - center) - np.outer((v - center) @ a, a), axis=1)
    radius = float(np.quantile(radial, radius_quantile))
    # rotation whose z column is a
    x = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    r = np.column_stack([x, np.cross(a, x), a])
    return r, center, radius, float(t.max() - t.min())


def simplify_collisions(robot: ET.Element, cfg: dict | None, mesh_dir: Path) -> list[str]:
    """Replace the CAD collision meshes by one primitive per rigid segment.

    Links in ``boxes`` (feet, trunk) get the axis-aligned box of their meshes
    at the zero pose; a mesh filter restricts the box to matching meshes (the
    foot box covers only the sole plate, so the sole and the ball contact
    surface keep their real size). All other segments get a cylinder along the main axis of
    their meshes (written as cylinder in the URDF, capsule in the MJCF).
    Segments that already have primitive collisions (neck, head) keep them.
    """
    if not cfg or cfg.get("mode", "mesh") == "mesh":
        return []
    world = world_transforms(robot)
    seg_of = segments(robot)
    links = {l.get("name"): l for l in robot.findall("link")}
    boxes = cfg.get("boxes", {})
    verts: dict[str, list[np.ndarray]] = {}
    for name, link in links.items():
        for col in link.findall("collision"):
            mesh = col.find("geometry/mesh")
            if mesh is None:
                continue
            link.remove(col)
            fname = mesh.get("filename").removeprefix("meshes/")
            flt = boxes.get(seg_of[name])
            if flt and flt not in fname:
                continue
            tris = load_stl(mesh_dir / fname)
            t = world[name] @ origin_matrix(col.find("origin"))
            v = np.unique(tris.reshape(-1, 3), axis=0) @ t[:3, :3].T + t[:3, 3]
            verts.setdefault(seg_of[name], []).append(v)

    log = []
    for seg, vs in sorted(verts.items()):
        v = np.concatenate(vs)
        tl = world[seg]
        col = ET.SubElement(links[seg], "collision", name=f"{seg}_collision")
        if seg in boxes:
            r, c, size = fit_box(v)
            geom = ("box", {"size": " ".join(map(fmt, size))})
            desc = f"box {np.round(size, 3).tolist()}"
        else:
            r, c, radius, length = fit_cylinder(v, cfg.get("radius_quantile", 0.9))
            geom = ("cylinder", {"radius": fmt(radius), "length": fmt(length)})
            desc = f"cylinder r={radius:.3f} l={length:.3f}"
        rel_r = tl[:3, :3].T @ r
        rel_p = tl[:3, :3].T @ (c - tl[:3, 3])
        ET.SubElement(col, "origin", xyz=" ".join(map(fmt, rel_p)), rpy=" ".join(map(fmt, matrix_to_rpy(rel_r))))
        ET.SubElement(ET.SubElement(col, "geometry"), geom[0], **geom[1])
        log.append(f"collision {seg}: {desc}")
    return log


def add_foot_contact_points(robot: ET.Element, feet: list[str] | None) -> list[str]:
    """Add ``<side>_foot_contact_point`` links at the center of each sole.

    Holosoma's locomotion env measures foot height and contacts on these
    bodies (``foot_height_name``). They sit at the bottom center of the foot
    collision box, are fixed with ``dont_collapse="true"`` so the importers
    keep them, and carry a tiny mass plus an invisible visual (IsaacLab's USD
    conversion needs a visual on every link).
    """
    if not feet:
        return []
    world = world_transforms(robot)
    links = {l.get("name"): l for l in robot.findall("link")}
    log = []
    for foot in feet:
        col = links[foot].find("collision")
        size = np.array(list(map(float, col.find("geometry/box").get("size").split())))
        t_box = world[foot] @ origin_matrix(col.find("origin"))
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * size / 2
        z_min = (corners @ t_box[:3, :3].T + t_box[:3, 3])[:, 2].min()
        p_world = np.array([t_box[0, 3], t_box[1, 3], z_min])
        p_local = np.linalg.inv(world[foot]) @ np.append(p_world, 1.0)
        name = foot.replace("_foot_link", "_foot_contact_point")
        link = ET.SubElement(robot, "link", name=name)
        set_inertial(link, 0.001, np.zeros(3), np.eye(3) * 1e-6)
        vis = ET.SubElement(link, "visual")
        ET.SubElement(vis, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(ET.SubElement(vis, "geometry"), "sphere", radius="0.0001")
        j = ET.SubElement(robot, "joint", name=name, type="fixed", dont_collapse="true")
        ET.SubElement(j, "parent", link=foot)
        ET.SubElement(j, "child", link=name)
        # orientation aligned with the base frame at the zero pose
        r = world[foot][:3, :3].T
        ET.SubElement(j, "origin", xyz=" ".join(map(fmt, p_local[:3])), rpy=" ".join(map(fmt, matrix_to_rpy(r))))
        log.append(f"{name}: sole center {np.round(p_world, 4).tolist()} (CAD frame)")
    return log


# --------------------------------------------------------------------------- base frame
def reroot_base(robot: ET.Element, cfg: dict | None) -> list[str]:
    """Move the base_link origin from the CAD origin (ground) to the pelvis.

    ``cfg.origin`` is ``hip_pitch_midpoint`` or an explicit [x, y, z] in the
    CAD base frame. Only base_link's own elements and the joints below it
    change, so every other link keeps its pose relative to the robot.
    """
    if not cfg:
        return []
    base = next(l.get("name") for l in robot.findall("link") if l.get("name") not in {j.find("child").get("link") for j in robot.findall("joint")})
    origin = cfg["origin"]
    if origin == "hip_pitch_midpoint":
        joints = {j.get("name"): j for j in robot.findall("joint")}
        world = world_transforms(robot)
        pts = [world[joints[n].find("parent").get("link")] @ origin_matrix(joints[n].find("origin")) for n in ("left_hip_pitch", "right_hip_pitch")]
        origin = (pts[0][:3, 3] + pts[1][:3, 3]) / 2
    shift = np.eye(4)
    shift[:3, 3] = origin
    base_link = next(l for l in robot.findall("link") if l.get("name") == base)
    origins = [j.find("origin") for j in robot.findall("joint") if j.find("parent").get("link") == base]
    origins += [el.find("origin") for tag in ("inertial", "visual", "collision") for el in base_link.findall(tag)]
    for o in origins:
        t = np.linalg.inv(shift) @ origin_matrix(o)
        o.set("xyz", " ".join(map(fmt, t[:3, 3])))
        o.set("rpy", " ".join(map(fmt, matrix_to_rpy(t[:3, :3]))))
    return [f"base_link moved to {np.round(origin, 4).tolist()} (CAD frame)"]


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
    log += apply_mass_target(robot, hw.get("mass_target"))
    log += simplify_collisions(robot, hw.get("collision"), mesh_dir)
    log += add_foot_contact_points(robot, hw.get("foot_contact_points"))
    # last: every *_base coordinate in hardware.yaml is in the CAD base frame
    log += reroot_base(robot, hw.get("base_frame"))

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
    print(f"wrote {out} and lupus_joints.yaml")


if __name__ == "__main__":
    main()
