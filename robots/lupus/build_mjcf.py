#!/usr/bin/env python3
"""Build the MuJoCo model (MJCF) for LUPUS from lupus.urdf and lupus_joints.yaml.

    python robots/lupus/build_mjcf.py            # run after build_urdf.py

Writes lupus.xml next to the URDF, laid out like holosoma's
data/robots/t1/t1_29dof.xml:
  - free joint on base_link, spawned so the soles touch the ground
  - fixed links fused into their segment (one body per movable joint;
    done here, MuJoCo's fusestatic misplaces the merged centers of mass)
  - armature per joint from lupus_joints.yaml (URDF has none)
  - one torque motor per joint, ctrlrange = +-effort (PD runs in the policy loop)
  - collision primitives from the URDF (cylinders become capsules of the same
    overall length); robot geoms collide with the floor and other objects
    (e.g. the ball), not with each other
  - CAD meshes as visual-only geoms (group 2), loaded from meshes/
"""

from __future__ import annotations

import argparse
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent


def _load_build_urdf():
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_urdf", HERE / "build_urdf.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fuse_fixed_links(robot: ET.Element) -> None:
    """Merge every link into its rigid segment (child link of a movable joint).

    MuJoCo's own ``fusestatic`` puts the merged centers of mass in the wrong
    place for this URDF (MuJoCo 3.8), so the fusion is done here: inertias are
    combined with the parallel-axis theorem, visuals and collisions move into
    the segment frame, and movable joints are re-parented.
    """
    b = _load_build_urdf()
    world = b.world_transforms(robot)
    seg_of = b.segments(robot)
    links = {l.get("name"): l for l in robot.findall("link")}

    acc: dict[str, list] = {s: [0.0, np.zeros(3), np.zeros((3, 3))] for s in set(seg_of.values())}
    for name, link in links.items():
        seg = seg_of[name]
        t = np.linalg.inv(world[seg]) @ world[name]
        inertial = link.find("inertial")
        if inertial is not None:
            mass = float(inertial.find("mass").get("value"))
            ti = t @ b.origin_matrix(inertial.find("origin"))
            e = inertial.find("inertia")
            i_loc = np.array(
                [[float(e.get("ixx")), float(e.get("ixy")), float(e.get("ixz"))],
                 [float(e.get("ixy")), float(e.get("iyy")), float(e.get("iyz"))],
                 [float(e.get("ixz")), float(e.get("iyz")), float(e.get("izz"))]]
            )
            a = acc[seg]
            c = ti[:3, 3]
            # inertia about the segment origin, combined below
            a[0] += mass
            a[1] += mass * c
            a[2] += ti[:3, :3] @ i_loc @ ti[:3, :3].T + mass * (c @ c * np.eye(3) - np.outer(c, c))
        if name == seg:
            continue
        for tag in ("visual", "collision"):
            for el in link.findall(tag):
                to = t @ b.origin_matrix(el.find("origin"))
                o = el.find("origin")
                if o is None:
                    o = ET.SubElement(el, "origin")
                o.set("xyz", " ".join(map(b.fmt, to[:3, 3])))
                o.set("rpy", " ".join(map(b.fmt, b.matrix_to_rpy(to[:3, :3]))))
                links[seg].append(el)

    for j in list(robot.findall("joint")):
        parent, child = j.find("parent").get("link"), j.find("child").get("link")
        if j.get("type") == "fixed":
            robot.remove(j)
        elif seg_of[parent] != parent:
            t = np.linalg.inv(world[seg_of[parent]]) @ world[parent] @ b.origin_matrix(j.find("origin"))
            j.find("parent").set("link", seg_of[parent])
            j.find("origin").set("xyz", " ".join(map(b.fmt, t[:3, 3])))
            j.find("origin").set("rpy", " ".join(map(b.fmt, b.matrix_to_rpy(t[:3, :3]))))
    for name, link in links.items():
        if seg_of[name] != name:
            robot.remove(link)
    for seg, (mass, mc, i_origin) in acc.items():
        if mass <= 0:
            continue
        c = mc / mass
        i_com = i_origin - mass * (c @ c * np.eye(3) - np.outer(c, c))
        b.set_inertial(links[seg], mass, c, i_com)


def compile_urdf(urdf: Path, mesh_dir: Path) -> ET.Element:
    """Fuse the URDF to one link per segment, let MuJoCo parse it, return the saved MJCF."""
    import mujoco

    root = ET.parse(urdf).getroot()
    fuse_fixed_links(root)
    # floating joint -> freejoint, otherwise MuJoCo attaches base_link to the world
    ET.SubElement(root, "link", name="world")
    fj = ET.SubElement(root, "joint", name="root", type="floating")
    ET.SubElement(fj, "parent", link="world")
    ET.SubElement(fj, "child", link="base_link")
    ext = ET.SubElement(root, "mujoco")
    ET.SubElement(ext, "compiler", meshdir=str(mesh_dir.resolve()), strippath="true", fusestatic="false", discardvisual="false", balanceinertia="false", inertiafromgeom="false")
    for mesh in root.iter("mesh"):
        mesh.set("filename", Path(mesh.get("filename")).name)
    for tag in ("visual", "collision"):
        for el in root.iter(tag):
            el.attrib.pop("name", None)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "lupus.xml"
        mujoco.mj_saveLastXML(str(out), model)
        return ET.parse(out).getroot()


def build(urdf: Path, joints_yaml: Path, mesh_dir: Path) -> ET.Element:
    import mujoco

    joints = yaml.safe_load(joints_yaml.read_text())["joints"]
    m = compile_urdf(urdf, mesh_dir)
    m.set("model", "lupus")

    comp = m.find("compiler")
    comp.attrib = {"angle": "radian", "meshdir": "meshes/"}
    for mesh in m.iter("mesh"):
        if mesh.get("file"):
            mesh.set("file", Path(mesh.get("file")).name)

    # defaults like t1_29dof.xml
    for d in m.findall("default"):
        m.remove(d)
    default = ET.Element("default")
    ET.SubElement(default, "joint", limited="true", damping="0.01", frictionloss="0.0")
    vis = ET.SubElement(default, "default", {"class": "visual"})
    ET.SubElement(vis, "geom", type="mesh", contype="0", conaffinity="0", group="2", density="0")
    col = ET.SubElement(default, "default", {"class": "collision"})
    # contype 1 / conaffinity 0: no robot-robot contacts, but contacts with the
    # floor and objects that have conaffinity 1 (ball).
    ET.SubElement(col, "geom", group="3", contype="1", conaffinity="0", condim="3", friction="1.0 0.01 0.005", rgba="0.2 0.6 0.9 0.4")
    foot = ET.SubElement(col, "default", {"class": "foot"})
    ET.SubElement(foot, "geom", condim="3", friction="1.0 0.005 0.0001", solimp="0.9 0.95 0.001 0.5 2", solref="0.02 1")
    m.insert(1, default)

    world = m.find("worldbody")
    base = world.find("body")
    if base is None or base.get("name") != "base_link":
        raise RuntimeError("expected base_link as the only top-level body")

    for body in m.iter("body"):
        for g in body.findall("geom"):
            if g.get("contype") == "0" and g.get("conaffinity") == "0":  # URDF <visual>
                keep = {k: g.get(k) for k in ("type", "mesh", "size", "pos", "quat", "rgba") if g.get(k)}
                g.attrib = {"class": "visual", **keep}
            else:
                keep = {"name": f"{body.get('name')}_collision", **{k: g.get(k) for k in ("type", "size", "pos", "quat") if g.get(k)}}
                if keep.get("type") == "cylinder":
                    r, half = map(float, keep["size"].split())
                    keep["type"] = "capsule"
                    keep["size"] = f"{r:.6g} {max(half - r, 1e-4):.6g}"
                g.attrib = {"class": "foot" if body.get("name", "").endswith("foot_link") else "collision", **keep}
        for j in body.findall("joint"):
            name = j.get("name")
            if name not in joints:
                continue
            j.set("armature", f"{joints[name]['armature']:.6g}")
            j.set("actuatorfrcrange", f"{-joints[name]['effort']:.6g} {joints[name]['effort']:.6g}")
            j.attrib.pop("actuatorfrclimited", None)

    fj = base.find("freejoint")
    if fj is None:
        fj = base.find("joint[@type='free']")
        fj.attrib = {"name": "root"}
        fj.tag = "freejoint"

    act = ET.SubElement(m, "actuator")
    for j in m.iter("joint"):
        if j.get("name") in joints:
            e = joints[j.get("name")]["effort"]
            ET.SubElement(act, "motor", name=j.get("name"), joint=j.get("name"), ctrlrange=f"{-e:.6g} {e:.6g}", ctrllimited="true")

    # spawn height: lowest collision point at the zero pose touches z = 0
    model = mujoco.MjModel.from_xml_string(_with_abs_meshdir(m, mesh_dir))
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    lowest = min(_geom_lowest(model, data, i) for i in range(model.ngeom) if model.geom_contype[i])
    base.set("pos", f"0 0 {-lowest:.6g}")

    # scene like holosoma's mjcf scenes: floor + light
    asset = m.find("asset")
    ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker", rgb1="0.1 0.2 0.3", rgb2="0.2 0.3 0.4", width="512", height="512")
    ET.SubElement(asset, "material", name="grid", texture="grid", texrepeat="1 1", texuniform="true", reflectance="0.2")
    world.insert(0, ET.Element("light", name="spotlight", mode="targetbodycom", target="base_link", pos="0 -4 4"))
    world.insert(0, ET.Element("geom", name="floor", type="plane", size="0 0 0.05", material="grid", contype="1", conaffinity="1", condim="3"))
    ET.indent(m, "  ")
    return m


def _with_abs_meshdir(m: ET.Element, mesh_dir: Path) -> str:
    m2 = ET.fromstring(ET.tostring(m))
    m2.find("compiler").set("meshdir", str(mesh_dir.resolve()) + "/")
    return ET.tostring(m2, encoding="unicode")


def _geom_lowest(model, data, i: int) -> float:
    import mujoco

    t = model.geom_type[i]
    pos, rot, size = data.geom_xpos[i], data.geom_xmat[i].reshape(3, 3), model.geom_size[i]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * size
        return float((corners @ rot.T + pos)[:, 2].min())
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return float(pos[2] - abs(rot[2, 2]) * size[1] - size[0])
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(pos[2] - size[0])
    return float(pos[2] - model.geom_rbound[i])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", type=Path, default=HERE / "lupus.urdf")
    ap.add_argument("--joints", type=Path, default=HERE / "lupus_joints.yaml")
    ap.add_argument("--mesh-dir", type=Path, default=HERE / "meshes")
    ap.add_argument("--out", type=Path, default=HERE / "lupus.xml")
    args = ap.parse_args()
    m = build(args.urdf, args.joints, args.mesh_dir)
    header = "<!-- Generated by robots/lupus/build_mjcf.py from lupus.urdf and lupus_joints.yaml. Do not edit. -->\n"
    args.out.write_text(header + ET.tostring(m, encoding="unicode") + "\n")
    print(f"wrote {args.out} (base spawn height {m.find('worldbody/body').get('pos').split()[2]} m)")


if __name__ == "__main__":
    main()
