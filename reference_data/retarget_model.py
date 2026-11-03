"""Build the LUPUS models used for retargeting human clips.

The holosoma interaction-mesh retargeter needs a MuJoCo model whose ground
geom is called ``ground``, bodies for every mapped human joint (toe, hand and
sole-corner points the plain LUPUS model does not have) and, for kicks, a
scene with the ball as a free body. The URDF twin is only used for the
optional viser visualisation.

    python -m reference_data.retarget_model --out /tmp/lupus_retarget

Everything is derived from ``robots/lupus/lupus.xml`` and ``lupus.urdf`` (and
the marker settings in ``lupus_retarget.yaml``), so a rebuild of the robot
model carries over automatically. The output is a cache, not versioned.
"""

from __future__ import annotations

import argparse
import copy
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().parent / "lupus_retarget.yaml"

SIDES = ("left", "right")


def load_config(path: Path | str | None = None) -> dict:
    return yaml.safe_load(Path(path or CONFIG_PATH).read_text())


@dataclass(frozen=True)
class RetargetModel:
    xml: Path            # robot + ground
    urdf: Path           # robot + marker links (visualisation only)
    scene_xml: Path | None  # robot + ground + ball (object_interaction)
    foot_sticking_links: tuple[str, ...]
    joint_names: tuple[str, ...]  # actuated joints in MJCF (= retargeter qpos) order


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _geom_frame(geom: ET.Element) -> tuple[np.ndarray, np.ndarray]:
    pos = np.array([float(v) for v in geom.get("pos", "0 0 0").split()])
    quat = np.array([float(v) for v in geom.get("quat", "1 0 0 0").split()])
    return pos, _quat_to_mat(quat)


def _find_body(root: ET.Element, name: str) -> ET.Element:
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    raise KeyError(f"body {name!r} not in model")


def _collision_geom(body: ET.Element) -> ET.Element:
    for geom in body.findall("geom"):
        if geom.get("class") == "collision" or (geom.get("name") or "").endswith("_collision"):
            return geom
    raise KeyError(f"body {body.get('name')!r} has no collision geom")


def marker_offsets(mjcf_root: ET.Element, cfg: dict) -> dict[str, tuple[str, np.ndarray]]:
    """Marker name -> (parent body, position in the parent body frame)."""
    mk = cfg["markers"]
    inset = float(mk["sole_corner_inset"])
    toe_back = float(mk["toe_from_front"])
    out: dict[str, tuple[str, np.ndarray]] = {}
    for side in SIDES:
        foot = _find_body(mjcf_root, f"{side}_foot_link")
        geom = _collision_geom(foot)
        if geom.get("type") != "box":
            raise ValueError(f"{side} foot collision must be a box, got {geom.get('type')!r}")
        half = np.array([float(v) for v in geom.get("size").split()])
        pos, rot = _geom_frame(geom)
        # The sole is the box face whose normal points down in the zero pose;
        # the parent frames are rotated (CAD export), so find it from the
        # world orientation of the foot body.
        body_rot_world = _body_world_rotation(mjcf_root, foot)
        down_local = rot.T @ body_rot_world.T @ np.array([0.0, 0.0, -1.0])
        fwd_local = rot.T @ body_rot_world.T @ np.array([1.0, 0.0, 0.0])
        lat_local = rot.T @ body_rot_world.T @ np.array([0.0, 1.0, 0.0])
        a_down, a_fwd, a_lat = (int(np.argmax(np.abs(v))) for v in (down_local, fwd_local, lat_local))
        if len({a_down, a_fwd, a_lat}) != 3:
            raise ValueError(f"{side} foot box is not axis aligned with the world in the zero pose")

        def point(fwd: float, lat: float) -> np.ndarray:
            p = np.zeros(3)
            p[a_down] = np.sign(down_local[a_down]) * half[a_down]
            p[a_fwd] = np.sign(fwd_local[a_fwd]) * fwd
            p[a_lat] = np.sign(lat_local[a_lat]) * lat
            return pos + rot @ p

        hx, hy = half[a_fwd] - inset, half[a_lat] - inset
        for k, (fx, ly) in enumerate([(hx, hy), (hx, -hy), (-hx, hy), (-hx, -hy)], start=1):
            out[f"{side}_foot_sphere_{k}_link"] = (foot.get("name"), point(fx, ly))
        out[f"{side}_toe_marker"] = (foot.get("name"), point(half[a_fwd] - toe_back, 0.0))

        forearm = _find_body(mjcf_root, f"{side}_forearm_link")
        geom = _collision_geom(forearm)
        if geom.get("type") != "capsule":
            raise ValueError(f"{side} forearm collision must be a capsule, got {geom.get('type')!r}")
        half_len = float(geom.get("size").split()[1])
        pos, rot = _geom_frame(geom)
        ends = [pos + rot @ np.array([0.0, 0.0, s * half_len]) for s in (-1.0, 1.0)]
        out[f"{side}_hand_marker"] = (forearm.get("name"), max(ends, key=np.linalg.norm))

    head = _find_body(mjcf_root, "head_link")
    pos, _ = _geom_frame(_collision_geom(head))
    out["head_marker"] = (head.get("name"), pos)
    return out


def _body_world_rotation(root: ET.Element, body: ET.Element) -> np.ndarray:
    parents = {c: p for p in root.iter() for c in p}
    rot = np.eye(3)
    node = body
    chain = []
    while node is not None and node.tag == "body":
        chain.append(node)
        node = parents.get(node)
    for b in reversed(chain):
        q = np.array([float(v) for v in b.get("quat", "1 0 0 0").split()])
        rot = rot @ _quat_to_mat(q)
    return rot


def _fmt(v: np.ndarray) -> str:
    return " ".join(f"{x:.6g}" for x in v)


def _retarget_mjcf(cfg: dict, markers: dict[str, tuple[str, np.ndarray]]) -> ET.ElementTree:
    tree = ET.parse(REPO / cfg["robot"]["mjcf"])
    root = tree.getroot()
    # Retargeting only needs kinematics and collision primitives.
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "geom" and child.get("class") == "visual":
                parent.remove(child)
    asset = root.find("asset")
    for mesh in list(asset.findall("mesh")):
        asset.remove(mesh)
    compiler = root.find("compiler")
    if compiler is not None and "meshdir" in compiler.attrib:
        del compiler.attrib["meshdir"]
    for geom in root.find("worldbody").findall("geom"):
        if geom.get("type") == "plane":
            geom.set("name", "ground")
            geom.set("size", "10 10 0.1")
    # holosoma reads joint limits for every *named* joint, so a named free
    # joint would shift all limits by one (G1/T1 models leave it unnamed).
    for fj in root.iter("freejoint"):
        fj.attrib.pop("name", None)
    actuator = root.find("actuator")
    if actuator is not None:
        root.remove(actuator)
    for name, (parent, pos) in markers.items():
        body = ET.SubElement(_find_body(root, parent), "body", {"name": name, "pos": _fmt(pos)})
        ET.SubElement(body, "inertial", {"pos": "0 0 0", "mass": "1e-6", "diaginertia": "1e-9 1e-9 1e-9"})
    return tree


def _retarget_urdf(cfg: dict, markers: dict[str, tuple[str, np.ndarray]]) -> ET.ElementTree:
    urdf_path = REPO / cfg["robot"]["urdf"]
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename")
        if fn and not Path(fn).is_absolute():
            mesh.set("filename", str((urdf_path.parent / fn).resolve()))
    for name, (parent, pos) in markers.items():
        ET.SubElement(root, "link", {"name": name})
        joint = ET.SubElement(root, "joint", {"name": f"{name}_joint", "type": "fixed"})
        ET.SubElement(joint, "parent", {"link": parent})
        ET.SubElement(joint, "child", {"link": name})
        ET.SubElement(joint, "origin", {"xyz": _fmt(pos), "rpy": "0 0 0"})
    return tree


def add_ball(tree: ET.ElementTree, radius: float, mass: float, name: str = "ball") -> ET.ElementTree:
    """Copy of ``tree`` with a free ball body (geom name contains ``name``)."""
    tree = copy.deepcopy(tree)
    world = tree.getroot().find("worldbody")
    body = ET.SubElement(world, "body", {"name": f"{name}_link", "pos": f"0.5 0 {radius:.6g}"})
    ET.SubElement(body, "freejoint")
    ET.SubElement(
        body,
        "geom",
        {
            "name": name,
            "type": "sphere",
            "size": f"{radius:.6g}",
            "mass": f"{mass:.6g}",
            "contype": "1",
            "conaffinity": "1",
            "rgba": "1 1 1 1",
        },
    )
    return tree


def build(out_dir: Path | str, cfg: dict | None = None, ball_radius: float | None = None) -> RetargetModel:
    """Write lupus_retarget.{xml,urdf} (+ ``_w_ball.xml``) into ``out_dir``."""
    cfg = cfg or load_config()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    src = ET.parse(REPO / cfg["robot"]["mjcf"]).getroot()
    markers = marker_offsets(src, cfg)

    mj = _retarget_mjcf(cfg, markers)
    xml_path = out / "lupus_retarget.xml"
    mj.write(xml_path)
    urdf_path = out / "lupus_retarget.urdf"
    _retarget_urdf(cfg, markers).write(urdf_path)

    scene = None
    if ball_radius is not None:
        scene = out / "lupus_retarget_w_ball.xml"
        add_ball(mj, ball_radius, robot_ball_mass(cfg)).write(scene)

    joints = tuple(
        j.get("name")
        for j in mj.getroot().find("worldbody").iter("joint")
        if j.get("type", "hinge") in ("hinge", "slide")
    )
    sticking = tuple(f"{s}_foot_sphere_{k}_link" for s in SIDES for k in range(1, 5))
    return RetargetModel(xml_path, urdf_path, scene, sticking, joints)


BALL_YAML = Path(__file__).resolve().parents[1] / "robots" / "ball" / "ball.yaml"


def _ball_spec(key: str, default: float) -> float:
    spec = yaml.safe_load(BALL_YAML.read_text()) if BALL_YAML.is_file() else {}
    v = (spec or {}).get(key, default)
    return float(v.get("value") if isinstance(v, dict) else v)


def robot_ball_radius(cfg: dict | None = None) -> float:
    """Robot ball radius: ``ball.robot_diameter`` of the config, else robots/ball/ball.yaml."""
    cfg = cfg or load_config()
    d = cfg.get("ball", {}).get("robot_diameter")
    return float(d if d is not None else _ball_spec("diameter", 0.15)) / 2


def robot_ball_mass(cfg: dict | None = None) -> float:
    cfg = cfg or load_config()
    m = cfg.get("ball", {}).get("robot_mass")
    return float(m if m is not None else _ball_spec("mass", 0.2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ball", action="store_true", help="also write the scene with the robot ball")
    args = ap.parse_args()
    cfg = load_config()
    radius = robot_ball_radius(cfg) if args.ball else None
    model = build(args.out, cfg, radius)
    print(model)


if __name__ == "__main__":
    main()
