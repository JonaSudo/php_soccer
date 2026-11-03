"""GMR (General Motion Retargeting) setup for LUPUS, as the light-weight path for walking clips.

The plan allows GMR for walking (no ball). GMR needs a MuJoCo model of the
robot and an IK config per human format. This module derives both for LUPUS:

- the model is the retargeting model of :mod:`reference_data.retarget_model`
  (markers for toes and hands) with the actuators kept, because GMR lists them;
- the IK configs are GMR's own G1 configs (``bvh_lafan1_to_g1.json``,
  ``smplx_to_g1.json``) with every G1 body replaced by its LUPUS counterpart.
  GMR's rotation offsets map a human frame onto a robot body frame; LUPUS
  body frames are rotated (CAD export), so each offset is re-expressed as
  ``q_off_lupus = q_off_g1 * R_g1^-1 * R_lupus`` with both body rotations
  taken in the zero pose (both robots stand upright with hanging arms there).

    python -m reference_data.gmr_lupus --gmr-root ~/GMR --out ~/GMR_lupus

and in the GMR environment, before creating the retargeter::

    from reference_data.gmr_lupus import register, paths_in
    register(paths_in("~/GMR_lupus"))
    retarget = GMR(src_human="bvh_lafan1", tgt_robot="lupus", actual_human_height=1.75)

The holosoma path (:mod:`reference_data.retarget_lupus`) is the default for
all clips; GMR is optional and was not run in the cloud (its code is not
vendored here). Converting a GMR pickle to the PHP format:
:func:`gmr_pickle_to_raw`.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from reference_data.retarget_model import REPO, _find_body, load_config, marker_offsets

# G1 body (GMR) -> LUPUS body. The LUPUS hip/shoulder chains are pitch-roll-yaw
# like G1, so the body holding the same joint is used where positions agree;
# the hip target goes to the thigh body at the hip centre.
G1_TO_LUPUS = {
    "pelvis": "base_link",
    "left_hip_yaw_link": "left_upper_thigh_link",
    "right_hip_yaw_link": "right_upper_thigh_link",
    "left_hip_roll_link": "left_upper_thigh_link",
    "right_hip_roll_link": "right_upper_thigh_link",
    "left_knee_link": "left_shin_link",
    "right_knee_link": "right_shin_link",
    "left_ankle_roll_link": "left_foot_link",
    "right_ankle_roll_link": "right_foot_link",
    "left_toe_link": "left_toe_marker",
    "right_toe_link": "right_toe_marker",
    "left_shoulder_yaw_link": "left_elbow_link",    # LUPUS: body of the shoulder-yaw joint
    "right_shoulder_yaw_link": "right_elbow_link",
    "left_elbow_link": "left_forearm_link",
    "right_elbow_link": "right_forearm_link",
    "left_wrist_yaw_link": "left_hand_marker",
    "right_wrist_yaw_link": "right_hand_marker",
    "torso_link": None,  # LUPUS has no waist; the trunk is the pelvis body
}

TEMPLATES = {"bvh_lafan1": "bvh_lafan1_to_g1.json", "smplx": "smplx_to_g1.json"}

# Human-to-LUPUS position scale per GMR human body (applied relative to the
# human root). Legs from hip height (LUPUS 0.574 m vs ~0.93 m for GMR's 1.8 m
# human), trunk from pelvis-shoulder (0.337 vs ~0.47 m); arms like the trunk
# because they hang from it. Status: estimate, tune on the first clips.
SCALE = {"legs": 0.62, "trunk": 0.72, "arms": 0.74}
_LEG_KEYS = ("Hips", "UpLeg", "Leg", "Foot", "pelvis", "hip", "knee", "foot")


def _scale_for(name: str) -> float:
    n = name.lower()
    if any(k.lower() in n for k in ("arm", "hand", "shoulder", "elbow", "wrist")):
        return SCALE["arms"]
    if any(k.lower() in n for k in _LEG_KEYS):
        return SCALE["legs"]
    return SCALE["trunk"]


def _strip_meshes(root: ET.Element) -> ET.Element:
    root = copy.deepcopy(root)
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "geom" and (child.get("type") == "mesh" or child.get("mesh")):
                parent.remove(child)
    asset = root.find("asset")
    if asset is not None:
        for mesh in list(asset.findall("mesh")):
            asset.remove(mesh)
    for comp in root.iter("compiler"):
        comp.attrib.pop("meshdir", None)
    return root


def zero_pose_rotations(xml: str) -> dict[str, np.ndarray]:
    """Body name -> world rotation matrix at qpos = 0 (root quat identity)."""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[:] = 0
    if model.njnt and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        data.qpos[3] = 1.0
    mujoco.mj_kinematics(model, data)
    return {model.body(i).name: data.xmat[i].reshape(3, 3).copy() for i in range(model.nbody)}


def lupus_gmr_model(cfg: dict | None = None) -> str:
    """LUPUS MJCF for GMR: collision primitives, markers, actuators, no meshes."""
    cfg = cfg or load_config()
    src = ET.parse(REPO / cfg["robot"]["mjcf"]).getroot()
    markers = marker_offsets(src, cfg)
    root = _strip_meshes(src)
    for name, (parent, pos) in markers.items():
        body = ET.SubElement(_find_body(root, parent), "body", {"name": name, "pos": " ".join(f"{v:.6g}" for v in pos)})
        ET.SubElement(body, "inertial", {"pos": "0 0 0", "mass": "1e-6", "diaginertia": "1e-9 1e-9 1e-9"})
    return ET.tostring(root, encoding="unicode")


def convert_config(template: dict, r_g1: dict[str, np.ndarray], r_lupus: dict[str, np.ndarray]) -> dict:
    """G1 IK config -> LUPUS IK config (body names, rotation offsets, scales)."""
    out = {k: copy.deepcopy(v) for k, v in template.items() if not k.startswith("ik_match_table")}
    out["robot_root_name"] = "base_link"
    for table in ("ik_match_table1", "ik_match_table2"):
        new = {}
        for g1_body, (human, w_pos, w_rot, pos_off, rot_off) in template[table].items():
            lupus_body = G1_TO_LUPUS.get(g1_body, "missing")
            if lupus_body == "missing":
                raise KeyError(f"no LUPUS body for G1 body {g1_body!r}")
            if lupus_body is None or lupus_body in new:
                continue
            q = Rotation.from_quat(rot_off, scalar_first=True)
            delta = Rotation.from_matrix(r_g1[g1_body]).inv() * Rotation.from_matrix(r_lupus[lupus_body])
            q_new = (q * delta).as_quat(scalar_first=True)
            new[lupus_body] = [human, w_pos, w_rot, [0.0, 0.0, 0.0], [float(v) for v in q_new]]
        out[table] = new
    out["human_scale_table"] = {k: _scale_for(k) for k in template["human_scale_table"]}
    return out


def build(gmr_root: Path, out_dir: Path, cfg: dict | None = None) -> dict[str, Path]:
    gmr_root, out_dir = Path(gmr_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    g1_xml = gmr_root / "assets/unitree_g1/g1_mocap_29dof.xml"
    r_g1 = zero_pose_rotations(ET.tostring(_strip_meshes(ET.parse(g1_xml).getroot()), encoding="unicode"))
    lupus_xml = lupus_gmr_model(cfg)
    r_lupus = zero_pose_rotations(lupus_xml)
    paths = {"xml": out_dir / "lupus_gmr.xml"}
    paths["xml"].write_text(lupus_xml)
    for fmt, name in TEMPLATES.items():
        template = json.loads((gmr_root / "general_motion_retargeting/ik_configs" / name).read_text())
        conf = convert_config(template, r_g1, r_lupus)
        paths[fmt] = out_dir / f"{fmt}_to_lupus.json"
        paths[fmt].write_text(json.dumps(conf, indent=2))
    return paths


def paths_in(out_dir: Path | str) -> dict[str, Path]:
    out_dir = Path(out_dir).expanduser()
    return {"xml": out_dir / "lupus_gmr.xml", **{fmt: out_dir / f"{fmt}_to_lupus.json" for fmt in TEMPLATES}}


def register(paths: dict[str, Path]) -> None:
    """Make "lupus" known to an imported GMR (call inside the GMR environment)."""
    from general_motion_retargeting import params

    params.ROBOT_XML_DICT["lupus"] = paths["xml"]
    for fmt in TEMPLATES:
        params.IK_CONFIG_DICT.setdefault(fmt, {})["lupus"] = paths[fmt]
    params.ROBOT_BASE_DICT["lupus"] = "base_link"
    params.VIEWER_CAM_DISTANCE_DICT["lupus"] = 2.0


def gmr_pickle_to_raw(pkl: Path | str, out: Path | str) -> Path:
    """GMR result (root_pos, root_rot xyzw, dof_pos, fps) -> ``*.lupus_raw.npz``."""
    from reference_data.lupus_kin import kinematics

    with open(pkl, "rb") as f:
        d = pickle.load(f)
    kin = kinematics()
    quat_wxyz = np.asarray(d["root_rot"])[:, [3, 0, 1, 2]]
    qpos_mj = np.concatenate([np.asarray(d["root_pos"]), quat_wxyz, np.asarray(d["dof_pos"])], axis=1)
    if qpos_mj.shape[1] != 7 + len(kin.mj_joint_names):
        raise ValueError(f"GMR dof_pos has {qpos_mj.shape[1] - 7} joints, LUPUS has {len(kin.mj_joint_names)}")
    np.savez(
        out, qpos_mj=qpos_mj, joint_names=np.array(kin.mj_joint_names), fps=float(d["fps"]),
        ball_radius=0.0, meta=json.dumps({"source": str(pkl), "retargeter": "gmr"}),
    )
    return Path(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gmr-root", type=Path, required=True, help="GMR checkout (github.com/YanjieZe/GMR)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    for k, v in build(args.gmr_root, args.out).items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
