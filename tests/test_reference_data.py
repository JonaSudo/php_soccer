"""Phase 1 reference-data pipeline (reference_data/), CPU only, no mocap data needed."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("scipy")

from reference_data import annotate as A  # noqa: E402
from reference_data import feasibility as F  # noqa: E402
from reference_data import human_clip as H  # noqa: E402
from reference_data import retarget_model as RM  # noqa: E402
from reference_data.clip_ops import LupusClip  # noqa: E402
from reference_data.lupus_kin import kinematics  # noqa: E402
from reference_data.synthetic import _base_qpos, stand_kick_clip, walk_clip  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def concat(*clips: LupusClip) -> LupusClip:
    """Join clips; later clips are shifted so the root continues where the previous ended."""
    q = [clips[0].qpos]
    balls = [clips[0].ball_pos if clips[0].ball_pos is not None else None]
    for c in clips[1:]:
        qq = c.qpos.copy()
        qq[:, 4:6] += q[-1][-1, 4:6] - qq[0, 4:6]
        q.append(qq)
        balls.append(c.ball_pos)
    ball = None
    if balls[-1] is not None:
        pre = sum(len(x) for x in q[:-1])
        ball = np.concatenate([np.tile(balls[-1][0], (pre, 1)), balls[-1]])
    return LupusClip(np.concatenate(q), clips[0].fps, ball, clips[-1].ball_radius)


# ---- retargeting model ----------------------------------------------------------


@pytest.fixture(scope="module")
def rt_model(tmp_path_factory):
    return RM.build(tmp_path_factory.mktemp("rt"), ball_radius=0.1)


def test_retarget_model_markers(rt_model):
    m = mujoco.MjModel.from_xml_path(str(rt_model.xml))
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    pos = lambda n: d.xpos[m.body(n).id]  # noqa: E731
    for side, sign in (("left", 1), ("right", -1)):
        corners = np.array([pos(f"{side}_foot_sphere_{k}_link") for k in range(1, 5)])
        assert np.allclose(corners[:, 2], 0.0, atol=1e-3)  # on the sole in the zero pose
        assert np.all(np.sign(corners[:, 1]) == sign)
        toe, ankle = pos(f"{side}_toe_marker"), pos(f"{side}_foot_link")
        assert toe[0] > ankle[0] + 0.05 and abs(toe[2]) < 1e-3
        hand, elbow = pos(f"{side}_hand_marker"), pos(f"{side}_forearm_link")
        assert hand[2] < elbow[2] - 0.2
    assert pos("head_marker")[2] > 1.0
    # holosoma reads limits of every named joint: the free joint must be unnamed
    assert m.joint(0).name == ""
    assert rt_model.joint_names == kinematics().mj_joint_names
    assert len(rt_model.foot_sticking_links) == 8
    assert all(("left" in n) != ("right" in n) for n in rt_model.foot_sticking_links)


def test_ball_scene(rt_model):
    m = mujoco.MjModel.from_xml_path(str(rt_model.scene_xml))
    assert m.nq == 7 + 22 + 7
    assert m.geom("ball").size[0] == pytest.approx(0.1)
    assert m.geom("ground").type == mujoco.mjtGeom.mjGEOM_PLANE


def test_robot_height_in_config_matches_model(rt_model):
    m = mujoco.MjModel.from_xml_path(str(rt_model.xml))
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    g = m.geom("head_link_collision").id
    top = d.geom_xpos[g][2] + m.geom_size[g][2]
    assert RM.load_config()["robot"]["height"] == pytest.approx(top, abs=0.02)


def test_joint_mappings_use_existing_bodies(rt_model):
    m = mujoco.MjModel.from_xml_path(str(rt_model.xml))
    bodies = {m.body(i).name for i in range(m.nbody)}
    cfg = RM.load_config()
    for fmt, mapping in cfg["joint_mappings"].items():
        assert set(mapping.values()) <= bodies, fmt
        assert set(cfg["toe_joints"][fmt]) <= set(mapping)


# ---- human clips ----------------------------------------------------------------

BVH = """HIERARCHY
ROOT Hips
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT LeftUpLeg
  {
    OFFSET 10 0 0
    CHANNELS 3 Zrotation Xrotation Yrotation
    End Site
    {
      OFFSET 0 -40 0
    }
  }
}
MOTION
Frames: 2
Frame Time: 0.0333333
0 90 0 0 0 0 0 0 0
0 90 0 0 0 90 0 0 0
"""


def test_bvh_forward_kinematics():
    b = H.parse_bvh(BVH)
    assert b.names == ["Hips", "LeftUpLeg"]
    assert b.frame_time == pytest.approx(1 / 30, rel=1e-4)
    p = H.bvh_global_positions(b)
    assert np.allclose(p[0, 1], [10, 90, 0])
    # 90 deg about the root's y axis turns the +x hip offset into -z
    assert np.allclose(p[1, 1], [0, 90, -10], atol=1e-9)


def test_y_up_to_z_up_keeps_handedness():
    x, y, z = np.eye(3)
    m = np.stack([H.y_up_to_z_up(v) for v in (x, y, z)])
    assert np.linalg.det(m) == pytest.approx(1.0)
    assert np.allclose(H.y_up_to_z_up(y), [0, 0, 1])


def test_bvh_name_mapping_by_structure():
    # Xsens naming: "Shoulder" is the upper arm, "Collar" the clavicle
    names = ["Hips", "Chest", "Head", "LeftCollar", "LeftShoulder", "LeftElbow", "LeftWrist",
             "LeftHip", "LeftKnee", "LeftAnkle", "LeftToe"]
    parents = np.array([-1, 0, 1, 1, 3, 4, 5, 0, 7, 8, 9])
    idx = H.map_bvh_names(names, parents)
    assert names[idx["LeftArm"]] == "LeftShoulder"
    assert names[idx["LeftForeArm"]] == "LeftElbow"
    assert names[idx["LeftHand"]] == "LeftWrist"
    assert names[idx["LeftUpLeg"]] == "LeftHip"
    assert names[idx["LeftLeg"]] == "LeftKnee"
    assert names[idx["LeftToeBase"]] == "LeftToe"
    assert "LeftShoulder" not in idx or idx["LeftShoulder"] != idx["LeftArm"]


def _human(T=31, fps=30.0, ball=False):
    names = ["Hips", "LeftUpLeg", "RightUpLeg", "Head"]
    j = np.zeros((T, 4, 3))
    t = np.arange(T) / fps
    j[:, 0] = np.stack([t, 0 * t, 0.9 + 0 * t], 1)
    j[:, 1] = j[:, 0] + [0, 0.1, 0]
    j[:, 2] = j[:, 0] + [0, -0.1, 0]
    j[:, 3] = j[:, 0] + [0, 0, 0.7]
    obj = np.stack([1 + t, 0 * t, 0.11 + 0 * t], 1) if ball else None
    return H.HumanClip(j, names, fps, "lafan", 1.75, object_pos=obj)


def test_human_clip_roundtrip_and_resample(tmp_path):
    c = _human(ball=True)
    p = c.save(tmp_path / "c.human.npz")
    d = H.HumanClip.load(p)
    assert d.joint_names == c.joint_names and np.allclose(d.joints, c.joints)
    assert d.has_ball and np.allclose(d.object_pos, c.object_pos)
    r = c.resample(60.0)
    assert r.num_frames == 61
    assert np.allclose(r.joint("Hips")[-1], c.joint("Hips")[-1])
    assert np.allclose(r.object_pos[2], c.object_pos[1])
    # walking along +x with the left hip at +y: heading is identity
    assert np.allclose(c.heading_quat(0), [1, 0, 0, 0], atol=1e-9)


# ---- kinematics / format --------------------------------------------------------


def test_php_mujoco_roundtrip_and_fk_order():
    kin = kinematics()
    rng = np.random.default_rng(0)
    q = _base_qpos(1)
    q[0, 7:] = rng.uniform(kin.lower, kin.upper) * 0.5
    assert np.allclose(kin.mj_to_php(kin.php_to_mj(q)), q)
    # FK through MuJoCo (MJCF order) must agree with the URDF kinematics of
    # motion matching (URDF order) -> the joint reordering is right.
    from motion_matching.kinematics.robot_spec import LUPUS_SPEC

    urdf = LUPUS_SPEC.make_kinematics()
    pos, _ = urdf.compute_joint_poses(
        root_position=q[0, 4:7], root_orientation=q[0, :4], joint_angles=q[0, 7:]
    )
    names = list(urdf.joint_names)
    kin.forward(q[0])
    m, d = kin.model, kin.data
    for jn in ("left_knee", "right_ankle_roll", "left_elbow", "head_pitch"):
        jid = m.joint(jn).id
        assert np.allclose(d.xanchor[jid], pos[names.index(jn)], atol=1e-4), jn


# ---- clip edits -----------------------------------------------------------------


def test_time_scale():
    c = walk_clip(steps=2)
    s = c.time_scale(0.5)
    assert s.duration == pytest.approx(2 * c.duration, abs=1 / c.fps)
    assert np.allclose(s.qpos[::2][: c.num_frames], c.qpos[: len(s.qpos[::2])], atol=1e-6)
    assert np.allclose(np.linalg.norm(s.root_quat, axis=1), 1.0)
    assert s.meta["time_scale"] == 0.5


def test_mirror():
    c = stand_kick_clip("right")
    m = c.mirror()
    assert np.allclose(m.mirror().qpos, c.qpos)
    fk_c, fk_m = kinematics().clip_fk(c.qpos), kinematics().clip_fk(m.qpos)
    flip = np.array([1, -1, 1])
    assert np.allclose(fk_m["left_center"], fk_c["right_center"] * flip, atol=1e-4)
    assert np.allclose(m.ball_pos, c.ball_pos * flip)


def test_place_on_ground():
    c = walk_clip(steps=2)
    up = c.shift_z(0.05)
    back = up.place_on_ground()
    assert np.allclose(back.root_pos[:, 2], c.root_pos[:, 2], atol=2e-3)


# ---- feasibility ------------------------------------------------------------------


def test_standing_is_feasible():
    c = LupusClip(_base_qpos(60), 30.0).place_on_ground()
    r = F.check(c)
    assert r.ok, r.failing
    tau, res = F.joint_torques(c, return_residual=True)
    assert res.max() < 1.0  # both feet carry the full weight
    kin = kinematics()
    # holding the bent default pose needs knee torque, but within the motors
    assert 0.5 < abs(tau[:, kin.urdf_names.index("left_knee")]).max() < kin.effort[kin.urdf_names.index("left_knee")]


def test_fast_motion_flagged_and_slowed():
    kin = kinematics()
    q = _base_qpos(90)
    t = np.arange(90) / 30
    i = 7 + kin.urdf_names.index("left_elbow")
    q[:, i] = -0.9 + 0.8 * np.sin(2 * np.pi * 3.0 * t)  # 15 rad/s peak, limit ~18.6
    i2 = 7 + kin.urdf_names.index("left_shoulder_roll")
    q[:, i2] = 1.3 + 1.0 * np.sin(2 * np.pi * 4.0 * t)  # 25 rad/s peak, limit ~18.6
    c = LupusClip(q, 30.0).place_on_ground()
    th = F.Thresholds(torque_ratio=1e9)
    assert "left_shoulder_roll" in F.check(c, th).failing
    speed, rep = F.slowest_feasible_scale(c, th)
    assert speed is not None and speed < 1.0 and rep.ok


def test_ground_penetration_flagged():
    c = LupusClip(_base_qpos(30), 30.0).place_on_ground().shift_z(-0.03)
    assert "ground_penetration" in F.check(c, torques=False).failing


# ---- annotation -----------------------------------------------------------------


def test_annotate_stand_kick():
    c = stand_kick_clip("right")
    a = A.annotate(c)
    assert a.group == "stand_kick" and a.kick_foot == "right"
    assert abs(a.contact_frame - c.meta["true_contact"]) <= 1
    assert a.entry_start < a.skill_start < a.contact_frame < a.skill_end
    assert abs(a.contact_distance) < 0.01
    assert a.ball_in_heading[0] > 0  # ball in front of the sole centre
    m = A.annotate(c.mirror())
    assert m.kick_foot == "left" and m.group == "stand_kick"
    assert np.allclose(m.ball_in_heading, np.array(a.ball_in_heading) * [1, -1, 1], atol=1e-6)


def test_annotate_walk_kick_and_reject():
    kick = stand_kick_clip("right").trim(25)  # drop the quiet stance
    wk = concat(walk_clip(steps=4).trim(0, 75), kick)  # kick right after the last step
    a = A.annotate(wk)
    assert a.group == "walk_kick", a.reason
    assert a.runup_touchdowns >= 2 and a.entry_start < a.skill_start
    lunge = concat(walk_clip(steps=1).trim(0, 30), kick)
    assert A.annotate(lunge).group == "reject"


def test_annotate_synthetic_ball_and_overrides():
    c = stand_kick_clip("right")
    c.ball_pos = None
    a = A.annotate(c, overrides={"skill_end": 80})
    assert a.ball_source == "synthetic" and abs(a.contact_distance) < 1e-3
    assert a.skill_end == 80
    track = A.ball_track(c, a)
    assert np.allclose(track[a.contact_frame], a.ball_pos_contact)
    assert A.scale_frames({"contact_frame": 35, "kick_foot": "left"}, 0.7) == {"contact_frame": 50, "kick_foot": "left"}
    assert A.mirror_fields({"kick_foot": "left"})["kick_foot"] == "right"


# ---- export + acceptance ----------------------------------------------------------


def test_postprocess_and_verify(tmp_path):
    from reference_data.postprocess import ClipSpec, process, write_manifest
    from reference_data.verify import main as verify_main
    from reference_data.verify import verify_clip

    c = stand_kick_clip("right")
    loose = F.Thresholds(torque_ratio=1e9)
    out = process(c, ClipSpec("synth", "stand_kick", speed=0.7), tmp_path, thresholds=loose)
    assert [e.name for e in out] == ["synth_0.7x", "synth_mirror_0.7x"]
    assert all(e.accepted for e in out), [e.reasons for e in out]
    rec = json.loads(out[1].json.read_text())
    assert rec["annotation"]["kick_foot"] == "left" and rec["mirrored"]
    with np.load(out[0].npz) as d:
        assert d["qpos"].shape[1] == 29 and "object_pos_w" in d
        assert float(d["fps"]) == 30.0
        assert d["qpos"].shape[0] == pytest.approx(c.num_frames / 0.7, abs=2)
    for e in out:
        r = verify_clip(e.npz)
        assert r.ok, r.problems
    write_manifest(tmp_path, out)
    assert verify_main([str(tmp_path)]) == 0
    # a clip whose ball is moved away fails the kick check
    bad = LupusClip.load(out[0].npz)
    bad.ball_pos[:, 0] += 0.2
    bad.save(out[0].npz, {"contact_frame": rec["annotation"]["contact_frame"]})
    assert not verify_clip(out[0].npz).ok


def test_kick_json_feeds_phase2_kick_database(tmp_path):
    """The exported *.kick.json loads with kick_io and bakes into a LUPUS kick database."""
    from motion_matching.build_kick_database import build_kick_database, collect_annotations
    from reference_data.postprocess import ClipSpec, process

    c = stand_kick_clip("right")
    out = process(c, ClipSpec("synth", "stand_kick", speed=0.7), tmp_path,
                  thresholds=F.Thresholds(torque_ratio=1e9))
    folder = tmp_path / "stand_kick"
    assert sorted(p.name for p in folder.glob("*.kick.json")) == ["synth_0.7x.kick.json"]  # no mirror copy
    anns = collect_annotations([folder])
    (clip_path, ann), = anns
    assert clip_path == out[0].npz and ann.kind == "stand" and ann.kick_foot == "right"
    rec = json.loads(out[0].json.read_text())["annotation"]
    assert ann.contact_frame == rec["contact_frame"] and ann.speed_up == 1.0
    side = build_kick_database(anns, robot="lupus", output=str(tmp_path / "db" / "database_lupus_kick_stand.bin"))
    assert len(side.ranges) == 2 and {r.kick_foot for r in side.ranges} == {"left", "right"}
    r = next(r for r in side.ranges if not r.mirrored)
    assert r.search[0] <= r.search[1] <= r.contact_frame <= r.search[2]
    assert np.allclose(r.ball_pos[:2], ann.ball_pos[:2])


def test_lupus_raw_roundtrip(tmp_path):
    kin = kinematics()
    c = stand_kick_clip("left")
    p = tmp_path / "x.lupus_raw.npz"
    np.savez(p, qpos_mj=kin.php_to_mj(c.qpos), joint_names=np.array(kin.mj_joint_names), fps=30.0,
             ball_pos=c.ball_pos, ball_radius=0.1, meta=json.dumps({"source": "test"}))
    r = LupusClip.from_retarget(p)
    assert np.allclose(r.qpos, c.qpos) and r.name == "x" and r.meta["source"] == "test"


# ---- catalog / pipeline / GMR -------------------------------------------------------


def test_catalog_valid():
    from reference_data.pipeline import load_catalog
    from reference_data.postprocess import GROUPS

    cat = load_catalog()
    ids = [c["id"] for c in cat["clips"]]
    assert len(ids) == len(set(ids))
    for c in cat["clips"]:
        assert c["group"] in GROUPS and c["source"] in cat["sources"]


def test_pipeline_expand(tmp_path):
    from reference_data.pipeline import expand

    (tmp_path / "lafan").mkdir()
    (tmp_path / "lafan" / "walk1_subject1.bvh").write_text(BVH)
    cap = tmp_path / "oc/raw_mocap/soccer/case1_kick_forward/2026_0001"
    cap.mkdir(parents=True)
    (cap / "motion_actor.bvh").write_text(BVH)
    npz_dir = tmp_path / "oc/npz/soccer/case1_kick_forward"
    npz_dir.mkdir(parents=True)
    np.savez(npz_dir / "2026_0001_with_contact.npz", object_pos_w=np.zeros((2, 3)), fps=np.array([30.0]))
    cat = {
        "sources": {"lafan": {"kind": "bvh"}, "omnicontact": {"kind": "omnicontact"}},
        "defaults": {"walk_kick": {"speed": "auto", "speeds": [0.7]}},
        "clips": [
            {"id": "w", "source": "lafan", "path": "walk1_subject1.bvh", "group": "walk", "trim": [0, 1]},
            {"id": "k", "source": "omnicontact", "path": "raw_mocap/soccer/case1_kick_forward/*/motion_actor.bvh",
             "npz_dir": "npz/soccer/case1_kick_forward", "group": "walk_kick"},
        ],
    }
    es = expand(cat, {"lafan": tmp_path / "lafan", "omnicontact": tmp_path / "oc"})
    assert [e.id for e in es] == ["w", "k"] or [e.id for e in es] == ["w", "k_2026_0001"]
    assert es[1].npz.name == "2026_0001_with_contact.npz" and es[1].speeds == [0.7]


def test_gmr_offset_conversion():
    from scipy.spatial.transform import Rotation

    from reference_data.gmr_lupus import convert_config

    rz = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    tmpl = {
        "robot_root_name": "pelvis", "human_root_name": "Hips", "ground_height": 0.0,
        "human_height_assumption": 1.8, "use_ik_match_table1": True, "use_ik_match_table2": True,
        "human_scale_table": {"Hips": 0.9, "LeftArm": 0.75},
        "ik_match_table1": {"pelvis": ["Hips", 0, 10, [0, 0, 0], [0.5, 0.5, 0.5, 0.5]],
                            "torso_link": ["Spine2", 0, 10, [0, 0, 0], [1, 0, 0, 0]]},
        "ik_match_table2": {"pelvis": ["Hips", 10, 5, [0, 0, 0], [1, 0, 0, 0]]},
    }
    out = convert_config(tmpl, {"pelvis": np.eye(3)}, {"base_link": rz})
    assert list(out["ik_match_table1"]) == ["base_link"]  # torso dropped (no waist)
    q = out["ik_match_table2"]["base_link"][4]
    assert np.allclose(Rotation.from_quat(q, scalar_first=True).as_matrix(), rz)
    # identical frames leave the offset unchanged
    same = convert_config(tmpl, {"pelvis": rz}, {"base_link": rz})
    assert np.allclose(np.abs(same["ik_match_table1"]["base_link"][4]), [0.5] * 4)


# ---- retargeting (only where the holosoma_retargeting env is installed) --------------

OMOMO = REPO / "thirdparty/holosoma/src/holosoma_retargeting/holosoma_retargeting/demo_data/OMOMO_new/sub3_largebox_003.pt"


def test_retarget_demo_clip(tmp_path):
    pytest.importorskip("cvxpy")
    pytest.importorskip("torch")
    pytest.importorskip("holosoma_retargeting")
    if not OMOMO.exists():
        pytest.skip("holosoma submodule not checked out")
    if not (mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body><joint/><geom size='.1'/></body></worldbody></mujoco>"
    ).jnt_type[0] in (mujoco.mjtJoint.mjJNT_HINGE,)):
        pytest.skip("holosoma retargeter needs mujoco==3.8.1 (enum comparison)")
    from reference_data.retarget_lupus import retarget

    clip = H.from_intermimic_pt(OMOMO)
    out = retarget(clip, tmp_path / "o.lupus_raw.npz", work_dir=tmp_path / "w", max_frames=12)
    raw = LupusClip.from_retarget(out)
    assert raw.num_frames == 12
    assert np.abs(raw.dofs).max() > 0.1  # joints move (not stuck at zero)
    assert kinematics().clip_fk(raw.qpos)["lowest"].min() > -0.01
