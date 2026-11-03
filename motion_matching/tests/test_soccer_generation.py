"""End-to-end soccer generation on synthetic placeholder databases (CPU only, no downloads).

Builds tiny locomotion / kick_run / kick_stand databases from
``placeholder_clips``, then runs one config of every soccer scenario through
``process_scenario`` and checks the exported kick fields in
the exported npz, for G1 and LUPUS.
"""

from __future__ import annotations


import numpy as np
import pytest

import motion_matching.utils as mm_utils
from motion_matching import run as mm_run
from motion_matching.io.kick_io import load_ball_radius, validate_kick_npz
from motion_matching.placeholder_clips import build_placeholder_databases
from motion_matching.scenarios import soccer as S


@pytest.fixture(scope="module")
def placeholder_db(tmp_path_factory):
    db_dir = tmp_path_factory.mktemp("db")
    for robot in ("g1", "lupus"):
        build_placeholder_databases(robot, tmp_path_factory.mktemp(f"clips_{robot}"), db_dir)
    # Locomotion searches the whole database (its registry search range is not
    # used), so only the database location needs redirecting.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mm_utils, "DATABASE_DIR", db_dir)
        yield db_dir


def _generate(cfg, out_dir, robot, speed_up):
    mm_run.init_worker(required_skills=mm_run.collect_required_skills([cfg]), robot=robot)
    mm_run.process_scenario(cfg, str(out_dir), force=True, speed_up_ratio=speed_up)
    return mm_run._worker_sys


@pytest.fixture(scope="module")
def g1_outputs(placeholder_db, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    results = {}
    for name, fn in S.SOCCER_SCENARIOS.items():
        configs, speed_up = fn(num_configs=2)
        for cfg in configs:
            sys = _generate(cfg, out, "g1", speed_up)
            results[cfg["name"]] = (dict(np.load(out / f"{cfg['name']}_motion.npz")), cfg, sys.kick_events)
    return results


def test_all_soccer_motions_validate(g1_outputs) -> None:
    assert len(g1_outputs) == 10
    for name, (data, cfg, _) in g1_outputs.items():
        validate_kick_npz(data)
        assert data["vel_cmd"].shape == (len(data["joint_pos"]), 2), name
        assert str(data["scenario"]) == cfg["soccer"]["scenario"]


def test_kicks_have_one_contact_at_the_ball(g1_outputs) -> None:
    for name, (data, cfg, events) in g1_outputs.items():
        if cfg["soccer"]["ball"]["mode"] != "kick":
            assert not data["contact_frame"].any() and int(data["kick_side"]) == -1
            continue
        (c,) = np.flatnonzero(data["contact_frame"])
        assert len(events) == 1
        foot = events[0]["kick_foot"]
        assert name.endswith(foot) or f"_{foot}_" in name
        # some body (the kick foot) is at the ball in the contact frame ...
        dist = np.linalg.norm(data["body_pos_w"][c, :, :2] - data["object_pos_w"][c, :2], axis=1).min()
        assert dist < 0.2, (name, dist)
        # ... and the ball lies on the kick foot's side of the root
        root_pos, root_quat = data["body_pos_w"][c, 0], data["body_quat_w"][c, 0]
        w, x, y, z = root_quat
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        rel = data["object_pos_w"][c, :2] - root_pos[:2]
        lateral = -np.sin(yaw) * rel[0] + np.cos(yaw) * rel[1]
        assert (lateral < 0) == (foot == "right"), (name, lateral)
        assert int(data["kick_side"]) == (0 if foot == "right" else 1)
        # ball rests on the ground and stays put until contact
        assert np.allclose(data["object_pos_w"][: c + 1], data["object_pos_w"][0])
        assert data["object_pos_w"][0, 2] == pytest.approx(load_ball_radius())
        # command held at (v_before, 0) during the commit phase
        commit = data["kick_phase"] == 1
        assert commit.any()
        assert np.allclose(data["vel_cmd"][commit, 1], 0.0)
        assert np.ptp(data["vel_cmd"][commit, 0]) == pytest.approx(0.0)


def test_stand_kick_enable_rises_before_commit(g1_outputs) -> None:
    for name, (data, cfg, _) in g1_outputs.items():
        if cfg["soccer"]["scenario"] != "stand_kick":
            continue
        ke, phase = data["kick_enable"], data["kick_phase"]
        first_on = int(np.argmax(ke > 0.5))
        assert ke[0] == 0.0 and first_on > 0
        assert int(np.argmax(phase == 1)) >= first_on - 1


def test_negative_balls_are_placed_as_specified(g1_outputs) -> None:
    for name, (data, cfg, _) in g1_outputs.items():
        mode = cfg["soccer"]["ball"]["mode"]
        if mode not in ("offpath", "far"):
            continue
        root_xy = data["body_pos_w"][:, 0, :2]
        d = S.path_distance(data["object_pos_w"][0, :2], root_xy)[0]
        if mode == "far":
            assert d >= S.FAR_BALL_MIN_DISTANCE - 0.05
        else:
            assert S.IGNORE_LATERAL_MIN - 0.05 <= d <= S.IGNORE_LATERAL_MAX + 0.05


@pytest.mark.skipif(not (mm_utils.LUPUS_DIR / "meshes").exists(), reason="LUPUS CAD meshes not present")
def test_lupus_kick_motions_export(placeholder_db, tmp_path) -> None:
    for fn in (S.walk_kick, S.stand_kick, S.walk_no_ball):
        configs, speed_up = fn(num_configs=1)
        cfg = configs[0]
        sys = _generate(cfg, tmp_path, "lupus", speed_up)
        assert sys.robot.name == "lupus"
        data = dict(np.load(tmp_path / f"{cfg['name']}_motion.npz"))
        validate_kick_npz(data)
        assert data["joint_pos"].shape[1] == sys.robot.num_dofs
        assert data["vel_cmd"].shape == (len(data["joint_pos"]), 2)
        expected_contacts = 1 if cfg["soccer"]["ball"]["mode"] == "kick" else 0
        assert int(data["contact_frame"].sum()) == expected_contacts


def test_lupus_kick_runs_through_motion_matching(placeholder_db, tmp_path, monkeypatch) -> None:
    """Without the CAD meshes: everything up to the tracking export."""
    saved = {}

    def fake_save(qpos, output_file, *, vel_cmd, extra_fields, robot, **_kw):
        saved.update(qpos=qpos, vel_cmd=vel_cmd, extra=extra_fields, robot=robot)
        return {}

    monkeypatch.setattr(mm_run, "save_motion_npz", fake_save)
    for fn in (S.walk_kick, S.stand_kick):
        configs, speed_up = fn(num_configs=1)
        sys = _generate(configs[0], tmp_path, "lupus", speed_up)
        assert saved["robot"] == "lupus" and len(sys.kick_events) == 1
        assert saved["qpos"].shape[1] == 7 + sys.robot.num_dofs
        fields = dict(saved["extra"], vel_cmd=saved["vel_cmd"].astype(np.float32))
        validate_kick_npz(fields)
        assert len(fields["kick_phase"]) == len(saved["vel_cmd"]) and fields["contact_frame"].sum() == 1


def test_phase6_motions_carry_kick_dir_and_dribble_mode(placeholder_db, tmp_path) -> None:
    from motion_matching.io.kick_io import KICK_MODE_DRIBBLE, KICK_MODE_KICK

    for fn in (S.walk_kick_dir, S.stand_kick_dir, S.walk_dribble):
        cfg = fn(num_configs=1)[0][0]
        _generate(cfg, tmp_path, "g1", 1.0)
        data = dict(np.load(tmp_path / f"{cfg['name']}_motion.npz"))
        validate_kick_npz(data)
        assert str(data["scenario"]) == fn.__name__
        if fn is S.walk_dribble:
            assert (data["kick_mode"] == KICK_MODE_DRIBBLE).all() and not data["contact_frame"].any()
            assert (data["kick_dir"] == 0).all() and (data["kick_enable"] == 0).all()
            # ball in front of the robot where it starts walking
            root_xy = data["body_pos_w"][:, 0, :2]
            assert S.path_distance(data["object_pos_w"][0, :2], root_xy)[0] < 0.1
        else:
            assert np.allclose(data["kick_dir"], cfg["soccer"]["kick_dir"])
            assert 0 < abs(float(data["kick_dir"][0])) <= S.KICK_DIR_MAX
            assert int(data["contact_frame"].sum()) == 1
            on = data["kick_enable"] > 0.5
            assert (data["kick_mode"][on] == KICK_MODE_KICK).all()
