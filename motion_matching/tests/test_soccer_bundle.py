"""Soccer training bundle (build_soccer_bundle) and the Phase-2 acceptance check.

CPU only, on the synthetic placeholder databases: builds a tiny bundle for G1
and LUPUS, checks it against ``wbt_training.utils.registry.validate_bundle``
and ``combine_motions`` (what the teacher's preprocess does with
``file://`` bundles) and runs the acceptance on it. The geometry helpers of
the acceptance are tested on hand-made signals.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

import motion_matching.utils as mm_utils
from motion_matching import build_soccer_bundle as B
from motion_matching import soccer_acceptance as A
from motion_matching.io.kick_io import KICK_NPZ_FIELDS
from motion_matching.placeholder_clips import build_placeholder_databases

REPO_ROOT = mm_utils.PKG_ROOT.parents[1]
LUPUS_MESHES = (mm_utils.LUPUS_DIR / "meshes").exists()
SMALL = {
    "walk_kick": 4,
    "stand_kick": 2,
    "stand_ball_outside_zone": 2,
    "walk_ball_offpath": 2,
    "walk_no_ball": 2,
}


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


def test_default_plan_matches_the_plan_amounts() -> None:
    counts = B.plan_counts()
    assert set(counts) == set(B.PHASE2_SHARES)
    assert 200 <= counts["walk_kick"] <= 400
    assert 80 <= counts["stand_kick"] <= 130
    total = sum(counts.values())
    for name, share in B.PHASE2_SHARES.items():
        assert counts[name] / total == pytest.approx(share, abs=0.01)
    assert sum(B.PHASE2_SHARES.values()) == pytest.approx(1.0)


def test_plan_overrides_and_phase6() -> None:
    counts = B.plan_counts(100, B.parse_counts("walk_kick=7, walk_no_ball=0"), phase6=True)
    assert counts["walk_kick"] == 7 and "walk_no_ball" not in counts
    assert set(B.PHASE6_SHARES) <= set(counts)
    with pytest.raises(ValueError):
        B.plan_counts(10, {"nope": 1})
    with pytest.raises(ValueError):
        B.parse_counts("walk_kick")


def test_seed_offset_draws_other_configs() -> None:
    a, _ = B.scenario_configs("walk_kick", 3)
    b, _ = B.scenario_configs("walk_kick", 3, seed_offset=1000)
    assert a == B.scenario_configs("walk_kick", 3)[0]
    assert [c["soccer"]["seed"] for c in a] != [c["soccer"]["seed"] for c in b]


# --------------------------------------------------------------------------
# Acceptance helpers
# --------------------------------------------------------------------------


def test_gait_phase_from_foot_offset() -> None:
    fps, hz = 50, 1.0
    t = np.arange(200) / fps
    offset = 0.1 * np.sin(2 * np.pi * hz * t)  # left ahead in the first half stride
    for k in (60, 80, 110, 140, 170):
        expected = (hz * t[k]) % 1.0
        assert A.gait_phase_at(offset, k) == pytest.approx(expected, abs=0.06), k
    assert A.gait_phase_at(np.zeros(100), 90) is None  # standing
    assert A.gait_phase_at(offset, 20) is None  # not one stride yet


def test_foot_index_for_both_robots() -> None:
    lupus = ["base_link", "left_foot_link", "left_foot_contact_point", "right_foot_link"]
    g1 = ["pelvis", "left_ankle_roll_link", "left_foot_contact_point", "right_ankle_roll_link"]
    assert A.foot_body_index(lupus, "left") == 1 and A.foot_body_index(lupus, "right") == 3
    assert A.foot_body_index(g1, "left") == 1 and A.foot_body_index(g1, "right") == 3


def test_terrain_flatness(tmp_path) -> None:
    flat = tmp_path / "a_terrain.npy"
    np.save(flat, np.zeros((0, 10, 10), dtype=np.float32))
    boxes = tmp_path / "b_terrain.npy"
    np.save(boxes, np.ones((2, 10, 10), dtype=np.float32))
    assert A.terrain_is_flat(flat, None)[0]
    assert not A.terrain_is_flat(boxes, None)[0]
    obj = tmp_path / "c_terrain.obj"
    obj.write_text("v 0 0 0\nv 1 0 -0.1\nv 0 1 0\nf 1 2 3\n")
    assert A.terrain_is_flat(None, obj)[0]
    obj.write_text("v 0 0 0.3\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    assert not A.terrain_is_flat(None, obj)[0]


# --------------------------------------------------------------------------
# End-to-end bundle on placeholder databases
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def placeholder_db(tmp_path_factory):
    db_dir = tmp_path_factory.mktemp("db")
    for robot in ("g1", "lupus"):
        build_placeholder_databases(robot, tmp_path_factory.mktemp(f"clips_{robot}"), db_dir)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mm_utils, "DATABASE_DIR", db_dir)
        yield db_dir


def _build(robot, tmp_path_factory):
    out = tmp_path_factory.mktemp(f"bundle_{robot}") / "bundle"
    manifest = B.build_bundle(out, robot=robot, counts=SMALL, jobs=1)
    return out, manifest


@pytest.fixture(scope="module")
def g1_bundle(placeholder_db, tmp_path_factory):
    return _build("g1", tmp_path_factory)


def _check_bundle_contract(out, manifest, robot) -> None:
    assert manifest["robot"] == robot
    assert manifest["failed"] == [], manifest["failed"]
    assert manifest["counts"] == SMALL
    assert manifest["registry_validation"] == "ok"
    on_disk = json.loads((out / "manifest.json").read_text())
    assert on_disk["counts"] == SMALL and len(on_disk["motions"]) == sum(SMALL.values())
    assert not any(out.rglob("*raw*")), "raw output must stay outside the bundle"
    for rec in on_disk["motions"]:
        path = out / rec["file"]
        with np.load(path) as d:
            T = len(d["joint_pos"])
            assert d["joint_pos"].shape[1] == 7 + len(d["joint_names"])
            for key in KICK_NPZ_FIELDS:
                assert len(d[key]) == T, key
            assert d["kick_side"].shape == (T,) and d["vel_cmd"].shape == (T, 2)
            assert "scenario" not in d.files
            assert rec["frames"] == T
        assert path.with_name(path.name.replace("_motion.npz", "_terrain.npy")).is_file()


def _check_registry_and_combine(out) -> None:
    sys.path.insert(0, str(REPO_ROOT))
    from wbt_training.utils.multi_motion_helpers import combine_motions
    from wbt_training.utils.registry import bundle_pairs, registry_reference

    assert registry_reference(f"file://{out}") == f"file://{out.resolve()}"
    pairs = bundle_pairs(out)
    combined = np.load(combine_motions([m for m, _, _ in pairs]))
    T = len(combined["joint_pos"])
    for key in (*KICK_NPZ_FIELDS, "kick_side", "vel_cmd", "motion_ends"):
        assert len(combined[key]) == T, key
    assert int(combined["contact_frame"].sum()) == SMALL["walk_kick"] + SMALL["stand_kick"]


def test_g1_bundle_is_a_valid_registry(g1_bundle) -> None:
    out, manifest = g1_bundle
    _check_bundle_contract(out, manifest, "g1")
    _check_registry_and_combine(out)


def test_g1_bundle_passes_acceptance(g1_bundle) -> None:
    out, manifest = g1_bundle
    acc = manifest["acceptance"]
    assert acc["passed"], acc["errors"]
    assert acc["flat_terrain"]
    assert acc["kick_sides"]["walk_kick"] == {"right": 2, "left": 2}
    assert acc["kick_sides"]["stand_kick"] == {"right": 1, "left": 1}
    assert acc["ball_foot_distance"]["max"] <= A.AcceptanceConfig.max_ball_foot_distance
    assert sum(acc["counts"].values()) == sum(SMALL.values())


def test_acceptance_catches_broken_bundles(g1_bundle, tmp_path) -> None:
    import shutil

    out, _ = g1_bundle
    bad = tmp_path / "bad"
    shutil.copytree(out, bad)
    # Move the ball of one kick away from the foot and put a box on one terrain.
    kick = next(bad.rglob("walk_kick_0000_*_motion.npz"))
    with np.load(kick) as d:
        data = {k: d[k] for k in d.files}
    data["object_pos_w"][:, 0] += 1.0
    np.savez(kick, **data)
    np.save(next(bad.rglob("walk_no_ball_*_terrain.npy")), np.ones((1, 10, 10), dtype=np.float32))
    report = A.check_bundle(bad)
    assert not report.passed and not report.flat_terrain
    assert any("from the" in e and "foot at contact" in e for e in report.errors)
    assert B.main(["check", str(bad)]) == 1
    assert B.main(["check", str(out)]) == 0


def test_bundle_refuses_to_overwrite(g1_bundle) -> None:
    out, _ = g1_bundle
    with pytest.raises(FileExistsError):
        B.build_bundle(out, robot="g1", counts={"walk_no_ball": 1}, jobs=1)
    with pytest.raises(ValueError):
        B.build_bundle(out.parent / "x", raw_dir=out.parent / "x" / "raw", counts={"walk_no_ball": 1}, jobs=1)


@pytest.mark.skipif(not LUPUS_MESHES, reason="LUPUS CAD meshes not present (symlink urdf_export to robots/lupus/meshes)")
def test_lupus_bundle(placeholder_db, tmp_path_factory) -> None:
    out, manifest = _build("lupus", tmp_path_factory)
    _check_bundle_contract(out, manifest, "lupus")
    _check_registry_and_combine(out)
    acc = manifest["acceptance"]
    assert acc["passed"], acc["errors"]
    with np.load(next(out.rglob("walk_kick_*_motion.npz"))) as d:
        names = [str(n) for n in d["body_names"]]
        assert "left_foot_link" in names and d["joint_pos"].shape[1] == 7 + 22
