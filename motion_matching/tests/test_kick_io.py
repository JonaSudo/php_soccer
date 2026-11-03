"""Kick annotations, sidecars and npz kick fields (no databases needed)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from motion_matching.generate_database import DatabaseEntry
from motion_matching.io import kick_io as K


def test_ball_radius_from_yaml_diameter(tmp_path) -> None:
    p = tmp_path / "ball.yaml"
    p.write_text("diameter: {value: 0.15, status: estimate}\nmass:\n  value: 0.4\n  status: estimate\n")
    assert K.load_ball_radius(p) == pytest.approx(0.075)
    p.write_text("radius:\n  value: 0.1\n")
    assert K.load_ball_radius(p) == pytest.approx(0.1)


def test_ball_radius_fallback(tmp_path) -> None:
    assert K.load_ball_radius(tmp_path / "missing.yaml") == K.DEFAULT_BALL_RADIUS


def _ann(**kw) -> dict:
    base = dict(clip="k.npz", fps=30, kick_foot="right", entry_start=10, skill_start=20, contact_frame=30,
                skill_end=40, ball_offset_foot=[0.08, 0.0, 0.0])
    base.update(kw)
    return base


def test_annotation_loading_and_validation(tmp_path) -> None:
    p = tmp_path / "a.kick.json"
    p.write_text(json.dumps([_ann(), _ann(clip="sub/b.npz", kind="stand")]))
    loaded = K.load_kick_annotations(p)
    assert [c for c, _ in loaded] == [tmp_path / "k.npz", tmp_path / "sub" / "b.npz"]
    for bad in (_ann(contact_frame=50), _ann(kick_foot="middle"), _ann(ball_pos=[0, 0, 0]), _ann(kind="volley")):
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            K.load_kick_annotations(p)


def _entry(mirrored: bool, start: int, n_src: int = 41, n_db: int = 81) -> DatabaseEntry:
    feet = np.zeros((n_db, 2, 3))
    feet[:, 0] = [0.0, 0.1, 0.05]  # left
    feet[:, 1] = [0.0, -0.1, 0.05]  # right
    if mirrored:
        feet = feet[:, ::-1] * np.array([1.0, -1.0, 1.0])
    rot = np.tile([1.0, 0.0, 0.0, 0.0], (n_db, 1))
    return DatabaseEntry("clip.npz", mirrored, start, start + n_db, n_src, rot, feet)


def test_kick_ranges_map_frames_and_mirror() -> None:
    ann = K.KickClipAnnotation(**_ann())
    entries = [_entry(False, 0), _entry(True, 81)]
    ranges = K.kick_ranges_from_entries([("clip.npz", ann)], entries, ball_radius=0.07)
    assert [r.kick_foot for r in ranges] == ["right", "left"]
    # 41 source frames -> 81 db frames: frame f -> 2f
    assert ranges[0].search == (20, 40, 80)
    assert ranges[0].contact_frame == 60
    assert ranges[1].search == (101, 121, 161)
    assert ranges[1].contact_frame == 141
    np.testing.assert_allclose(ranges[0].ball_pos, [0.08, -0.1, 0.07])
    np.testing.assert_allclose(ranges[1].ball_pos, [0.08, 0.1, 0.07])


def test_sidecar_roundtrip_and_select(tmp_path) -> None:
    ann = K.KickClipAnnotation(**_ann())
    sc = K.KickSidecar("g1", K.kick_ranges_from_entries([("clip.npz", ann)], [_entry(False, 0), _entry(True, 81)],
                                                        ball_radius=0.07), 0.07)
    path = K.sidecar_path_for(tmp_path / "database_g1_kick_run.bin")
    assert path.endswith("database_g1_kick_run.kicks.json")
    sc.save(path)
    back = K.KickSidecar.load(path)
    assert back == sc
    assert len(back.select(kick_foot="left")) == 1 and back.select(kind="stand") == []


def test_build_kick_fields_validate() -> None:
    f = K.build_kick_fields(num_frames=10, kick_enable=np.ones(10), commit_start=3, contact=6,
                            ball_pos=np.array([1.0, 0.0, 0.075]))
    assert f["kick_phase"].tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
    assert np.flatnonzero(f["contact_frame"]).tolist() == [6]
    assert f["kick_mode"].tolist() == [1] * 10
    data = dict(f, vel_cmd=np.zeros((10, 2), np.float32))
    K.validate_kick_npz(data)
    with pytest.raises(ValueError):
        K.validate_kick_npz(dict(data, vel_cmd=np.zeros((10, 15), np.float32)))
    broken = dict(data, kick_phase=np.array([0, 2, 0, 0, 0, 0, 2, 2, 2, 2], np.int8))
    with pytest.raises(ValueError):
        K.validate_kick_npz(broken)


def test_no_kick_fields() -> None:
    f = K.build_kick_fields(num_frames=5, kick_enable=np.zeros(5), commit_start=None, contact=None,
                            ball_pos=np.array([0.0, 6.0, 0.075]))
    assert not f["contact_frame"].any() and not f["kick_phase"].any()
    assert f["kick_mode"].tolist() == [0] * 5
    K.validate_kick_npz(f)


def test_resample_helpers() -> None:
    assert K.map_frame(0, 60, 50) == 0 and K.map_frame(59, 60, 50) == 49
    held = K.resample_hold(np.arange(6), 3)
    assert held.tolist() == [0, 2, 5]
