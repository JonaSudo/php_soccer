"""Phase 6: kick direction and dribble mode (fields, terms, rewards, presets). CPU only."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import tyro
from holosoma.utils.tyro_utils import TYRO_CONIFG

from wbt_training.config_values import experiment
from wbt_training.config_values import lupus_kick as K
from wbt_training.config_values import lupus_phase6 as P6
from wbt_training.config_values.kick_command_obs import StudentCommandObsTerm
from wbt_training.tests.test_lupus_kick import _fake_env


def _phase6_env(n=4, mode=1, kick_dir=0.0, **kw):
    env, term, cmd = _fake_env(n=n, **kw)
    cmd.kick_mode = torch.full((n,), mode, dtype=torch.long)
    cmd.kick_dir = torch.full((n,), float(kick_dir))
    return env, term, cmd


# ------------------------------------------------------------------ fields
def test_load_phase6_fields_concatenates_and_falls_back(tmp_path):
    a, b, c = tmp_path / "a.npz", tmp_path / "b.npz", tmp_path / "c.npz"
    np.savez(a, joint_pos=np.zeros((3, 5)), kick_dir=np.array([0.1, 0.2, 0.3]), kick_mode=np.array([2, 2, 2], np.int8))
    np.savez(b, joint_pos=np.zeros((2, 5)), kick_enable=np.array([0.0, 1.0]))
    np.savez(c, joint_pos=np.zeros((2, 5)))
    f = P6.load_phase6_fields([str(a), str(b), str(c)], 7)
    np.testing.assert_allclose(f["kick_dir"], [0.1, 0.2, 0.3, 0, 0, 0, 0], atol=1e-6)
    assert f["kick_mode"].tolist() == [2, 2, 2, 0, 1, 0, 0]
    with pytest.raises(ValueError):
        P6.load_phase6_fields([str(a)], 4)


def test_phase6_command_exposes_fields(monkeypatch, tmp_path):
    path = tmp_path / "m.npz"
    np.savez(path, joint_pos=np.zeros((4, 5)), kick_dir=np.array([0.0, 0.1, 0.2, 0.3]), kick_mode=np.array([0, 1, 2, 1]))

    def fake_setup(self):
        self._kick = {"kick_enable": torch.zeros(4)}

    monkeypatch.setattr(K.LupusKickMotionCommand, "setup", fake_setup)
    cmd = P6.LupusPhase6MotionCommand.__new__(P6.LupusPhase6MotionCommand)
    cmd.motion = SimpleNamespace(time_step_total=4)
    cmd.device = "cpu"
    cmd._motion_files_for = lambda motion: [str(path)]
    cmd.time_steps = torch.tensor([3, 1])
    cmd.setup()
    assert cmd.kick_dir.tolist() == pytest.approx([0.3, 0.1])
    assert cmd.kick_mode.tolist() == [1, 1]


# ------------------------------------------------------------ observations
def test_observation_terms_and_type_check():
    env, _, cmd = _phase6_env(n=3, kick_dir=0.2)
    cmd.kick_mode = torch.tensor([0, 1, 2])
    assert P6.kick_dir(env).shape == (3, 1) and P6.kick_dir(env)[0, 0].item() == pytest.approx(0.2)
    assert P6.kick_mode_onehot(env).tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    plain, _, _ = _fake_env(n=1)
    with pytest.raises(TypeError, match="LupusPhase6MotionCommand"):
        P6.kick_dir(plain)


def test_student_command_reads_phase6_fields():
    env, _, cmd = _phase6_env(n=2, mode=2, kick_dir=0.3)
    cmd.vel_cmd = torch.tensor([[0.5, 0.0], [0.5, 0.0]])
    params = {
        "mode": "mode3",
        "include_kick_dir": True,
        "kick_dir_max": P6.KICK_DIR_MAX,
        "commit_replace_prob": 0.0,
        "flicker_prob": 0.0,
        "v_noise_std": 0.0,
        "omega_noise_std": 0.0,
        "kick_dir_noise_std": 0.0,
    }
    term = StudentCommandObsTerm(SimpleNamespace(params=params), env)
    out = term(env)
    assert out.shape == (2, 2 + 1 + 3)
    assert out[0, 2].item() == pytest.approx(0.3)
    assert out[0, 3:].tolist() == [0, 0, 1]


# ----------------------------------------------------------------- rewards
def test_kick_direction_measures_against_heading_plus_kick_dir():
    env, _, cmd = _phase6_env(n=3, kick_dir=math.pi / 6)
    cmd.frames_since_contact = torch.tensor([5, 5, 5])
    cmd.contact_heading = torch.tensor([0.0, 0.0, 0.0])
    d = torch.tensor([math.pi / 6, 0.0, -math.pi / 6])
    env.simulator.ball[:, 7] = 2.0 * torch.cos(d)
    env.simulator.ball[:, 8] = 2.0 * torch.sin(d)
    r = P6.ball_kick_direction_dir(env)
    assert r[0].item() == pytest.approx(2.0, abs=1e-5)  # exactly along kick_dir
    assert r[0] > r[1] > r[2]
    cmd.kick_dir[:] = 0.0  # kick_dir 0 is the Phase-3 reward
    np.testing.assert_allclose(P6.ball_kick_direction_dir(env).numpy(), K.ball_kick_direction(env).numpy(), atol=1e-6)


def test_dribble_rewards_only_in_dribble_mode():
    foot = K.FOOT_BODY_NAMES[0]
    env, _, cmd = _phase6_env(n=2, ball_xy=(0.0, 0.0))
    names = list(env.simulator.body_names)
    env.simulator._rigid_body_pos[:, names.index(foot), :2] = torch.tensor([0.05, 0.0])
    env.simulator._rigid_body_pos[:, names.index(K.FOOT_BODY_NAMES[1]), :2] = torch.tensor([1.0, 0.0])
    cmd.kick_mode = torch.tensor([2, 1])
    near = P6.ball_near_foot(env, sigma=0.15)
    assert near[0].item() == pytest.approx(math.exp(-((0.05 / 0.15) ** 2)), rel=1e-4)
    assert near[1].item() == 0.0

    cmd.vel_cmd = torch.tensor([[0.5, 0.0], [0.5, 0.0]])
    env.simulator.ball[:, 7] = 0.5
    r = P6.ball_along_heading(env)
    assert r[0].item() == pytest.approx(1.0) and r[1].item() == 0.0
    env.simulator.ball[:, 8] = 0.4  # sideways roll costs
    assert P6.ball_along_heading(env)[0].item() == pytest.approx(1.0 - 0.5 * 0.4)


def test_ball_untouched_allows_contact_when_dribbling():
    env, _, cmd = _phase6_env(n=2, mode=0)
    cmd.kick_mode = torch.tensor([0, 2])
    term = P6.BallUntouchedUnlessDribble(SimpleNamespace(params={}), env)
    term.detector.update = lambda: torch.tensor([True, True])
    assert term(env).tolist() == [1.0, 0.0]


# ----------------------------------------------------------------- presets
@pytest.mark.parametrize(
    "name", ["lupus_kickdir_teacher", "lupus_kickdir_distill_finetune", "lupus_dribble_teacher", "lupus_dribble_distill_finetune"]
)
def test_cli_selects_phase6_presets(name):
    cfg = tyro.cli(experiment.AnnotatedExperimentConfig, args=[f"exp:{name.replace('_', '-')}"], config=TYRO_CONIFG)
    assert cfg == experiment.DEFAULTS[name]
    assert cfg.training.name == name
    assert cfg.command.setup_terms["motion_command"].func.endswith("lupus_phase6:LupusPhase6MotionCommand")
    assert cfg.reward.terms["ball_kick_direction"].func.endswith(":ball_kick_direction_dir")


def test_teacher_presets_add_inputs_and_rewards():
    kd, dr = experiment.lupus_kickdir_teacher, experiment.lupus_dribble_teacher
    for cfg in (kd, dr):
        for g in ("actor_obs", "critic_obs"):
            assert "kick_dir" in cfg.observation.groups[g].terms
    assert "kick_mode" not in kd.observation.groups["actor_obs"].terms
    assert "kick_mode" in dr.observation.groups["actor_obs"].terms
    assert "ball_near_foot" not in kd.reward.terms
    assert kd.reward.terms["ball_untouched"].func == experiment.lupus_kick_teacher.reward.terms["ball_untouched"].func
    assert dr.reward.terms["ball_untouched"].func.endswith(":BallUntouchedUnlessDribble")
    assert dr.reward.terms["ball_near_foot"].weight > 0 and dr.reward.terms["ball_along_heading"].weight > 0
    # everything else stays the Phase-3 teacher
    assert kd.algo == experiment.lupus_kick_teacher.algo and kd.robot == experiment.lupus_kick_teacher.robot


def test_distill_presets_match_their_teachers():
    for dribble in (False, True):
        t = experiment.lupus_dribble_teacher if dribble else experiment.lupus_kickdir_teacher
        s = experiment.lupus_dribble_distill_finetune if dribble else experiment.lupus_kickdir_distill_finetune
        teacher_terms = set(t.observation.groups["actor_obs"].terms)
        distill_teacher = set(s.observation.groups["teacher_obs"].terms) - {"which_motion"}
        assert distill_teacher == teacher_terms  # the student's teacher sees what the teacher was trained on
        student = s.observation.groups["actor_obs"].terms
        assert "kick_dir" not in student and "kick_mode" not in student
        p = student["command"].params
        assert p["include_kick_dir"] and p["kick_dir_max"] == pytest.approx(math.radians(30))
        assert p["mode"] == ("mode3" if dribble else "enable")
        assert set(s.reward.terms) == set(t.reward.terms)
