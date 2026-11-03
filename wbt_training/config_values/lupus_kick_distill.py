"""Presets ``lupus_kick_distill`` and ``lupus_kick_distill_finetune`` (plan, Phase 4).

Distil the Phase-3 walk-kick teacher (``lupus_kick_teacher``) into a student
that sees proprioception, the behavior's command and the noisy ball detection
(:mod:`ball_distillation`). The recipe is PHP's
(``g1_wbt_terrain_warp_distill[_finetune]``) without the depth camera:

* DAgger + PPO (``_finetune``): the PPO weight ramps 0 -> 0.9 over the first
  half of the 20k iterations, i.e. the DAgger weight falls to 0.1; PPO gate
  unchanged. The PPO reward is the teacher reward *including* the ball terms
  (``ball_kick_direction``, ``ball_untouched``, ``ball_step_on``), which matters
  more here than in PHP because success is measured at the ball.
* ``RelaxTerminationThresholds``: BadTracking 0.5 m -> 1 m (x2).
* Adaptive sampling off, scenarios sampled uniformly.
* Export: one ``policy.onnx`` (``BallDistillationPPO``/``BallDistillation``).

Start commands are in ``wbt_training/BALL_DISTILL.md``.
"""

from __future__ import annotations

from dataclasses import replace

from holosoma.config_types.algo import (
    DistillationAlgoConfig,
    DistillationConfig,
    DistillationPPOAlgoConfig,
    DistillationPPOConfig,
    StudentTeacherModuleConfig,
)
from holosoma.config_types.command import CommandTermCfg
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.observation import ObsTermCfg

from wbt_training.config_values.ball_distillation import (
    NO_DEPTH_BACKBONE,
    ball_detection_term,
    lupus_ball_student_actor_obs,
    make_ball_distill_observation,
    with_phase6_command,
)
from wbt_training.config_values.depth_distillation import (
    g1_29dof_wbt_terrain_distill_command,
    wbt_distill_relax_termination_term,
)
from wbt_training.config_values.lupus_kick import (
    BALL_NAME,
    BALL_RADIUS,
    LUPUS_END_EFFECTORS,
    lupus_kick_actor_obs,
    lupus_kick_critic_obs,
)

# PHP's PPO-gate parameters, everything except the motion config.
_PHP_GATE_PARAMS = {
    k: v
    for k, v in g1_29dof_wbt_terrain_distill_command.setup_terms["motion_command"].params.items()
    if k.startswith("ppo_gate")
}

_student_module = StudentTeacherModuleConfig(
    student_hidden_dims=[1024, 512, 256, 128],
    teacher_hidden_dims=[512, 256, 128],  # MUST match the teacher's actor (g1_wbt_terrain_ref)
    critic_hidden_dims=[512, 256, 128],
    activation="ELU",
    depth_backbone=NO_DEPTH_BACKBONE,
    depth_output_dim=0,
)

_finetune_algo = DistillationPPOAlgoConfig(
    _target_="wbt_training.utils.student_teacher.BallDistillationPPO",
    _recursive_=False,
    config=DistillationPPOConfig(
        module=_student_module,
        num_learning_epochs=2,
        num_mini_batches=96,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.001,
        learning_rate=3e-4,
        max_grad_norm=1.0,
        schedule="adaptive",
        desired_kl=0.01,
        num_steps_per_env=24,
        save_interval=500,
        init_noise_std=0.01,
        num_learning_iterations=20000,
        empirical_normalization=False,
        ppo_start_epoch=0,
        # ppo_coef reaches 0.9 (DAgger weight 0.1) after the first half.
        dagger_end_epoch=10000,
        distill_loss_type="mse",
        dagger_loss_coef=10.0,
        weight_decay=0.0,
    ),
)

_pure_algo = DistillationAlgoConfig(
    _target_="wbt_training.utils.student_teacher.BallDistillation",
    _recursive_=False,
    config=DistillationConfig(
        module=replace(_student_module, student_hidden_dims=[512, 256, 128]),
        num_learning_epochs=2,
        num_mini_batches=96,
        learning_rate=3e-4,
        max_grad_norm=1.0,
        num_steps_per_env=24,
        save_interval=500,
        init_noise_std=0.001,
        num_learning_iterations=10000,
        distill_loss_type="mse",
        distillation_warmup_steps=0,
    ),
)


def _which_motion_term() -> ObsTermCfg:
    return ObsTermCfg(
        func="wbt_training.config_values.motion_command:which_motion",
        scale=1.0,
        noise=0.0,
        params={"bad_motion_body_pos_body_names": tuple(LUPUS_END_EFFECTORS)},
    )


def lupus_kick_distill_observation(phase6_kick_dir: bool = False, phase6_dribble: bool = False):
    actor = replace(
        lupus_ball_student_actor_obs,
        terms={
            **lupus_ball_student_actor_obs.terms,
            "ball_detection": replace(ball_detection_term, params={**ball_detection_term.params, "object_name": BALL_NAME, "ball_radius": BALL_RADIUS}),
        },
    )
    teacher = replace(lupus_kick_actor_obs, terms={**lupus_kick_actor_obs.terms, "which_motion": _which_motion_term()})
    obs = make_ball_distill_observation(teacher, lupus_kick_critic_obs, actor_obs=actor)
    if phase6_kick_dir or phase6_dribble:
        obs = with_phase6_command(obs, kick_dir=phase6_kick_dir, dribble=phase6_dribble)
    return obs


def _distill_command(teacher: ExperimentConfig):
    term = teacher.command.setup_terms["motion_command"]
    motion_config = replace(term.params["motion_config"], use_adaptive_timesteps_sampler=False)
    params = {**term.params, "motion_config": motion_config, **_PHP_GATE_PARAMS}
    return replace(
        teacher.command,
        setup_terms={**teacher.command.setup_terms, "motion_command": CommandTermCfg(func=term.func, params=params)},
    )


def build_lupus_kick_distill(teacher: ExperimentConfig, *, pure: bool = False, **obs_kwargs) -> ExperimentConfig:
    """Student preset from the teacher preset (same robot, scene, rewards, DR)."""
    curriculum = replace(
        teacher.curriculum,
        step_terms={**teacher.curriculum.step_terms, "relax_termination": wbt_distill_relax_termination_term},
    )
    return replace(
        teacher,
        training=replace(
            teacher.training,
            name="lupus_kick_distill" if pure else "lupus_kick_distill_finetune",
            num_envs=4096,  # no depth rendering, same budget as the teacher
        ),
        algo=_pure_algo if pure else _finetune_algo,
        command=_distill_command(teacher),
        observation=lupus_kick_distill_observation(**obs_kwargs),
        curriculum=curriculum,
    )


__all__ = ["build_lupus_kick_distill", "lupus_kick_distill_observation"]
