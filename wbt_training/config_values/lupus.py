"""Holosoma robot config and locomotion test experiment for LUPUS.

LUPUS is described in ``robots/lupus`` (built by ``build_urdf.py`` and
``build_mjcf.py``). Limits, effort, velocity and armature come from
``robots/lupus/lupus_joints.yaml`` so this file never duplicates hardware
numbers; change them in ``hardware.yaml`` and rebuild.

Joint order (= MJCF depth-first order, required by the MuJoCo backend):
left leg, right leg, left arm, right arm, neck. The neck (``head_yaw``,
``head_pitch``) belongs to the behavior, not to the policy: holosoma ties the
action vector to all DoFs, so :class:`PassiveNeckJointPositionActionTerm`
keeps the two neck outputs but replaces them with a neck target (default pose,
later the head-scan generator) before the PD controller.

``exp:lupus-loco`` is the Phase-0 validation from the plan: a plain holosoma
velocity-tracking policy with the T1 locomotion recipe. It shows model errors
(masses, limits, gains, contacts) before the expensive tracking training and
gives a policy for the MuJoCo Sim2Sim check (``lupus.xml``).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import yaml
from holosoma.config_types.action import ActionManagerCfg, ActionTermCfg
from holosoma.config_types.command import CommandTermCfg
from holosoma.config_types.experiment import ExperimentConfig, NightlyConfig, TrainingConfig
from holosoma.config_types.robot import (
    RobotAssetConfig,
    RobotBridgeConfig,
    RobotConfig,
    RobotControlConfig,
    RobotInitState,
)
from holosoma.config_types.scene import PhysicsConfig, PhysXPhysicsConfig
from holosoma.config_values import algo, simulator, terrain
from holosoma.config_values.loco.t1 import (
    command as t1_command,
    curriculum as t1_curriculum,
    observation as t1_observation,
    randomization as t1_randomization,
    reward as t1_reward,
    termination as t1_termination,
)
from holosoma.managers.action.terms.joint_control import JointPositionActionTerm

LUPUS_DIR = Path(__file__).resolve().parents[2] / "robots" / "lupus"
_JOINTS = yaml.safe_load((LUPUS_DIR / "lupus_joints.yaml").read_text())["joints"]

LEG_DOF_NAMES = [
    f"{side}_{j}"
    for side in ("left", "right")
    for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
]
ARM_DOF_NAMES = [
    f"{side}_{j}" for side in ("left", "right") for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")
]
NECK_DOF_NAMES = ["head_yaw", "head_pitch"]
DOF_NAMES = LEG_DOF_NAMES + ARM_DOF_NAMES + NECK_DOF_NAMES
POLICY_DOF_NAMES = LEG_DOF_NAMES + ARM_DOF_NAMES

BODY_NAMES = [
    "base_link",
    *(
        f"{side}_{b}"
        for side in ("left", "right")
        for b in (
            "hip_link",
            "upper_thigh_link",
            "lower_thigh_link",
            "shin_link",
            "ankle_cross_link",
            "foot_link",
            "foot_contact_point",
        )
    ),
    *(f"{side}_{b}" for side in ("left", "right") for b in ("shoulder_link", "upper_arm_link", "elbow_link", "forearm_link")),
    "neck_link",
    "head_link",
]

# Signs (see hardware.yaml): hip/shoulder pitch negative = forward, knee
# positive = flex, ankle pitch positive = toes down, elbow negative = flex.
# Feet stay flat with ankle_pitch = -(hip_pitch + knee).
DEFAULT_JOINT_ANGLES = {
    **{f"{s}_hip_pitch": -0.2 for s in ("left", "right")},
    **{f"{s}_hip_roll": 0.0 for s in ("left", "right")},
    **{f"{s}_hip_yaw": 0.0 for s in ("left", "right")},
    **{f"{s}_knee": 0.4 for s in ("left", "right")},
    **{f"{s}_ankle_pitch": -0.2 for s in ("left", "right")},
    **{f"{s}_ankle_roll": 0.0 for s in ("left", "right")},
    **{f"{s}_shoulder_pitch": 0.0 for s in ("left", "right")},
    **{f"{s}_shoulder_roll": 0.0 for s in ("left", "right")},
    **{f"{s}_shoulder_yaw": 0.0 for s in ("left", "right")},
    **{f"{s}_elbow": -0.3 for s in ("left", "right")},
    "head_yaw": 0.0,
    "head_pitch": 0.0,
}
# Base height in DEFAULT_JOINT_ANGLES with the soles on the ground (MuJoCo FK
# of lupus.xml, 0.563 m; checked in tests/test_lupus_holosoma.py) plus 1 cm.
DEFAULT_BASE_HEIGHT = 0.573

# Mirror across the sagittal plane: in the base frame left and right joints
# share their axes, so every roll and yaw joint flips sign (same rule as
# LUPUS_SPEC in motion matching).
SYMMETRY_JOINT_NAMES = {
    n: (n.replace("left_", "right_", 1) if n.startswith("left_") else n.replace("right_", "left_", 1))
    for n in DOF_NAMES
}
FLIP_SIGN_JOINT_NAMES = [n for n in DOF_NAMES if n.endswith(("_roll", "_yaw"))]

# PD gains. Starting values in the K1/T1 range, scaled to LUPUS's motors
# (34 N*m X4-36 vs 45-60 N*m on T1). Ankle stiffness must stay above
# m*g*h ~ 50 N*m/rad per foot, otherwise LUPUS tips over as an inverted
# pendulum (robots/lupus/README.md).
STIFFNESS = {
    "hip_pitch": 100.0,
    "hip_roll": 100.0,
    "hip_yaw": 100.0,
    "knee": 100.0,
    "ankle_pitch": 100.0,
    "ankle_roll": 100.0,
    "shoulder": 20.0,
    "elbow": 20.0,
    "head": 5.0,
}
DAMPING = {
    "hip_pitch": 3.0,
    "hip_roll": 3.0,
    "hip_yaw": 3.0,
    "knee": 3.0,
    "ankle_pitch": 3.0,
    "ankle_roll": 3.0,
    "shoulder": 0.5,
    "elbow": 0.5,
    "head": 0.5,
}


def _per_joint(key: str) -> list[float]:
    return [float(_JOINTS[n][key]) for n in DOF_NAMES]


lupus_22dof = RobotConfig(
    num_bodies=len(BODY_NAMES),
    dof_obs_size=len(DOF_NAMES),
    actions_dim=len(DOF_NAMES),
    policy_obs_dim=-1,
    critic_obs_dim=-1,
    algo_obs_dim_dict={},
    key_bodies=["left_foot_contact_point", "right_foot_contact_point"],
    num_feet=2,
    foot_body_name="foot_link",
    foot_height_name="foot_contact_point",
    knee_name="shin_link",
    torso_name="base_link",
    dof_names=DOF_NAMES,
    upper_dof_names=ARM_DOF_NAMES + NECK_DOF_NAMES,
    upper_left_arm_dof_names=ARM_DOF_NAMES[:4],
    upper_right_arm_dof_names=ARM_DOF_NAMES[4:],
    lower_dof_names=LEG_DOF_NAMES,
    has_torso=True,
    has_upper_body_dof=True,
    left_ankle_dof_names=["left_ankle_pitch", "left_ankle_roll"],
    right_ankle_dof_names=["right_ankle_pitch", "right_ankle_roll"],
    knee_dof_names=["left_knee", "right_knee"],
    hips_dof_names=[n for n in LEG_DOF_NAMES if "_hip_" in n],
    dof_pos_lower_limit_list=_per_joint("lower"),
    dof_pos_upper_limit_list=_per_joint("upper"),
    dof_vel_limit_list=_per_joint("velocity"),
    dof_effort_limit_list=_per_joint("effort"),
    dof_armature_list=_per_joint("armature"),
    dof_joint_friction_list=[0.01] * len(DOF_NAMES),  # estimate, not measured
    body_names=BODY_NAMES,
    terminate_after_contacts_on=["base_link", "hip_link", "thigh", "shoulder", "upper_arm", "neck", "head"],
    penalize_contacts_on=["base_link", "hip_link", "thigh", "shin", "ankle_cross", "shoulder", "upper_arm", "elbow", "forearm", "head"],
    init_state=RobotInitState(
        pos=[0.0, 0.0, DEFAULT_BASE_HEIGHT],
        rot=[0.0, 0.0, 0.0, 1.0],
        lin_vel=[0.0, 0.0, 0.0],
        ang_vel=[0.0, 0.0, 0.0],
        default_joint_angles=DEFAULT_JOINT_ANGLES,
    ),
    randomize_link_body_names=["base_link"] + [b for b in BODY_NAMES if any(k in b for k in ("hip", "thigh", "shin", "ankle", "foot_link"))],
    waist_dof_names=[],
    waist_yaw_dof_name=None,
    waist_roll_dof_name=None,
    waist_pitch_dof_name=None,
    arm_dof_names=ARM_DOF_NAMES,
    left_arm_dof_names=ARM_DOF_NAMES[:4],
    right_arm_dof_names=ARM_DOF_NAMES[4:],
    symmetry_joint_names=SYMMETRY_JOINT_NAMES,
    flip_sign_joint_names=FLIP_SIGN_JOINT_NAMES,
    contact_pairs_multiplier=16,
    control=RobotControlConfig(
        control_type="P",
        stiffness=STIFFNESS,
        damping=DAMPING,
        # 0.25 as in the T1 locomotion recipe; the WBT presets use PHP's unit scale.
        action_scale=0.25,
        action_clip_value=100.0,
        clip_actions=True,
        clip_torques=True,
    ),
    asset=RobotAssetConfig(
        asset_root=str(LUPUS_DIR.parent),
        collapse_fixed_joints=True,
        replace_cylinder_with_capsule=True,  # same capsules as lupus.xml
        flip_visual_attachments=False,
        armature=0.001,
        thickness=0.01,
        urdf_file="lupus/lupus.urdf",  # IsaacSim converts it to USD at start-up
        usd_file=None,
        xml_file="lupus/lupus.xml",
        robot_type="lupus_22dof",
        enable_self_collisions=False,
        default_dof_drive_mode=3,
        fix_base_link=False,
        link_physics=PhysicsConfig(
            density=0.001,
            physx=PhysXPhysicsConfig(
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
            ),
        ),
    ),
    # No SDK bridge for LUPUS yet (own firmware); only used on the real robot.
    bridge=RobotBridgeConfig(sdk_type="ros2", motor_type="serial"),
    knee_joint_min_threshold=0.2,
)


class PassiveNeckJointPositionActionTerm(JointPositionActionTerm):
    """Joint position PD term whose neck entries are not driven by the policy.

    The policy still emits ``actions_dim`` values (holosoma ties actions to all
    DoFs), but the entries of ``params["passive_dof_names"]`` are replaced by
    :attr:`passive_targets` (offsets from the default pose, zero by default)
    before delay and PD. The head-scan generator (Phase 3) writes into
    :attr:`passive_targets`; on the robot the behavior commands the neck and the
    policy's neck outputs are ignored.
    """

    def __init__(self, cfg: ActionTermCfg, env: Any):
        super().__init__(cfg, env)
        names = list(cfg.params.get("passive_dof_names", NECK_DOF_NAMES))
        dof_names = list(env.robot_config.dof_names)
        self.passive_dof_ids = torch.tensor([dof_names.index(n) for n in names], dtype=torch.long, device=env.device)
        # in action units (offset / action_scale), so PD sees default + target
        self.passive_targets = torch.zeros(env.num_envs, len(names), device=env.device)

    def process_actions(self, actions: torch.Tensor) -> None:
        actions = actions.clone()
        actions[:, self.passive_dof_ids] = self.passive_targets
        super().process_actions(actions)


lupus_joint_pos = ActionManagerCfg(
    terms={
        "joint_control": ActionTermCfg(
            func="wbt_training.config_values.lupus:PassiveNeckJointPositionActionTerm",
            params={"passive_dof_names": NECK_DOF_NAMES},
            scale=1.0,
            clip=None,
        ),
    }
)

# Pose regularization per DoF in DOF_NAMES order, T1 recipe: hip pitch, knee
# and the ankles free to move; hip roll/yaw, arms and neck held.
_POSE_WEIGHTS = {
    "hip_pitch": 0.01,
    "hip_roll": 1.0,
    "hip_yaw": 5.0,
    "knee": 0.01,
    "ankle_pitch": 5.0,
    "ankle_roll": 5.0,
}
lupus_loco_reward = replace(
    t1_reward.t1_29dof_loco,
    terms={
        **t1_reward.t1_29dof_loco.terms,
        "pose": replace(
            t1_reward.t1_29dof_loco.terms["pose"],
            params={
                "pose_weights": [
                    next((w for k, w in _POSE_WEIGHTS.items() if n.endswith(k)), 50.0) for n in DOF_NAMES
                ]
            },
        ),
    },
)

# Plan: continuous (v, omega) without side steps.
_t1_cmd = t1_command.t1_29dof_command
lupus_loco_command = replace(
    _t1_cmd,
    setup_terms={
        **_t1_cmd.setup_terms,
        "locomotion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.locomotion:LocomotionCommand",
            params={
                "command_ranges": {
                    "lin_vel_x": [-0.5, 1.0],
                    "lin_vel_y": [0.0, 0.0],
                    "ang_vel_yaw": [-1.0, 1.0],
                    "heading": [-3.14, 3.14],
                },
                "stand_prob": 0.2,
            },
        ),
    },
)

lupus_loco = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="lupus-loco", name="lupus_loco"),
    algo=replace(algo.ppo, config=replace(algo.ppo.config, num_learning_iterations=25000, use_symmetry=True)),
    simulator=simulator.isaacsim,
    robot=lupus_22dof,
    terrain=terrain.terrain_locomotion_mix,
    observation=t1_observation.t1_29dof_loco_single_wolinvel,
    action=lupus_joint_pos,
    termination=t1_termination.t1_29dof_termination,
    randomization=t1_randomization.t1_29dof_randomization,
    command=lupus_loco_command,
    curriculum=t1_curriculum.t1_29dof_curriculum,
    reward=lupus_loco_reward,
    nightly=NightlyConfig(
        iterations=10000,
        metrics={"Episode/rew_tracking_ang_vel": [0.8, "inf"], "Episode/rew_tracking_lin_vel": [0.75, "inf"]},
    ),
)

__all__ = ["lupus_22dof", "lupus_joint_pos", "lupus_loco", "PassiveNeckJointPositionActionTerm"]
