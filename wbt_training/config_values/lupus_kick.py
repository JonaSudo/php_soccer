"""Phase 3: walk-kick teacher for LUPUS (``exp:lupus-kick-teacher``).

One teacher tracks all motion-matching scenarios (walk_kick, stand_kick,
stand_ball_outside_zone, walk_ball_offpath, walk_no_ball) with privileged ball
state. The ball is a free rigid object (``robots/ball``), the neck is moved by
the head-scan generator (:mod:`wbt_training.utils.head_scan`), and three ball
rewards replace holosoma's object-tracking terms.

Motion NPZ contract (on top of holosoma's WBT format, written by the Phase-2
export):

* ``object_pos_w`` (T, 3), ``object_quat_w`` (T, 4, wxyz), ``object_lin_vel_w``
  (T, 3): ball reference. Constant rest pose up to the contact frame. Holosoma
  needs an object in every motion, so ``walk_no_ball`` parks the ball far away.
* ``kick_enable`` (T,) in {0, 1}
* ``kick_phase`` (T,) in {0, 1, 2}: walk/stand, run-up/commit, after contact
* ``contact_frame`` (T,) bool: True exactly at the contact frame

Motions without these fields still load (no kick: ``kick_enable`` = 0,
``kick_phase`` = 0), so walk-only data can be used for smoke runs.

Ball contact is detected without pairwise contact forces, which holosoma does
not expose on every backend: a robot-ball contact is a jump of the ball's
horizontal velocity while a robot body is within reach of the ball
(:class:`BallContactDetector`). Feet use their real collision boxes from
``lupus.urdf``; other bodies a sphere around the body origin.

Every hardware or league number used here comes from ``robots/ball/ball.yaml``
or ``robots/lupus``; the remaining training constants are marked as estimates.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg, MotionConfig, NoiseToInitialPoseConfig
from holosoma.config_types.experiment import ExperimentConfig, TrainingConfig
from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.config_types.randomization import (
    IsaacGymMaterialDR,
    IsaacSimMaterialDR,
    MaterialRandomizationConfig,
    MujocoMaterialDR,
    RandomizationManagerCfg,
    RandomizationTermCfg,
)
from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg
from holosoma.config_types.scene import PhysicsConfig, PhysXPhysicsConfig, RigidObjectConfig, SceneConfig
from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg
from holosoma.managers.reward.base import RewardTermBase
from holosoma.utils.rotations import quat_apply, quat_rotate_inverse

from wbt_training.config_values import lupus as L
from wbt_training.config_values.motion_command import PhpMotionCommand

logger = logging.getLogger(__name__)

ROBOTS_DIR = Path(__file__).resolve().parents[2] / "robots"
BALL_DIR = ROBOTS_DIR / "ball"


def _load_ball_spec() -> dict:
    spec_mod = importlib.util.spec_from_file_location("_build_ball", BALL_DIR / "build_ball.py")
    assert spec_mod is not None and spec_mod.loader is not None
    mod = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(mod)
    return mod.load_spec(BALL_DIR / "ball.yaml")


BALL = _load_ball_spec()
BALL_RADIUS: float = float(BALL["radius"])
BALL_NAME = "ball"
G = 9.81


# ---------------------------------------------------------------------------
# Foot collision boxes from lupus.urdf (the ball contact surface)
# ---------------------------------------------------------------------------
def _rpy_matrix(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def foot_boxes(urdf: Path = L.LUPUS_DIR / "lupus.urdf") -> dict[str, dict[str, np.ndarray]]:
    """Collision box of each ``*_foot_link``: center and rotation in the link frame, half extents."""
    root = ET.parse(urdf).getroot()
    out: dict[str, dict[str, np.ndarray]] = {}
    for link in root.findall("link"):
        name = link.get("name", "")
        if not name.endswith("_foot_link"):
            continue
        for col in link.findall("collision"):
            box = col.find("geometry/box")
            if box is None:
                continue
            origin = col.find("origin")
            xyz = [float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()]
            rpy = [float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]
            out[name] = {
                "center": np.array(xyz),
                "rot": _rpy_matrix(*rpy),
                "half": np.array([float(v) for v in box.get("size").split()]) / 2.0,
            }
    return out


FOOT_BOXES = foot_boxes()
FOOT_BODY_NAMES = ["left_foot_link", "right_foot_link"]


def closest_point_on_box(
    p_w: torch.Tensor, link_pos_w: torch.Tensor, link_quat_w: torch.Tensor, box: dict[str, np.ndarray]
) -> torch.Tensor:
    """Closest point (world) on a link's collision box to ``p_w``. Quaternions xyzw."""
    dev = p_w.device
    center = torch.tensor(box["center"], dtype=torch.float32, device=dev)
    rot = torch.tensor(box["rot"], dtype=torch.float32, device=dev)
    half = torch.tensor(box["half"], dtype=torch.float32, device=dev)
    p_link = quat_rotate_inverse(link_quat_w, p_w - link_pos_w, w_last=True)
    p_box = (p_link - center) @ rot  # = rot^T (p_link - center)
    q_box = torch.maximum(torch.minimum(p_box, half), -half)
    q_link = q_box @ rot.T + center
    return quat_apply(link_quat_w, q_link, w_last=True) + link_pos_w


def yaw_from_quat_xyzw(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# ---------------------------------------------------------------------------
# Kick fields from the motion NPZ
# ---------------------------------------------------------------------------
def frames_since_contact(contact: np.ndarray, clip_start: np.ndarray) -> np.ndarray:
    """Frames since the latest contact frame of the same clip, -1 before the first contact.

    ``contact`` and ``clip_start`` are (T,) bool; ``clip_start[i]`` marks the
    first frame of a source clip in the concatenated motion buffer.
    """
    t = len(contact)
    idx = np.arange(t)
    last_contact = np.maximum.accumulate(np.where(contact, idx, -1))
    start = np.maximum.accumulate(np.where(clip_start, idx, 0))
    valid = last_contact >= start
    return np.where(valid, idx - last_contact, -1)


def load_kick_fields(files: list[str], total_frames: int) -> dict[str, np.ndarray]:
    """Concatenate ``kick_enable``, ``kick_phase``, ``contact_frame`` over ``files`` (frame order)."""
    ke, kp, cf, starts = [], [], [], []
    missing: list[str] = []
    for path in files:
        with np.load(path) as data:
            n = int(data["joint_pos"].shape[0])
            if "kick_phase" in data.files:
                ke.append(np.asarray(data["kick_enable"], dtype=np.float32).reshape(n))
                kp.append(np.asarray(data["kick_phase"], dtype=np.int64).reshape(n))
                cf.append(np.asarray(data["contact_frame"], dtype=bool).reshape(n))
            else:
                missing.append(path)
                ke.append(np.zeros(n, dtype=np.float32))
                kp.append(np.zeros(n, dtype=np.int64))
                cf.append(np.zeros(n, dtype=bool))
            s = np.zeros(n, dtype=bool)
            s[0] = True
            if "motion_ends" in data.files:  # combined NPZ: several source clips
                ends = np.asarray(data["motion_ends"], dtype=bool).reshape(n)
                s[1:] |= ends[:-1]
            starts.append(s)
    if missing:
        logger.warning(f"{len(missing)} motion file(s) without kick fields, treated as no-kick: {missing[:3]}")
    out = {
        "kick_enable": np.concatenate(ke),
        "kick_phase": np.concatenate(kp),
        "contact_frame": np.concatenate(cf),
    }
    clip_start = np.concatenate(starts)
    if len(out["kick_phase"]) != total_frames:
        raise ValueError(
            f"kick fields have {len(out['kick_phase'])} frames but the motion buffer has {total_frames}; "
            "default-pose prepend/append must stay off for lupus_kick presets"
        )
    out["frames_since_contact"] = frames_since_contact(out["contact_frame"], clip_start)
    return out


class LupusKickMotionCommand(PhpMotionCommand):
    """``PhpMotionCommand`` plus kick fields, ball parking and the heading at contact.

    * ``kick_enable``, ``kick_phase``, ``frames_since_contact``: per env at its
      current motion frame.
    * Reset into a frame after contact (``kick_phase`` = 2) parks the ball
      ``park_distance`` behind the robot: the reference has no meaningful ball
      pose after the kick, and a ball spawned mid-flight would teach nothing.
    * ``contact_heading``: robot yaw latched at the contact frame, the reference
      direction for :func:`ball_kick_direction` ("Blickrichtung beim Kontakt").
    """

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = cfg.params or {}
        self.park_distance = float(params.get("park_distance", 5.0))
        self._kick: dict[str, torch.Tensor] | None = None
        self.contact_heading = torch.full((env.num_envs,), float("nan"), device=env.device)

    def setup(self) -> None:
        super().setup()
        fields = load_kick_fields(self._motion_files_for(self.motion), int(self.motion.time_step_total))
        self._kick = {
            k: torch.as_tensor(v, device=self.device)
            for k, v in fields.items()
        }
        self._kick["kick_enable"] = self._kick["kick_enable"].float()

    # ---------------------------------------------------------------- fields

    def _field(self, key: str) -> torch.Tensor:
        assert self._kick is not None, "LupusKickMotionCommand.setup() has not run"
        return self._kick[key][self.time_steps]

    @property
    def kick_enable(self) -> torch.Tensor:
        return self._field("kick_enable")

    @property
    def kick_phase(self) -> torch.Tensor:
        return self._field("kick_phase")

    @property
    def frames_since_contact(self) -> torch.Tensor:
        return self._field("frames_since_contact")

    # ------------------------------------------------------------- lifecycle

    def reset(self, env_ids: torch.Tensor | None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0 or self._kick is None:
            return
        self.contact_heading[env_ids] = float("nan")
        if not self.motion.has_object:
            return
        after = env_ids[self._kick["kick_phase"][self.time_steps[env_ids]] == 2]
        if after.numel() > 0:
            self._park_ball(after)

    def step(self) -> None:
        super().step()
        if self._kick is None:
            return
        fsc = self.frames_since_contact
        root_yaw = yaw_from_quat_xyzw(self._env.simulator.robot_root_states[:, 3:7])
        at_contact = fsc == 0
        self.contact_heading = torch.where(at_contact, root_yaw, self.contact_heading)
        self.contact_heading = torch.where(fsc < 0, torch.full_like(root_yaw, float("nan")), self.contact_heading)

    def _park_ball(self, env_ids: torch.Tensor) -> None:
        sim = self._env.simulator
        root = sim.robot_root_states[env_ids]
        yaw = yaw_from_quat_xyzw(root[:, 3:7])
        states = torch.zeros(env_ids.numel(), 13, device=self.device)
        states[:, 0] = root[:, 0] - self.park_distance * torch.cos(yaw)
        states[:, 1] = root[:, 1] - self.park_distance * torch.sin(yaw)
        states[:, 2] = BALL_RADIUS
        states[:, 6] = 1.0  # identity quaternion, xyzw
        sim.set_actor_states([self.object_name], env_ids, states)


# ---------------------------------------------------------------------------
# Observation term
# ---------------------------------------------------------------------------
def _kick_command(env: Any) -> LupusKickMotionCommand:
    cmd = env.command_manager.get_state("motion_command")
    if not hasattr(cmd, "kick_phase"):
        raise TypeError(
            f"ball terms need LupusKickMotionCommand, got {type(cmd).__name__}; point the command term at "
            "wbt_training.config_values.lupus_kick:LupusKickMotionCommand"
        )
    return cmd


def kick_enable(env: Any) -> torch.Tensor:
    """``kick_enable`` from the motion data, shape [num_envs, 1]."""
    return _kick_command(env).kick_enable.view(env.num_envs, 1)


# ---------------------------------------------------------------------------
# Ball state and contact detection
# ---------------------------------------------------------------------------
def ball_state_w(env: Any, name: str = BALL_NAME) -> torch.Tensor:
    """Ball root state [num_envs, 13] in world frame (xyzw quaternion)."""
    env_ids = torch.arange(env.num_envs, device=env.device)
    return env.simulator.get_actor_states([name], env_ids)


class BallContactDetector:
    """Robot-ball contact from the ball's velocity jump plus a proximity gate.

    ``update`` must be called once per control step. It returns a bool
    [num_envs] mask: the ball's horizontal velocity changed by more than
    ``dv_threshold`` since the previous step **and** a robot body is within
    ``margin`` of the ball surface (feet: collision box; other bodies: a sphere
    of ``body_radius`` around the body origin). Free rolling, ground friction
    and damping change the velocity far less per step; a bounce changes mostly
    the vertical part. Steps right after a reset or a ball teleport are skipped.
    """

    def __init__(
        self,
        env: Any,
        dv_threshold: float = 0.15,
        margin: float = 0.03,
        body_radius: float = 0.08,
        object_name: str = BALL_NAME,
    ):
        self.env = env
        self.dv_threshold = dv_threshold
        self.margin = margin
        self.body_radius = body_radius
        self.object_name = object_name
        body_names = list(env.simulator.body_names)
        self.foot_ids = [body_names.index(n) for n in FOOT_BODY_NAMES]
        self.other_ids = torch.tensor(
            [i for i, n in enumerate(body_names) if n not in FOOT_BODY_NAMES and not n.endswith("contact_point")],
            dtype=torch.long,
            device=env.device,
        )
        self.prev_vel = torch.zeros(env.num_envs, 2, device=env.device)
        self.valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None:
            self.valid.zero_()
        else:
            self.valid[env_ids] = False

    def foot_closest_points(self, ball_pos: torch.Tensor) -> list[torch.Tensor]:
        sim = self.env.simulator
        return [
            closest_point_on_box(ball_pos, sim._rigid_body_pos[:, i], sim._rigid_body_rot[:, i], FOOT_BOXES[n])
            for i, n in zip(self.foot_ids, FOOT_BODY_NAMES)
        ]

    def near(self, ball_pos: torch.Tensor) -> torch.Tensor:
        sim = self.env.simulator
        reach = BALL_RADIUS + self.margin
        near = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        for p in self.foot_closest_points(ball_pos):
            near |= torch.linalg.norm(p - ball_pos, dim=1) < reach
        if self.other_ids.numel() > 0:
            d = torch.linalg.norm(sim._rigid_body_pos[:, self.other_ids] - ball_pos[:, None], dim=-1)
            near |= (d < reach + self.body_radius).any(dim=1)
        return near

    def update(self) -> torch.Tensor:
        ball = ball_state_w(self.env, self.object_name)
        vel = ball[:, 7:9]
        jump = torch.linalg.norm(vel - self.prev_vel, dim=1) > self.dv_threshold
        cmd = self.env.command_manager.get_state("motion_command")
        teleported = getattr(cmd, "motion_end_reset", torch.zeros_like(self.valid))
        valid = self.valid & ~teleported.bool() & (self.env.episode_length_buf > 1)
        contact = jump & valid & self.near(ball[:, :3])
        self.prev_vel = vel.clone()
        self.valid[:] = True
        return contact


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------
def ball_kick_direction(
    env: Any,
    window_s: float = 0.5,
    max_speed: float = 4.0,
    lateral_weight: float = 1.0,
    object_name: str = BALL_NAME,
) -> torch.Tensor:
    """Ball speed along the robot heading at contact, minus the sideways part.

    Active for ``window_s`` after the contact frame (``kick_phase`` 2). The
    forward speed is capped at ``max_speed`` (plan risk table: limit instead of
    maximize, so stand kicks stay stable); backward speed counts negative.
    """
    cmd = _kick_command(env)
    fsc = cmd.frames_since_contact
    active = (fsc >= 0) & (fsc.float() * env.dt <= window_s)
    heading = torch.where(
        torch.isnan(cmd.contact_heading),
        yaw_from_quat_xyzw(env.simulator.robot_root_states[:, 3:7]),
        cmd.contact_heading,
    )
    v = ball_state_w(env, object_name)[:, 7:9]
    h = torch.stack([torch.cos(heading), torch.sin(heading)], dim=1)
    along = (v * h).sum(dim=1).clamp(-max_speed, max_speed)
    lateral = torch.abs(v[:, 0] * h[:, 1] - v[:, 1] * h[:, 0])
    return (along - lateral_weight * lateral) * active.float()


class BallUntouched(RewardTermBase):
    """1 for every robot-ball contact while no kick is under way (``kick_phase`` 0).

    Covers ``kick_enable`` = 0 (stand at the ball, waiting for the trigger) and
    the offpath / no-ball scenarios, where ``kick_phase`` stays 0. Use with a
    negative weight.
    """

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        p = cfg.params or {}
        self.detector = BallContactDetector(
            env,
            dv_threshold=float(p.get("dv_threshold", 0.15)),
            margin=float(p.get("margin", 0.03)),
            body_radius=float(p.get("body_radius", 0.08)),
        )

    def __call__(self, env: Any, **kwargs: Any) -> torch.Tensor:
        contact = self.detector.update()
        return (contact & (_kick_command(env).kick_phase == 0)).float()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self.detector.reset(env_ids)


class BallStepOn(RewardTermBase):
    """Vertical foot force while a foot presses on top of the ball (always active).

    A foot is "on" the ball when its sole box is within ``margin`` of the ball
    surface and the contact normal points up (the closest box point lies above
    the ball center by more than ``cos(max_normal_angle)`` of the distance).
    The penalty is the foot's vertical net contact force over ``force_scale``,
    capped at ``max_penalty``. Use with a negative weight.
    """

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        p = cfg.params or {}
        self.margin = float(p.get("margin", 0.01))
        self.cos_max = math.cos(float(p.get("max_normal_angle", math.radians(50.0))))
        self.force_scale = float(p.get("force_scale", 50.0))
        self.max_penalty = float(p.get("max_penalty", 2.0))
        self.detector = BallContactDetector(env)

    def on_ball(self, env: Any) -> torch.Tensor:
        """Bool [num_envs, 2]: foot presses on top of the ball."""
        ball = ball_state_w(env)[:, :3]
        out = []
        for p in self.detector.foot_closest_points(ball):
            d_vec = p - ball
            d = torch.linalg.norm(d_vec, dim=1)
            out.append((d < BALL_RADIUS + self.margin) & (d_vec[:, 2] > self.cos_max * d))
        return torch.stack(out, dim=1)

    def __call__(self, env: Any, **kwargs: Any) -> torch.Tensor:
        on = self.on_ball(env)
        fz = env.simulator.contact_forces[:, self.detector.foot_ids, 2].clamp_min(0.0)
        return (on.float() * (fz / self.force_scale)).sum(dim=1).clamp(max=self.max_penalty)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
# Tracked bodies: pelvis/torso (base_link), legs, arms. Not the head: the
# behavior moves it, the reference does not know where it looks.
LUPUS_TRACK_BODIES = [
    "base_link",
    "left_hip_link",
    "left_shin_link",
    "left_foot_link",
    "right_hip_link",
    "right_shin_link",
    "right_foot_link",
    "left_shoulder_link",
    "left_elbow_link",
    "left_forearm_link",
    "right_shoulder_link",
    "right_elbow_link",
    "right_forearm_link",
]
LUPUS_END_EFFECTORS = ["left_foot_link", "right_foot_link", "left_forearm_link", "right_forearm_link"]

lupus_kick_init_pose = NoiseToInitialPoseConfig(
    overall_noise_scale=1.0,
    dof_pos=0.1,
    root_pos=[0.05, 0.05, 0.01],
    root_rot=[0.1, 0.1, 0.2],
    root_lin_vel=[0.1, 0.1, 0.05],
    root_ang_vel=[0.1, 0.1, 0.1],
    # plan: ball start position +-3-5 cm around the reference
    object_pos=[0.04, 0.04, 0.0],
)

lupus_kick_motion_config = MotionConfig(
    # Spliced in by apply_terrain_preprocess from training.registry_name, or
    # set --command.setup_terms.motion_command.params.motion_config.motion_file
    motion_file="",
    body_names_to_track=LUPUS_TRACK_BODIES,
    body_name_ref=["base_link"],
    # plan: keep adaptive sampling on, hard skills do not converge without it
    use_adaptive_timesteps_sampler=True,
    adaptive_kernel_size=3,
    adaptive_lambda=0.8,
    adaptive_alpha=0.001,
    adaptive_uniform_ratio=1e-6,
    start_at_timestep_zero_prob=0.0,
    freeze_at_timestep_zero_prob=0.0,
    resample_on_motion_end=True,
    enable_default_pose_prepend=False,  # would misalign the per-frame kick fields
    enable_default_pose_append=False,
    noise_to_initial_pose=lupus_kick_init_pose,
)

_KICK_COMMAND_FUNC = "wbt_training.config_values.lupus_kick:LupusKickMotionCommand"
lupus_kick_command = CommandManagerCfg(
    params={},
    setup_terms={
        "motion_command": CommandTermCfg(
            func=_KICK_COMMAND_FUNC,
            params={"motion_config": lupus_kick_motion_config, "park_distance": 5.0},
        )
    },
    reset_terms={"motion_command": CommandTermCfg(func=_KICK_COMMAND_FUNC)},
    step_terms={"motion_command": CommandTermCfg(func=_KICK_COMMAND_FUNC)},
)

lupus_ball_scene = SceneConfig(
    env_spacing=0.0,
    rigid_objects={
        BALL_NAME: RigidObjectConfig(
            urdf_file=str(BALL_DIR / "ball.urdf"),
            xml_file=str(BALL_DIR / "ball.xml"),
            physics=PhysicsConfig(
                physx=PhysXPhysicsConfig(
                    linear_damping=float(BALL["linear_damping"]),
                    angular_damping=float(BALL["angular_damping"]),
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                ),
            ),
        )
    },
)


def _obs(func: str, noise: float = 0.0, **params: Any) -> ObsTermCfg:
    return ObsTermCfg(func=func, scale=1.0, noise=noise, params=params)


_WBT = "holosoma.managers.observation.terms.wbt"
_OBJ = "holosoma.managers.observation.terms.objects"
_ball_terms = {
    "object_pos_b": _obs(f"{_OBJ}:object_pos_b", object_names=[BALL_NAME]),
    "object_lin_vel_b": _obs(f"{_OBJ}:object_lin_vel_b", object_names=[BALL_NAME]),
    "kick_enable": _obs("wbt_training.config_values.lupus_kick:kick_enable"),
}

# Teacher actor: reference terms, base velocities, joints, last action, ball
# (privileged: exact position and velocity) and kick_enable; no height_scan.
lupus_kick_actor_obs = ObsGroupCfg(
    concatenate=True,
    enable_noise=True,
    history_length=1,
    terms={
        "motion_command": _obs(f"{_WBT}:motion_command"),
        "motion_ref_pos_b": _obs(f"{_WBT}:motion_ref_pos_b", 0.25),
        "motion_ref_ori_b": _obs(f"{_WBT}:motion_ref_ori_b", 0.05),
        "base_lin_vel": _obs(f"{_WBT}:base_lin_vel", 0.5),
        "base_ang_vel": _obs(f"{_WBT}:base_ang_vel", 0.2),
        "dof_pos": _obs(f"{_WBT}:dof_pos", 0.01),
        "dof_vel": _obs(f"{_WBT}:dof_vel", 0.5),
        "actions": _obs(f"{_WBT}:actions"),
        **_ball_terms,
    },
)

lupus_kick_critic_obs = ObsGroupCfg(
    concatenate=True,
    enable_noise=False,
    history_length=1,
    terms={
        "motion_command": _obs(f"{_WBT}:motion_command"),
        "motion_ref_pos_b": _obs(f"{_WBT}:motion_ref_pos_b"),
        "motion_ref_ori_b": _obs(f"{_WBT}:motion_ref_ori_b"),
        "robot_body_pos_b": _obs(f"{_WBT}:robot_body_pos_b"),
        "robot_body_ori_b": _obs(f"{_WBT}:robot_body_ori_b"),
        "base_lin_vel": _obs(f"{_WBT}:base_lin_vel"),
        "base_ang_vel": _obs(f"{_WBT}:base_ang_vel"),
        "dof_pos": _obs(f"{_WBT}:dof_pos"),
        "dof_vel": _obs(f"{_WBT}:dof_vel"),
        "actions": _obs(f"{_WBT}:actions"),
        **_ball_terms,
    },
)

lupus_kick_observation = ObservationManagerCfg(
    groups={"actor_obs": lupus_kick_actor_obs, "critic_obs": lupus_kick_critic_obs}
)


def _rew(func: str, weight: float, **params: Any) -> RewardTermCfg:
    return RewardTermCfg(func=func, weight=weight, params=params)


_RW = "holosoma.managers.reward.terms.wbt"
_UNDESIRED = "^(?!left_foot_contact_point$)(?!right_foot_contact_point$)(?!left_foot_link$)(?!right_foot_link$).+$"
# PHP terrain reward (global + relative tracking, regularizers) without the
# object_global_ref_* terms: the ball follows no reference after contact.
lupus_kick_reward = RewardManagerCfg(
    terms={
        "motion_global_ref_position_error_exp": _rew(f"{_RW}:motion_global_ref_position_error_exp", 1.0, sigma=0.3),
        "motion_global_ref_orientation_error_exp": _rew(
            f"{_RW}:motion_global_ref_orientation_error_exp", 1.0, sigma=0.4
        ),
        "motion_relative_body_position_error_exp": _rew(
            f"{_RW}:motion_relative_body_position_error_exp", 1.0, sigma=0.3
        ),
        "motion_relative_body_orientation_error_exp": _rew(
            f"{_RW}:motion_relative_body_orientation_error_exp", 1.0, sigma=0.4
        ),
        "motion_global_body_lin_vel": _rew(f"{_RW}:motion_global_body_lin_vel", 1.0, sigma=1.0),
        "motion_global_body_ang_vel": _rew(f"{_RW}:motion_global_body_ang_vel", 1.0, sigma=3.14),
        "action_rate_l2": _rew(f"{_RW}:penalty_action_rate", -0.1),
        "limits_dof_pos": _rew(f"{_RW}:limits_dof_pos", -10.0, soft_dof_pos_limit=0.9),
        "undesired_contacts": _rew(
            f"{_RW}:UndesiredContacts", -0.5, threshold=1.0, undesired_contacts_body_names=_UNDESIRED
        ),
        # Ball terms (weights are starting values, estimate)
        "ball_kick_direction": _rew(
            "wbt_training.config_values.lupus_kick:ball_kick_direction",
            1.0,
            window_s=0.5,
            max_speed=4.0,
            lateral_weight=1.0,
        ),
        "ball_untouched": _rew("wbt_training.config_values.lupus_kick:BallUntouched", -5.0, dv_threshold=0.15),
        "ball_step_on": _rew("wbt_training.config_values.lupus_kick:BallStepOn", -2.0, force_scale=50.0),
    }
)

# BadTracking with the PHP thresholds (0.5 m); object thresholds switched off
# by making them unreachable (the ball follows no reference after contact).
lupus_kick_termination = TerminationManagerCfg(
    terms={
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded", is_timeout=True
        ),
        "bad_tracking": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:BadTracking",
            params={
                "bad_ref_pos_threshold": 0.5,
                "bad_ref_ori_threshold": 0.8,
                "bad_motion_body_pos_threshold": 0.5,
                "body_names_to_track": LUPUS_TRACK_BODIES,
                "bad_motion_body_pos_body_names": LUPUS_END_EFFECTORS,
                "bad_object_pos_threshold": 1.0e6,
                "bad_object_ori_threshold": 1.0e6,
            },
        ),
    }
)

_LOCO = "holosoma.managers.randomization.terms.locomotion"
_OBJ_DR = "holosoma.managers.randomization.terms.objects"
_HEAD_SCAN = "wbt_training.utils.head_scan:HeadScanGenerator"
_m = float(BALL["mass"])

lupus_kick_setup_terms = {
    # Kunstrasen: static friction 0.4-1.3 as in PHP (check on real turf)
    "randomize_robot_rigid_body_material_startup": RandomizationTermCfg(
        func=f"{_LOCO}:randomize_robot_rigid_body_material_startup",
        params={"static_friction_range": [0.4, 1.3], "dynamic_friction_range": [0.4, 1.1], "restitution_range": [0.0, 0.3]},
    ),
    "randomize_base_com_startup": RandomizationTermCfg(
        func=f"{_LOCO}:randomize_base_com_startup",
        params={"base_com_range": {"x": [-0.02, 0.02], "y": [-0.03, 0.03], "z": [-0.03, 0.03]}, "enabled": True},
    ),
    "setup_dof_pos_bias": RandomizationTermCfg(
        func=f"{_LOCO}:setup_dof_pos_bias",
        params={"dof_pos_bias_range": [-0.01, 0.01], "dof_pos_bias_additive_overrides": {"ankle": [-0.03, 0.03]}, "enabled": True},
    ),
    # Pushes, PD gains, motor strength (random force injection) and action
    # delay: switched on for LUPUS (off in PHP). Delay 0-2 steps (0-40 ms at
    # 50 Hz) until the LUPUS latency is measured.
    "push_randomizer_state": RandomizationTermCfg(
        func=f"{_LOCO}:PushRandomizerState",
        params={"push_interval_s": [1.0, 3.0], "max_push_vel": [0.3, 0.3, 0.1, 0.3, 0.3, 0.4], "enabled": True},
    ),
    "actuator_randomizer_state": RandomizationTermCfg(
        func=f"{_LOCO}:ActuatorRandomizerState",
        params={
            "kp_range": [0.9, 1.1],
            "kd_range": [0.9, 1.1],
            "rfi_lim_range": [0.5, 1.5],
            "rfi_lim": 0.1,
            "enable_pd_gain": True,
            "enable_rfi_lim": True,
        },
    ),
    "setup_torque_rfi": RandomizationTermCfg(func=f"{_LOCO}:setup_torque_rfi", params={"enabled": True, "rfi_lim": 0.1}),
    "setup_action_delay_buffers": RandomizationTermCfg(
        func=f"{_LOCO}:setup_action_delay_buffers", params={"ctrl_delay_step_range": [0, 2], "enabled": True}
    ),
    # Ball: mass +-10 %, turf-like friction and restitution, damping as a
    # stand-in for rolling resistance (PhysX has no rolling friction).
    "randomize_object_rigid_body_material_startup": RandomizationTermCfg(
        func=f"{_OBJ_DR}:randomize_object_rigid_body_material_startup",
        params={
            "material": MaterialRandomizationConfig(
                isaacgym=IsaacGymMaterialDR(friction=[0.5, 1.0], restitution=[0.3, 0.7]),
                isaacsim=IsaacSimMaterialDR(static_friction=[0.5, 1.0], dynamic_friction=[0.4, 0.9], restitution=[0.3, 0.7]),
                mujoco=MujocoMaterialDR(sliding_friction=[0.5, 1.0]),
            ),
            "object_names": [BALL_NAME],
        },
    ),
    "randomize_object_rigid_body_mass_startup": RandomizationTermCfg(
        func=f"{_OBJ_DR}:randomize_object_rigid_body_mass_startup",
        params={"mass_distribution_params": [-0.1 * _m, 0.1 * _m], "object_names": [BALL_NAME]},
    ),
    "randomize_object_linear_damping_startup": RandomizationTermCfg(
        func=f"{_OBJ_DR}:randomize_object_linear_damping_startup",
        params={"damping_range": [0.1, 0.6], "object_names": [BALL_NAME]},
    ),
    "randomize_object_angular_damping_startup": RandomizationTermCfg(
        func=f"{_OBJ_DR}:randomize_object_angular_damping_startup",
        params={"damping_range": [0.1, 0.6], "object_names": [BALL_NAME]},
    ),
    "head_scan": RandomizationTermCfg(func=_HEAD_SCAN, params={}),
}

lupus_kick_reset_terms = {
    "push_randomizer_state": RandomizationTermCfg(func=f"{_LOCO}:PushRandomizerState"),
    "randomize_push_schedule": RandomizationTermCfg(func=f"{_LOCO}:randomize_push_schedule"),
    "randomize_action_delay": RandomizationTermCfg(func=f"{_LOCO}:randomize_action_delay"),
    "actuator_randomizer_state": RandomizationTermCfg(func=f"{_LOCO}:ActuatorRandomizerState"),
    "head_scan": RandomizationTermCfg(func=_HEAD_SCAN),
}

lupus_kick_step_terms = {
    "push_randomizer_state": RandomizationTermCfg(func=f"{_LOCO}:PushRandomizerState"),
    "apply_pushes": RandomizationTermCfg(func=f"{_LOCO}:apply_pushes"),
    "head_scan": RandomizationTermCfg(func=_HEAD_SCAN),
}

lupus_kick_randomization = RandomizationManagerCfg(
    setup_terms=lupus_kick_setup_terms,
    reset_terms=lupus_kick_reset_terms,
    step_terms=lupus_kick_step_terms,
)

# LUPUS with PHP's unit action scale (plan, Phase 0) and the passive neck term.
lupus_22dof_wbt = replace(
    L.lupus_22dof,
    control=replace(L.lupus_22dof.control, action_scale=1.0),
)


def build_lupus_kick_teacher(base: ExperimentConfig) -> ExperimentConfig:
    """``lupus_kick_teacher`` on top of PHP's ``g1_wbt_terrain_ref`` (algo, sim, logger, preprocess)."""
    return replace(
        base,
        training=replace(
            base.training,
            project="lupus-kick",
            name="lupus_kick_teacher",
            num_envs=4096,  # per GPU; no depth rendering, so scale up later
            # flat floor, no on-path obstacles; still resolves registry_name
            preprocess_hook_kwargs='{"add_onpath_obstacle": false}',
        ),
        robot=lupus_22dof_wbt,
        action=L.lupus_joint_pos,
        scene=lupus_ball_scene,
        command=lupus_kick_command,
        observation=lupus_kick_observation,
        reward=lupus_kick_reward,
        termination=lupus_kick_termination,
        randomization=lupus_kick_randomization,
        algo=replace(base.algo, config=replace(base.algo.config, num_learning_iterations=20000)),
    )


__all__ = [
    "BALL",
    "BALL_RADIUS",
    "BallContactDetector",
    "BallStepOn",
    "BallUntouched",
    "LupusKickMotionCommand",
    "ball_kick_direction",
    "build_lupus_kick_teacher",
    "closest_point_on_box",
    "foot_boxes",
    "frames_since_contact",
    "kick_enable",
    "load_kick_fields",
]
