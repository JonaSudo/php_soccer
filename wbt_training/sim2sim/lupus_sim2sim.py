"""LUPUS Sim2Sim: run an exported holosoma locomotion policy in plain MuJoCo.

Holosoma's own sim2sim path (``run_sim.py`` + ``holosoma_inference``) needs a
robot SDK bridge (unitree / booster). LUPUS has none yet, so this module
replays the training loop of ``exp:lupus-loco`` directly:

* ``robots/lupus/lupus.xml`` at ``sim_dt`` (200 Hz), PD torques recomputed in
  every substep, ``decimation`` substeps per policy step (50 Hz);
* the actor observation exactly as holosoma's ``t1_29dof_loco_single_wolinvel``
  builds it (terms concatenated in alphabetical order, no history);
* the gait phase clock of ``LocomotionGait`` in eval mode (left 0, right -pi,
  ``stand_phase_value`` when the command is ~zero);
* the neck masked like ``PassiveNeckJointPositionActionTerm``.

Run time needs only ``numpy``, ``mujoco`` and (for an ONNX policy)
``onnxruntime``. Gains and action scale are read from the ONNX metadata that
holosoma's exporter attaches; everything else comes from :class:`LupusSim2SimConfig`.

Example::

    python -m wbt_training.sim2sim.lupus_sim2sim --onnx model_25000.onnx --vx 0.5 --seconds 10
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
LUPUS_DIR = REPO_ROOT / "robots" / "lupus"

# Kept in sync with wbt_training/config_values/lupus.py (checked by
# wbt_training/tests/test_lupus_sim2sim.py); duplicated so this module imports
# without holosoma / torch.
NECK_DOF_NAMES = ("head_yaw", "head_pitch")
DEFAULT_JOINT_ANGLES = {
    "left_hip_pitch": -0.2,
    "left_knee": 0.4,
    "left_ankle_pitch": -0.2,
    "right_hip_pitch": -0.2,
    "right_knee": 0.4,
    "right_ankle_pitch": -0.2,
    "left_elbow": -0.3,
    "right_elbow": -0.3,
}
STIFFNESS = {
    "hip": 100.0,
    "knee": 100.0,
    "ankle": 100.0,
    "shoulder": 20.0,
    "elbow": 20.0,
    "head": 5.0,
}
DAMPING = {
    "hip": 3.0,
    "knee": 3.0,
    "ankle": 3.0,
    "shoulder": 0.5,
    "elbow": 0.5,
    "head": 0.5,
}

# Alphabetical, as holosoma's ObservationManager concatenates (name, scale).
OBS_TERMS = (
    ("actions", 1.0),
    ("base_ang_vel", 1.0),
    ("command_ang_vel", 1.0),
    ("command_lin_vel", 1.0),
    ("cos_phase", 1.0),
    ("dof_pos", 1.0),
    ("dof_vel", 0.1),
    ("projected_gravity", 1.0),
    ("sin_phase", 1.0),
)


def gains_for(dof_names: Sequence[str], table: dict[str, float]) -> np.ndarray:
    """Substring match like holosoma's ``get_control_gains_from_config`` (exactly one key per DoF)."""
    out = []
    for name in dof_names:
        hits = [k for k in table if k in name]
        if len(hits) != 1:
            raise ValueError(f"expected exactly one gain key in {name!r}, got {hits}")
        out.append(float(table[hits[0]]))
    return np.asarray(out)


@dataclasses.dataclass
class LupusSim2SimConfig:
    xml_path: Path = LUPUS_DIR / "lupus.xml"
    sim_dt: float = 1.0 / 200.0
    decimation: int = 4
    action_scale: float | Sequence[float] = 0.25
    action_clip: float = 100.0
    stiffness: dict[str, float] = dataclasses.field(default_factory=lambda: dict(STIFFNESS))
    damping: dict[str, float] = dataclasses.field(default_factory=lambda: dict(DAMPING))
    default_joint_angles: dict[str, float] = dataclasses.field(default_factory=lambda: dict(DEFAULT_JOINT_ANGLES))
    passive_dof_names: tuple[str, ...] = NECK_DOF_NAMES
    gait_period: float = 1.0
    stand_phase_value: float = float(np.pi)
    init_base_height: float = 0.573
    # fall detection for the metrics (holosoma terminates on base contact)
    min_base_height: float = 0.3

    @property
    def policy_dt(self) -> float:
        return self.sim_dt * self.decimation


def apply_onnx_metadata(cfg: LupusSim2SimConfig, meta: dict[str, str], dof_names: Sequence[str]) -> LupusSim2SimConfig:
    """Take kp/kd/action_scale from holosoma's exported metadata and check the DoF order."""
    if "dof_names" in meta:
        exported = json.loads(meta["dof_names"])
        if list(exported) != list(dof_names):
            raise ValueError(f"policy DoF order {exported} != MJCF order {list(dof_names)}")
    changes: dict = {}
    if "kp" in meta and "kd" in meta:
        kp, kd = json.loads(meta["kp"]), json.loads(meta["kd"])
        changes["stiffness"] = {n: float(v) for n, v in zip(dof_names, kp)}
        changes["damping"] = {n: float(v) for n, v in zip(dof_names, kd)}
    if "action_scale" in meta:
        changes["action_scale"] = json.loads(meta["action_scale"])
    return dataclasses.replace(cfg, **changes)


def _exact_gains(dof_names: Sequence[str], table: dict[str, float]) -> np.ndarray:
    if set(table) == set(dof_names):
        return np.asarray([table[n] for n in dof_names], dtype=float)
    return gains_for(dof_names, table)


def quat_rotate_inverse_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world vector ``v`` into the frame of unit quaternion ``q`` (w, x, y, z)."""
    w, xyz = q[0], q[1:]
    return v * (2.0 * w * w - 1.0) - np.cross(xyz, v) * w * 2.0 + xyz * np.dot(xyz, v) * 2.0


class GaitClock:
    """``LocomotionGait`` in eval mode: phase = k * 2 pi dt / T + offset, wrapped to [-pi, pi)."""

    def __init__(self, policy_dt: float, period: float, stand_value: float):
        self.phase_dt = 2.0 * np.pi * policy_dt / period
        self.offset = np.array([0.0, -np.pi])
        self.stand_value = stand_value
        self.phase = self.offset.copy()

    def reset(self) -> None:
        self.phase = self.offset.copy()

    def update(self, step: int, command: np.ndarray) -> None:
        p = step * self.phase_dt + self.offset
        self.phase = np.fmod(p + np.pi, 2.0 * np.pi) - np.pi
        if np.linalg.norm(command[:2]) < 0.01 and abs(command[2]) < 0.01:
            self.phase = np.full(2, self.stand_value)


class LupusSim2Sim:
    """MuJoCo LUPUS + PD + holosoma observation, driven by any ``policy(obs) -> action``."""

    def __init__(self, cfg: LupusSim2SimConfig | None = None):
        import mujoco

        self.mujoco = mujoco
        self.cfg = cfg or LupusSim2SimConfig()
        self.model = mujoco.MjModel.from_xml_path(str(self.cfg.xml_path))
        self.model.opt.timestep = self.cfg.sim_dt
        self.data = mujoco.MjData(self.model)
        m = self.model
        if m.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("first joint of the LUPUS MJCF must be the free root joint")
        self.dof_names = [m.joint(i).name for i in range(1, m.njnt)]
        self.num_dof = len(self.dof_names)
        self.qpos_adr = np.array([m.jnt_qposadr[i] for i in range(1, m.njnt)])
        self.qvel_adr = np.array([m.jnt_dofadr[i] for i in range(1, m.njnt)])
        # actuators drive the joints in DoF order (checked once)
        act_joints = [m.joint(m.actuator_trnid[a, 0]).name for a in range(m.nu)]
        if act_joints != self.dof_names:
            raise ValueError(f"actuator order {act_joints} != joint order {self.dof_names}")
        self.torque_limit = np.array([m.actuator_ctrlrange[a, 1] for a in range(m.nu)])
        self.passive_ids = np.array([self.dof_names.index(n) for n in self.cfg.passive_dof_names], dtype=int)
        self.configure(self.cfg)
        self.command = np.zeros(3)
        self.reset()

    # -- configuration -------------------------------------------------------
    def configure(self, cfg: LupusSim2SimConfig) -> None:
        self.cfg = cfg
        self.default_pos = np.array([cfg.default_joint_angles.get(n, 0.0) for n in self.dof_names])
        self.kp = _exact_gains(self.dof_names, cfg.stiffness)
        self.kd = _exact_gains(self.dof_names, cfg.damping)
        self.action_scale = np.broadcast_to(np.asarray(cfg.action_scale, dtype=float), (self.num_dof,)).copy()
        self.gait = GaitClock(cfg.policy_dt, cfg.gait_period, cfg.stand_phase_value)

    # -- state ---------------------------------------------------------------
    def reset(self, base_height: float | None = None) -> np.ndarray:
        mj = self.mujoco
        mj.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = [0.0, 0.0, self.cfg.init_base_height if base_height is None else base_height]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[self.qpos_adr] = self.default_pos
        mj.mj_forward(self.model, self.data)
        self.step_count = 0
        self.last_action = np.zeros(self.num_dof)
        self.gait.reset()
        self.gait.update(0, self.command)
        return self.observation()

    def set_command(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0) -> None:
        self.command = np.array([vx, vy, wz], dtype=float)

    @property
    def dof_pos(self) -> np.ndarray:
        return self.data.qpos[self.qpos_adr].copy()

    @property
    def dof_vel(self) -> np.ndarray:
        return self.data.qvel[self.qvel_adr].copy()

    @property
    def base_quat_wxyz(self) -> np.ndarray:
        return self.data.qpos[3:7].copy()

    @property
    def base_ang_vel(self) -> np.ndarray:
        # freejoint qvel[3:6] is already body-local
        return self.data.qvel[3:6].copy()

    @property
    def base_lin_vel(self) -> np.ndarray:
        """Base linear velocity in the base frame (for metrics; not observed by the actor)."""
        return quat_rotate_inverse_wxyz(self.base_quat_wxyz, self.data.qvel[:3])

    def projected_gravity(self) -> np.ndarray:
        return quat_rotate_inverse_wxyz(self.base_quat_wxyz, np.array([0.0, 0.0, -1.0]))

    def observation_terms(self) -> dict[str, np.ndarray]:
        return {
            "actions": self.last_action,
            "base_ang_vel": self.base_ang_vel,
            "command_ang_vel": self.command[2:3],
            "command_lin_vel": self.command[:2],
            "cos_phase": np.cos(self.gait.phase),
            "dof_pos": self.dof_pos - self.default_pos,
            "dof_vel": self.dof_vel,
            "projected_gravity": self.projected_gravity(),
            "sin_phase": np.sin(self.gait.phase),
        }

    def observation(self) -> np.ndarray:
        terms = self.observation_terms()
        return np.concatenate([terms[n] * s for n, s in OBS_TERMS]).astype(np.float32)

    @property
    def obs_dim(self) -> int:
        return 3 * self.num_dof + 3 + 1 + 2 + 2 + 3 + 2

    # -- stepping ------------------------------------------------------------
    def process_action(self, action: np.ndarray) -> np.ndarray:
        """Neck masking + clip, as ``PassiveNeckJointPositionActionTerm.process_actions``.

        The ``actions`` observation is holosoma's ``action_manager.action``: the
        raw policy output, before masking and clipping.
        """
        raw = np.asarray(action, dtype=float).reshape(self.num_dof).copy()
        self.last_action = raw
        a = raw.copy()
        a[self.passive_ids] = 0.0
        return np.clip(a, -self.cfg.action_clip, self.cfg.action_clip)

    def pd_torque(self, action: np.ndarray) -> np.ndarray:
        target = self.default_pos + action * self.action_scale
        tau = self.kp * (target - self.dof_pos) - self.kd * self.dof_vel
        return np.clip(tau, -self.torque_limit, self.torque_limit)

    def step(self, action: np.ndarray) -> np.ndarray:
        a = self.process_action(action)
        for _ in range(self.cfg.decimation):
            self.data.ctrl[:] = self.pd_torque(a)
            self.mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        self.gait.update(self.step_count, self.command)
        return self.observation()

    def fallen(self) -> bool:
        return bool(self.data.qpos[2] < self.cfg.min_base_height or self.projected_gravity()[2] > -0.5)


# -- policies ------------------------------------------------------------------


def load_onnx_policy(path: str | Path) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, str]]:
    """Return ``policy(obs) -> action`` and the ONNX metadata (holosoma exporter layout)."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    meta = dict(sess.get_modelmeta().custom_metadata_map)
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    def policy(obs: np.ndarray) -> np.ndarray:
        return sess.run([out_name], {in_name: obs.reshape(1, -1).astype(np.float32)})[0].reshape(-1)

    return policy, meta


def zero_policy(obs: np.ndarray, num_dof: int = 22) -> np.ndarray:
    return np.zeros(num_dof)


@dataclasses.dataclass
class RolloutResult:
    steps: int
    fell: bool
    fall_time: float | None
    mean_vx: float
    mean_vy: float
    mean_wz: float
    max_torque_ratio: float
    distance_xy: float

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def rollout(
    sim: LupusSim2Sim,
    policy: Callable[[np.ndarray], np.ndarray],
    command: Sequence[float] = (0.0, 0.0, 0.0),
    seconds: float = 10.0,
    stop_on_fall: bool = True,
) -> RolloutResult:
    """Run ``policy`` for ``seconds`` under a constant command and collect basic tracking metrics."""
    sim.set_command(*command)
    obs = sim.reset()
    start = sim.data.qpos[:2].copy()
    n = int(round(seconds / sim.cfg.policy_dt))
    v, w, tau_ratio = [], [], 0.0
    fell, fall_time, k = False, None, 0
    for k in range(1, n + 1):
        obs = sim.step(policy(obs))
        v.append(sim.base_lin_vel[:2])
        w.append(sim.base_ang_vel[2])
        tau_ratio = max(tau_ratio, float(np.max(np.abs(sim.data.ctrl) / sim.torque_limit)))
        if sim.fallen():
            fell, fall_time = True, k * sim.cfg.policy_dt
            if stop_on_fall:
                break
    v_arr = np.asarray(v) if v else np.zeros((1, 2))
    return RolloutResult(
        steps=k,
        fell=fell,
        fall_time=fall_time,
        mean_vx=float(v_arr[:, 0].mean()),
        mean_vy=float(v_arr[:, 1].mean()),
        mean_wz=float(np.mean(w)) if w else 0.0,
        max_torque_ratio=tau_ratio,
        distance_xy=float(np.linalg.norm(sim.data.qpos[:2] - start)),
    )


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--onnx", type=Path, help="exported holosoma policy; omitted -> zero actions (PD hold)")
    p.add_argument("--vx", type=float, default=0.0)
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--wz", type=float, default=0.0)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--xml", type=Path, default=LupusSim2SimConfig.xml_path)
    p.add_argument("--viewer", action="store_true", help="open the passive MuJoCo viewer (needs a display)")
    args = p.parse_args(argv)

    cfg = LupusSim2SimConfig(xml_path=args.xml)
    sim = LupusSim2Sim(cfg)
    if args.onnx:
        policy, meta = load_onnx_policy(args.onnx)
        sim.configure(apply_onnx_metadata(cfg, meta, sim.dof_names))
    else:
        policy = lambda obs: zero_policy(obs, sim.num_dof)  # noqa: E731

    if args.viewer:
        import time

        import mujoco.viewer

        sim.set_command(args.vx, args.vy, args.wz)
        obs = sim.reset()
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            t_end = time.time() + args.seconds
            while viewer.is_running() and time.time() < t_end:
                t0 = time.time()
                obs = sim.step(policy(obs))
                viewer.sync()
                time.sleep(max(0.0, sim.cfg.policy_dt - (time.time() - t0)))
        return 0

    result = rollout(sim, policy, (args.vx, args.vy, args.wz), args.seconds)
    print(json.dumps(result.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
