"""LUPUS kick Sim2Sim: the exported ball student in MuJoCo with ball, head scan and vision noise.

Runs the deploy code path (:class:`wbt_training.deploy.runner.PolicyNode`)
against a simulated robot, so what is tested here is what runs on the Jetson:

* ``robots/lupus/lupus.xml`` plus a free ball (``robots/ball/ball.yaml``), PD at
  200 Hz in 50 Hz policy steps, torque limits from the MJCF actuators;
* the vision node simulated with the training noise model
  (:mod:`wbt_training.utils.ball_detection`): FOV of the head camera, depth and
  bearing noise, 25-40 Hz frames, latency and dropouts; frames reach the node
  after their latency, like the real vision process;
* a behavior stand-in: scripted (v, omega, kick_enable) with optional jitter
  (:mod:`wbt_training.utils.command_jitter`) and a head scan (triangle sweep,
  ball tracking when a kick is pending and the ball is closer than 1 m);
* kick metrics per scenario (:mod:`wbt_training.utils.kick_metrics`) and the
  PHP-style grid.

Examples::

    python -m wbt_training.sim2sim.lupus_kick_sim2sim --onnx model_20000/policy.onnx --approach 1.5 --v 0.5
    python -m wbt_training.sim2sim.lupus_kick_sim2sim --onnx ... --stand-kick --approach 0.2 --trigger 1.0
    python -m wbt_training.sim2sim.lupus_kick_sim2sim --onnx ... --grid --episodes 5 --out report.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import yaml

from wbt_training.deploy.policy_io import RobotState, StudentPolicy, quat_rotate_inverse_wxyz
from wbt_training.deploy.runner import BehaviorCommand, PolicyNode
from wbt_training.utils.ball_detection import (
    BallVisionCfg,
    camera_pose_w,
    in_field_of_view,
    noisy_camera_measurement,
    world_to_robot_xy,
)
from wbt_training.utils.command_jitter import CommandJitter, CommandJitterCfg
from wbt_training.utils.kick_metrics import KickEpisodeTracker, KickEvalGrid, KickMetricsCfg, summarize, write_report

REPO = Path(__file__).resolve().parents[2]
LUPUS_XML = REPO / "robots" / "lupus" / "lupus.xml"
BALL_YAML = REPO / "robots" / "ball" / "ball.yaml"
FOOT_BODIES = ("left_foot_link", "right_foot_link")
FOOT_OFFSET_Y = 0.133  # m, |y| of the feet in the default pose


@dataclasses.dataclass
class KickScenario:
    approach: float = 1.5  # m, ball ahead of the base at the start
    lateral: float = 0.0  # m, + = left, relative to the base (feet at +-0.133 m)
    v: float = 0.5
    omega: float = 0.0
    kick_enable: bool = True
    trigger_time: float | None = None  # stand kick: kick_enable rises at this time (v forced to 0)
    seconds: float = 6.0
    head_scan: bool = True
    head_pitch: float = 0.6  # rad, looking down while scanning
    jitter: dict | None = None
    seed: int = 0
    kick_dir: float = 0.0  # Phase 6: rad relative to the heading at contact
    mode: int | None = None  # Phase 6 policies with 3 modes: 0 ignore, 1 kick, 2 dribble (None: from kick_enable)


def _ball_spec() -> dict:
    spec = yaml.safe_load(BALL_YAML.read_text())
    return {k: (v["value"] if isinstance(v, dict) else v) for k, v in spec.items()}


def build_model(xml: Path = LUPUS_XML):
    """lupus.xml plus a free ball body named ``ball``."""
    import mujoco

    spec = mujoco.MjSpec.from_file(str(xml))
    b = _ball_spec()
    r, m = 0.5 * float(b["diameter"]), float(b["mass"])
    inertia = (2.0 / 3.0 if b.get("shell_inertia", True) else 0.4) * m * r * r
    body = spec.worldbody.add_body(name="ball", pos=[2.0, 0.0, r])
    body.add_freejoint(name="ball_free")
    body.add_geom(
        name="ball",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[r, 0, 0],
        friction=[float(b["friction"]), 0.005, 0.0001],
        mass=m,
        rgba=[0.95, 0.95, 0.95, 1.0],
    )
    body.explicitinertial = True
    body.ipos = [0.0, 0.0, 0.0]  # MjSpec otherwise leaves the COM at the body position offset
    body.iquat = [1.0, 0.0, 0.0, 0.0]
    body.mass = m
    body.inertia = [inertia, inertia, inertia]
    model = spec.compile()
    return model, r, float(b.get("linear_damping", 0.3))


def damp_ball(qvel_ball: np.ndarray, linear: float, angular: float, dt: float) -> None:
    """Linear and angular damping of the free ball's 6 velocities, in place (as PhysX does).

    PhysX damps both, so a ball rolling without slip (thin shell, I = 2/3 m r^2)
    slows down with ``v' = -(0.6 linear + 0.4 angular) v``. Damping only the
    linear part would roll ~1.7x further in MuJoCo than in IsaacSim.
    """
    qvel_ball[:3] *= max(0.0, 1.0 - linear * dt)
    qvel_ball[3:6] *= max(0.0, 1.0 - angular * dt)


class SimVision:
    """Camera frames at 25-40 Hz with latency/dropouts, delivered to ``on_frame`` after their latency."""

    def __init__(self, cfg: BallVisionCfg, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.rate = rng.uniform(*cfg.update_hz_range)
        self.latency = rng.uniform(*cfg.latency_s_range)
        self.dropout = rng.uniform(*cfg.dropout_range)
        self.bias = rng.uniform(*cfg.depth_bias_range)
        self.next_capture = rng.uniform(0.0, 1.0 / self.rate)
        self.pending: list[tuple[float, float, bool, float, float]] = []  # deliver_at, capture, found, x, y
        self.g = torch.Generator().manual_seed(int(rng.integers(1 << 31)))

    def capture(self, t: float, ball_w, head_pos, head_quat_xyzw, base_pos, base_quat_xyzw) -> None:
        while t >= self.next_capture:
            c = self.cfg
            T = lambda a: torch.as_tensor(np.asarray(a, float), dtype=torch.float32)[None]  # noqa: E731
            cam_p, cam_r = camera_pose_w(T(head_pos), T(head_quat_xyzw), c)
            p_cam = (cam_r.transpose(-1, -2) @ (T(ball_w) - cam_p).unsqueeze(-1)).squeeze(-1)
            found = bool(in_field_of_view(p_cam, c)[0]) and self.rng.random() >= self.dropout
            meas = noisy_camera_measurement(p_cam, cam_p, cam_r, torch.tensor([self.bias]), c, self.g)
            xy = world_to_robot_xy(meas, T(base_pos), T(base_quat_xyzw), c.frame)[0].tolist()
            self.pending.append((self.next_capture + self.latency, self.next_capture, found, xy[0], xy[1]))
            self.next_capture += 1.0 / self.rate

    def deliver(self, t: float, node: PolicyNode) -> None:
        keep = []
        for item in self.pending:
            if item[0] <= t:
                node.ball.on_frame(item[1], item[2], item[3], item[4])
                node.stats.vision_latency.add(item[0] - item[1])
            else:
                keep.append(item)
        self.pending = keep


class LupusKickSim2Sim:
    """MuJoCo LUPUS + ball driven through :class:`PolicyNode`."""

    def __init__(self, onnx: str | Path, xml: Path = LUPUS_XML, sim_dt: float = 0.005):
        import mujoco

        self.mj = mujoco
        self.model, self.ball_radius, self.ball_damping = build_model(xml)
        self.ball_ang_damping = float(_ball_spec().get("angular_damping", self.ball_damping))
        self.model.opt.timestep = sim_dt
        self.data = mujoco.MjData(self.model)
        self.policy = StudentPolicy(onnx)
        meta = self.policy.meta
        self.decimation = max(1, int(round(meta.policy_dt / sim_dt)))
        m = self.model
        names = [m.joint(i).name for i in range(m.njnt)]
        missing = [n for n in meta.joint_names if n not in names]
        if missing:
            raise ValueError(f"policy joints not in MJCF: {missing}")
        jid = [names.index(n) for n in meta.joint_names]
        self.qpos_adr = np.array([m.jnt_qposadr[j] for j in jid])
        self.qvel_adr = np.array([m.jnt_dofadr[j] for j in jid])
        act_j = {m.joint(m.actuator_trnid[a, 0]).name: a for a in range(m.nu)}
        self.act = np.array([act_j[n] for n in meta.joint_names])
        self.tau_lim = np.array([m.actuator_ctrlrange[a, 1] for a in self.act])
        bj = names.index("ball_free")
        self.ball_q, self.ball_v = m.jnt_qposadr[bj], m.jnt_dofadr[bj]
        self.base_q, self.base_v = 0, 0  # robot freejoint first
        self.head_id = m.body("head_link").id
        self.foot_ids = [m.body(n).id for n in FOOT_BODIES]
        self.robot_body_ids = [i for i in range(1, m.nbody) if m.body(i).name != "ball"]
        self.targets = meta.default_joint_pos.copy()
        self.kp, self.kd = meta.kp.copy(), meta.kd.copy()
        self.node = PolicyNode(self.policy, self, self, clock=lambda: self.t)
        self.t = 0.0
        self.cmd = BehaviorCommand()

    # -- Firmware protocol ---------------------------------------------------
    def read_state(self) -> RobotState:
        d = self.data
        return RobotState(
            joint_pos=d.qpos[self.qpos_adr].copy(),
            joint_vel=d.qvel[self.qvel_adr].copy(),
            base_quat_wxyz=d.qpos[3:7].copy(),
            base_ang_vel=d.qvel[3:6].copy(),
        )

    def send_targets(self, q, kp, kd) -> None:
        self.targets, self.kp, self.kd = np.asarray(q, float), np.asarray(kp, float), np.asarray(kd, float)

    # -- Behavior protocol ---------------------------------------------------
    def command(self) -> BehaviorCommand:
        return self.cmd

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _xyzw(q_wxyz):
        return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])

    def base_yaw(self) -> float:
        w, x, y, z = self.data.qpos[3:7]
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def reset(self, sc: KickScenario, base_height: float = 0.573) -> None:
        mj, d = self.mj, self.data
        mj.mj_resetData(self.model, d)
        d.qpos[:3] = [0.0, 0.0, base_height]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.qpos[self.qpos_adr] = self.policy.meta.default_joint_pos
        d.qpos[self.ball_q : self.ball_q + 3] = [sc.approach, sc.lateral, self.ball_radius]
        d.qpos[self.ball_q + 3 : self.ball_q + 7] = [1.0, 0.0, 0.0, 0.0]
        mj.mj_forward(self.model, d)
        self.t = 0.0
        self.targets = self.policy.meta.default_joint_pos.copy()
        self.node.reset()

    def head_targets(self, sc: KickScenario, ball_xy_b: np.ndarray) -> tuple[float, float]:
        dist = float(np.hypot(*ball_xy_b))
        if self.cmd.kick_enable and dist < 1.0:  # behavior rule: track the ball before a kick
            yaw = math.atan2(ball_xy_b[1], ball_xy_b[0])
            pitch = math.atan2(1.0 - self.ball_radius, max(dist, 0.05))
            return float(np.clip(yaw, -1.0, 1.0)), float(np.clip(pitch, -0.3, 0.79))
        if not sc.head_scan:
            return 0.0, sc.head_pitch
        period = 2.0
        tri = 2.0 * abs((self.t / period) % 1.0 - 0.5) * 2.0 - 1.0  # -1..1
        return 0.8 * tri, sc.head_pitch

    def _ball_b(self) -> np.ndarray:
        d = self.data
        rel = d.qpos[self.ball_q : self.ball_q + 3] - d.qpos[:3]
        yaw = self.base_yaw()
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([c * rel[0] + s * rel[1], -s * rel[0] + c * rel[1]])

    def _ball_near_robot(self) -> bool:
        p = self.data.qpos[self.ball_q : self.ball_q + 3]
        d = np.linalg.norm(self.data.xpos[self.robot_body_ids] - p, axis=1)
        return bool(d.min() < self.ball_radius + 0.15)

    def run(self, sc: KickScenario, tracker: KickEpisodeTracker | None = None, policy_override=None) -> dict:
        mj, d = self.mj, self.data
        rng = np.random.default_rng(sc.seed)
        vision = SimVision(BallVisionCfg.from_params(self.policy.meta.ball_vision_cfg), rng)
        jitter = CommandJitter(1, CommandJitterCfg.from_params({"dt": self.policy.meta.policy_dt, **sc.jitter}), generator=torch.Generator().manual_seed(sc.seed)) if sc.jitter else None
        self.reset(sc)
        tracker = tracker or KickEpisodeTracker(1, KickMetricsCfg(dt=self.policy.meta.policy_dt, foot_offset_y=FOOT_OFFSET_Y))
        dt = self.policy.meta.policy_dt
        prev_ball_v = np.zeros(2)
        steps = int(round(sc.seconds / dt))
        kicked_contact = False
        for k in range(steps):
            stand = sc.trigger_time is not None
            kick = sc.kick_enable and (not stand or self.t >= sc.trigger_time)
            v = 0.0 if stand else sc.v
            vel = np.array([v, sc.omega])
            if jitter is not None:
                vj, kj = jitter.apply(torch.tensor([vel], dtype=torch.float32), torch.tensor([float(kick)]))
                vel, kick = vj[0].numpy(), bool(kj[0] > 0.5)
            ball_b = self._ball_b()
            self.cmd = BehaviorCommand(
                v=float(vel[0]), omega=float(vel[1]), kick_enable=bool(kick), kick_dir=float(sc.kick_dir), mode=sc.mode
            )
            self.cmd.neck = self.head_targets(sc, ball_b)
            head_q = self._xyzw(d.xquat[self.head_id])
            vision.capture(self.t, d.qpos[self.ball_q : self.ball_q + 3].copy(), d.xpos[self.head_id].copy(), head_q, d.qpos[:3].copy(), self._xyzw(d.qpos[3:7]))
            vision.deliver(self.t, self.node)
            if policy_override is None:
                self.node.tick()
            else:
                self.send_targets(policy_override(self), self.policy.meta.kp, self.policy.meta.kd)
            for _ in range(self.decimation):
                q, qd = d.qpos[self.qpos_adr], d.qvel[self.qvel_adr]
                tau = np.clip(self.kp * (self.targets - q) - self.kd * qd, -self.tau_lim, self.tau_lim)
                d.ctrl[self.act] = tau
                # rolling loss on turf as linear damping (PhysX/MuJoCo have no rolling friction)
                damp_ball(d.qvel[self.ball_v : self.ball_v + 6], self.ball_damping, self.ball_ang_damping, self.model.opt.timestep)
                mj.mj_step(self.model, d)
            self.t += dt
            ball_v = d.qvel[self.ball_v : self.ball_v + 3].copy()
            contact = bool(np.linalg.norm(ball_v[:2] - prev_ball_v) > 0.15 and self._ball_near_robot())
            prev_ball_v = ball_v[:2].copy()
            ball_b = self._ball_b()
            pending = kick and (0.0 < ball_b[0] < 1.0) and abs(ball_b[1]) < 0.25
            phase = 2 if kicked_contact else (1 if pending else 0)
            kicked_contact |= contact and phase >= 1
            yaw = self.base_yaw()
            lin_b = quat_rotate_inverse_wxyz(d.qpos[3:7], d.qvel[:3])
            w, x, y, _ = d.qpos[3:7]
            tracker.step(
                {
                    "ball_pos_w": torch.tensor(np.array([d.qpos[self.ball_q : self.ball_q + 3]])),
                    "ball_vel_w": torch.tensor(np.array([ball_v])),
                    "base_pos_w": torch.tensor(np.array([d.qpos[:3]])),
                    "base_yaw": torch.tensor([yaw]),
                    "base_up_z": torch.tensor([1 - 2 * (x * x + y * y)]),
                    "base_fwd_vel": torch.tensor([lin_b[0]]),
                    "base_yaw_rate": torch.tensor([d.qvel[5]]),
                    "cmd_v": torch.tensor([v]),
                    "cmd_w": torch.tensor([sc.omega]),
                    "kick_phase": torch.tensor([phase]),
                    "contact": torch.tensor([contact]),
                    "target_heading": torch.tensor([yaw + sc.kick_dir]),
                    "kick_mode": torch.tensor([sc.mode if sc.mode is not None else int(sc.kick_enable)]),
                    "feet_pos_w": torch.tensor(np.array([d.xpos[self.foot_ids]])),
                }
            )
        tracker.finish([0], terminated=[bool(tracker.fell[0])])
        rec = dict(tracker.records[-1])
        rec["scenario"] = dataclasses.asdict(sc)
        return rec


def run_grid(sim: LupusKickSim2Sim, grid: KickEvalGrid, episodes: int, seconds: float = 6.0, jitter: dict | None = None) -> list[dict]:
    recs = []
    for a, l, s in grid.cells():
        for e in range(episodes):
            sc = KickScenario(approach=a, lateral=FOOT_OFFSET_Y + l, v=s, seconds=seconds, trigger_time=1.0 if s == 0.0 else None, jitter=jitter, seed=e)
            recs.append(sim.run(sc))
    return recs


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--approach", type=float, default=1.5)
    ap.add_argument("--lateral", type=float, default=0.0)
    ap.add_argument("--v", type=float, default=0.5)
    ap.add_argument("--omega", type=float, default=0.0)
    ap.add_argument("--no-kick", action="store_true")
    ap.add_argument("--stand-kick", action="store_true", help="stand at the ball, kick_enable rises at --trigger")
    ap.add_argument("--trigger", type=float, default=1.0)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--no-head-scan", action="store_true")
    ap.add_argument("--kick-dir-deg", type=float, default=0.0, help="Phase 6: kick direction relative to the heading")
    ap.add_argument("--dribble", action="store_true", help="Phase 6: mode 2 (dribble), no kick")
    ap.add_argument("--jitter", action="store_true", help="behavior-stack command jitter (defaults of CommandJitterCfg)")
    ap.add_argument("--grid", action="store_true", help="run the PHP-style grid instead of one scenario")
    ap.add_argument("--episodes", type=int, default=3, help="episodes per grid cell")
    ap.add_argument("--out", default=None, help="write a JSON report")
    args = ap.parse_args(argv)

    sim = LupusKickSim2Sim(args.onnx)
    jitter = {} if args.jitter else None
    if args.grid:
        grid = KickEvalGrid(episodes_per_cell=args.episodes)
        recs = run_grid(sim, grid, args.episodes, args.seconds, jitter)
        report = write_report(args.out, recs, grid) if args.out else {"summary": summarize(recs)}
        print(json.dumps(report["summary"], indent=1))
    else:
        sc = KickScenario(
            approach=args.approach,
            lateral=args.lateral,
            v=0.0 if args.stand_kick else args.v,
            omega=args.omega,
            kick_enable=not (args.no_kick or args.dribble),
            trigger_time=args.trigger if args.stand_kick else None,
            seconds=args.seconds,
            head_scan=not args.no_head_scan,
            jitter=jitter,
            kick_dir=math.radians(args.kick_dir_deg),
            mode=2 if args.dribble else None,
        )
        rec = sim.run(sc)
        rec["latency"] = sim.node.stats.as_dict()
        if args.out:
            Path(args.out).write_text(json.dumps(rec, indent=1, default=float))
        print(json.dumps(rec, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
