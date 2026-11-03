"""Fixed-rate policy process for LUPUS (Jetson Orin), separate from YOLO.

The policy runs in its own process at a fixed rate (50 Hz). Each tick:

1. read the latest joint/IMU state from the firmware,
2. take the newest ball detection from the vision node (already queued by its
   own thread / callback via :meth:`PolicyNode.on_vision_frame`),
3. take (v, omega, kick_enable[, kick_dir, mode]) and the neck targets from the behavior,
4. run the ONNX student and send joint targets with the policy's kp/kd; the
   neck gets the behavior's targets.

Transport (ROS 2, serial, shared memory) is not fixed yet, so the three sides
are small protocols (:class:`Firmware`, :class:`Behavior`); the vision side
pushes frames into the node. :class:`LatencyStats` records loop period,
policy compute time and camera-to-policy latency, the numbers needed to adjust
the randomisation ranges of Phase 3/4.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from wbt_training.deploy.policy_io import BallInput, RobotState, StudentPolicy, command_vector


class Firmware(Protocol):
    def read_state(self) -> RobotState: ...

    def send_targets(self, q: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> None: ...


@dataclass
class BehaviorCommand:
    v: float = 0.0
    omega: float = 0.0
    kick_enable: bool = False
    kick_dir: float = 0.0
    mode: int | None = None
    neck: tuple[float, float] | None = None  # head_yaw, head_pitch targets (rad)


class Behavior(Protocol):
    def command(self) -> BehaviorCommand: ...


@dataclass
class _Series:
    values: list[float] = field(default_factory=list)

    def add(self, x: float) -> None:
        self.values.append(float(x))

    def stats(self) -> dict[str, float]:
        if not self.values:
            return {"n": 0}
        a = np.asarray(self.values)
        return {"n": len(a), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)), "p99": float(np.percentile(a, 99)), "max": float(a.max())}


@dataclass
class LatencyStats:
    loop_period: _Series = field(default_factory=_Series)
    compute: _Series = field(default_factory=_Series)
    vision_latency: _Series = field(default_factory=_Series)  # receive time - capture time
    overruns: int = 0

    def as_dict(self) -> dict:
        return {
            "loop_period_s": self.loop_period.stats(),
            "compute_s": self.compute.stats(),
            "vision_latency_s": self.vision_latency.stats(),
            "overruns": self.overruns,
        }


class PolicyNode:
    def __init__(
        self,
        policy: StudentPolicy,
        firmware: Firmware,
        behavior: Behavior,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.policy = policy
        self.firmware = firmware
        self.behavior = behavior
        self.clock = clock
        self.ball = BallInput(cap=float(policy.meta.ball_vision_cfg.get("time_since_seen_cap", 2.0)))
        self.stats = LatencyStats()
        self.names = policy.meta.joint_names
        self.neck_ids = [self.names.index(n) for n in policy.meta.passive_dof_names]
        self._last_tick: float | None = None

    def on_vision_frame(self, capture_time: float, found: bool, x: float = 0.0, y: float = 0.0) -> None:
        """Vision node callback (any thread). ``x, y`` in the policy's ball frame."""
        self.stats.vision_latency.add(self.clock() - capture_time)
        self.ball.on_frame(capture_time, found, x, y)

    def reset(self) -> None:
        self.policy.reset()
        self.ball.reset()

    def tick(self) -> np.ndarray:
        t0 = self.clock()
        if self._last_tick is not None:
            self.stats.loop_period.add(t0 - self._last_tick)
        self._last_tick = t0
        state = self.firmware.read_state()
        b = self.behavior.command()
        cmd = command_vector(b.v, b.omega, b.kick_enable, kick_dir=b.kick_dir, mode=b.mode, cfg=self.policy.meta.command_obs_cfg)
        q = self.policy.step(state, cmd, self.ball.observe(t0))
        if b.neck is not None:
            q[self.neck_ids] = b.neck
        self.firmware.send_targets(q, self.policy.meta.kp, self.policy.meta.kd)
        self.stats.compute.add(self.clock() - t0)
        return q

    def run(self, seconds: float | None = None, sleep: Callable[[float], None] = time.sleep) -> LatencyStats:
        """Fixed-rate loop with absolute deadlines (no drift); counts overruns."""
        dt = self.policy.meta.policy_dt
        start = self.clock()
        k = 0
        while seconds is None or k * dt < seconds:
            self.tick()
            k += 1
            deadline = start + k * dt
            slack = deadline - self.clock()
            if slack > 0:
                sleep(slack)
            else:
                self.stats.overruns += 1
                # resync instead of bursting to catch up
                start = self.clock() - k * dt
        return self.stats


__all__ = ["Behavior", "BehaviorCommand", "Firmware", "LatencyStats", "PolicyNode"]
