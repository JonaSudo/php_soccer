"""Parallel ankle of LUPUS: serial pitch/roll (policy, simulation) <-> two motors.

Policy and simulation act on the serial equivalent ``ankle_pitch``/``ankle_roll``
(see ``robots/lupus/hardware.yaml``, ``ankle_linkage``). The two RMD-X2-7 per
foot drive the foot through cranks (radius ``r``) and rods: for pitch both
motors turn the same way (rod lever ``d``), for roll they turn opposite (lateral
rod offset ``w``). This module does the conversion on the robot.

Model (rods vertical, attachment points at ``(d, +-w)`` from the ankle centre)::

    r * sin(m1) = d * sin(pitch) + w * sin(roll)
    r * sin(m2) = d * sin(pitch) - w * sin(roll)

For small angles this is the constant-ratio model the simulation assumes
(``tau_joint = 2 * tau_motor * ratio``, ratio ``d/r`` resp. ``w/r``), so
``model="linear"`` reproduces exactly what was trained and ``model="sine"``
adds the first-order geometric nonlinearity. Both are estimates until the
lever lengths are measured; ``signs`` and ``offsets`` map the model motor
angles onto the encoder convention of each motor (calibrate on the stand).

Velocities: ``m_dot = J q_dot``; torques: ``tau_joint = J^T tau_motor``.
PD gains: a joint-space PD ``K_q`` is a motor-space PD ``K_m = J^-T K_q J^-1``;
motors only take a diagonal gain, so :meth:`AnkleLinkage.motor_gains` returns
its diagonal and :meth:`AnkleLinkage.gain_coupling` how much is lost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from wbt_training.deploy.policy_io import RobotState

REPO = Path(__file__).resolve().parents[2]
HARDWARE_YAML = REPO / "robots" / "lupus" / "hardware.yaml"
SIDES = ("left", "right")


@dataclass
class AnkleLinkage:
    """One foot: ``q = (pitch, roll)`` <-> ``m = (motor_1, motor_2)``."""

    crank_radius_r: float = 0.040
    pitch_lever_d: float = 0.040
    roll_lever_w: float = 0.027
    model: Literal["linear", "sine"] = "linear"
    signs: tuple[float, float] = (1.0, 1.0)  # encoder direction per motor
    offsets: tuple[float, float] = (0.0, 0.0)  # encoder reading at pitch = roll = 0 (rad)

    # ------------------------------------------------------------- geometry
    def _model_motor(self, q: np.ndarray) -> np.ndarray:
        r, d, w = self.crank_radius_r, self.pitch_lever_d, self.roll_lever_w
        p, rl = float(q[0]), float(q[1])
        if self.model == "linear":
            return np.array([(d * p + w * rl) / r, (d * p - w * rl) / r])
        s = np.array([d * np.sin(p) + w * np.sin(rl), d * np.sin(p) - w * np.sin(rl)]) / r
        if np.any(np.abs(s) > 1.0):
            raise ValueError(f"ankle pose {q} outside the linkage workspace (crank would pass 90 deg)")
        return np.arcsin(s)

    def _model_jacobian(self, q: np.ndarray, m: np.ndarray) -> np.ndarray:
        r, d, w = self.crank_radius_r, self.pitch_lever_d, self.roll_lever_w
        if self.model == "linear":
            return np.array([[d, w], [d, -w]]) / r
        cp, cr = np.cos(q[0]), np.cos(q[1])
        c1, c2 = np.cos(m[0]), np.cos(m[1])
        return np.array([[d * cp / (r * c1), w * cr / (r * c1)], [d * cp / (r * c2), -w * cr / (r * c2)]])

    def _to_encoder(self, m_model: np.ndarray) -> np.ndarray:
        return np.asarray(self.signs) * m_model + np.asarray(self.offsets)

    def _from_encoder(self, m_enc: np.ndarray) -> np.ndarray:
        return (np.asarray(m_enc, float) - np.asarray(self.offsets)) / np.asarray(self.signs)

    # ------------------------------------------------------------- public
    def joint_to_motor(self, q: np.ndarray) -> np.ndarray:
        """Motor encoder angles for ``q = (pitch, roll)``."""
        return self._to_encoder(self._model_motor(np.asarray(q, float)))

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """``d m_encoder / d q`` at ``q`` (2x2)."""
        q = np.asarray(q, float)
        return np.diag(self.signs) @ self._model_jacobian(q, self._model_motor(q))

    def motor_to_joint(self, m: np.ndarray, q0: np.ndarray | None = None, tol: float = 1e-10, iters: int = 20) -> np.ndarray:
        """``(pitch, roll)`` from motor encoder angles (closed form for linear, Newton for sine)."""
        mm = self._from_encoder(m)
        r, d, w = self.crank_radius_r, self.pitch_lever_d, self.roll_lever_w
        if self.model == "linear":
            return np.array([r * (mm[0] + mm[1]) / (2 * d), r * (mm[0] - mm[1]) / (2 * w)])
        # sine model has a closed form too: sin(p) = r(s1+s2)/(2d), sin(roll) = r(s1-s2)/(2w)
        s1, s2 = np.sin(mm)
        sp, sr = r * (s1 + s2) / (2 * d), r * (s1 - s2) / (2 * w)
        if abs(sp) > 1 or abs(sr) > 1:
            raise ValueError(f"motor angles {m} have no ankle pose")
        return np.array([np.arcsin(sp), np.arcsin(sr)])

    def joint_vel_to_motor(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        return self.jacobian(q) @ np.asarray(qd, float)

    def motor_vel_to_joint(self, q: np.ndarray, md: np.ndarray) -> np.ndarray:
        return np.linalg.solve(self.jacobian(q), np.asarray(md, float))

    def motor_torque_to_joint(self, q: np.ndarray, tau_m: np.ndarray) -> np.ndarray:
        return self.jacobian(q).T @ np.asarray(tau_m, float)

    def joint_torque_to_motor(self, q: np.ndarray, tau_q: np.ndarray) -> np.ndarray:
        return np.linalg.solve(self.jacobian(q).T, np.asarray(tau_q, float))

    def motor_gain_matrix(self, q: np.ndarray, k_joint: np.ndarray) -> np.ndarray:
        """Motor-space stiffness/damping matrix equivalent to the joint diagonal ``k_joint``."""
        jinv = np.linalg.inv(self.jacobian(q))
        return jinv.T @ np.diag(np.asarray(k_joint, float)) @ jinv

    def motor_gains(self, q: np.ndarray, k_joint: np.ndarray) -> np.ndarray:
        """Diagonal motor gains (what a motor PD can take) for joint gains ``k_joint``."""
        return np.diag(self.motor_gain_matrix(q, k_joint)).copy()

    def gain_coupling(self, q: np.ndarray, k_joint: np.ndarray) -> float:
        """|off-diagonal| / diagonal of the motor gain matrix; 0 when the diagonal is exact.

        With the linear model it is 0 iff ``k_pitch / d^2 == k_roll / w^2``.
        """
        k = self.motor_gain_matrix(q, k_joint)
        return float(abs(k[0, 1]) / max(np.sqrt(abs(k[0, 0] * k[1, 1])), 1e-12))

    @classmethod
    def from_hardware(cls, path: Path | str = HARDWARE_YAML, **over) -> "AnkleLinkage":
        import yaml

        lk = yaml.safe_load(Path(path).read_text())["ankle_linkage"]
        kw = {k: float(lk[k]) for k in ("crank_radius_r", "pitch_lever_d", "roll_lever_w") if k in lk}
        kw.update(over)
        return cls(**kw)


@dataclass
class AnkleLinkageFirmware:
    """Wraps a firmware that speaks motor angles in the ankle slots.

    ``inner`` reports and accepts the policy's joint order, except that the
    slots of ``<side>_ankle_pitch`` / ``<side>_ankle_roll`` hold ``motor_1`` /
    ``motor_2`` of that foot (encoder angle, velocity; targets and gains in motor
    space). This wrapper converts both ways, so :class:`PolicyNode` sees the
    serial ankle it was trained on.
    """

    inner: object
    joint_names: list[str]
    linkages: dict[str, AnkleLinkage] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.linkages:
            self.linkages = {s: AnkleLinkage.from_hardware() for s in SIDES}
        self.slots = {s: (self.joint_names.index(f"{s}_ankle_pitch"), self.joint_names.index(f"{s}_ankle_roll")) for s in SIDES}
        self._q = {s: np.zeros(2) for s in SIDES}

    def read_state(self) -> RobotState:
        st = self.inner.read_state()
        q, qd = np.array(st.joint_pos, float), np.array(st.joint_vel, float)
        for s, (ip, ir) in self.slots.items():
            lk = self.linkages[s]
            qj = lk.motor_to_joint(q[[ip, ir]])
            qd[[ip, ir]] = lk.motor_vel_to_joint(qj, qd[[ip, ir]])
            q[[ip, ir]] = qj
            self._q[s] = qj
        return RobotState(q, qd, st.base_quat_wxyz, st.base_ang_vel)

    def send_targets(self, q: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> None:
        q, kp, kd = (np.array(a, float) for a in (q, kp, kd))
        for s, (ip, ir) in self.slots.items():
            lk = self.linkages[s]
            idx = [ip, ir]
            at = self._q[s]  # gains at the measured pose
            kp[idx], kd[idx] = lk.motor_gains(at, kp[idx]), lk.motor_gains(at, kd[idx])
            q[idx] = lk.joint_to_motor(q[idx])
        self.inner.send_targets(q, kp, kd)


__all__ = ["AnkleLinkage", "AnkleLinkageFirmware", "HARDWARE_YAML"]
