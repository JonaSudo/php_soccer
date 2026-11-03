"""Feasibility check of a LUPUS clip against the joint table (Phase 1, step 5).

Checks per joint: position limits, velocity limits (central differences) and
the torque an inverse-dynamics pass needs (``mj_inverse`` on the floating-base
model; the root wrench is ignored, so this is a lower bound for the stance leg
and an estimate for the swing leg). Also checks ground penetration.

Limits come from ``robots/lupus/lupus_joints.yaml``. Clips whose worst ratio
exceeds the thresholds are flagged; :func:`slowest_feasible_scale` finds the
time-scaling factor (speed) that brings velocities and torques back inside.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import mujoco
import numpy as np

from reference_data.clip_ops import LupusClip
from reference_data.lupus_kin import kinematics


@dataclass
class Thresholds:
    pos_margin: float = 0.0          # rad allowed beyond the joint stop
    vel_ratio: float = 1.0           # |qdot| / velocity limit
    torque_ratio: float = 1.0        # |tau| / peak torque
    max_penetration: float = 0.01    # m below the ground
    violation_fraction: float = 0.02  # tolerated fraction of frames per joint


@dataclass
class JointReport:
    name: str
    pos_violation: float      # rad, worst excursion beyond the stops
    vel_ratio: float          # worst |qdot| / limit
    torque_ratio: float       # worst |tau| / peak torque
    frac_pos: float
    frac_vel: float
    frac_torque: float


@dataclass
class FeasibilityReport:
    ok: bool
    worst_vel_ratio: float
    worst_torque_ratio: float
    worst_pos_violation: float
    max_penetration: float
    failing: list[str] = field(default_factory=list)
    joints: list[JointReport] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["joints"] = [asdict(j) for j in self.joints if j.vel_ratio > 0.8 or j.torque_ratio > 0.8 or j.pos_violation > 0]
        return d


def joint_velocities(clip: LupusClip) -> np.ndarray:
    return np.gradient(clip.dofs, 1.0 / clip.fps, axis=0)


def _smooth_derivative(x: np.ndarray, dt: float) -> np.ndarray:
    """d/dt with a Savitzky-Golay filter (window 7, cubic); edges use the
    filter's polynomial fit instead of one-sided differences, which spike."""
    from scipy.signal import savgol_filter

    if len(x) >= 7:
        return savgol_filter(x, 7, 3, deriv=1, delta=dt, axis=0, mode="interp")
    return np.gradient(x, dt, axis=0)


def stance_mask(clip: LupusClip, height: float = 0.02, speed: float = 0.25) -> dict[str, np.ndarray]:
    """Per foot: frames where the sole is on the ground (low and slow)."""
    fk = kinematics().clip_fk(clip.qpos)
    out = {}
    for side in ("left", "right"):
        low = fk[f"{side}_corners"][:, :, 2].min(axis=1) < height
        v = np.linalg.norm(np.gradient(fk[f"{side}_center"], 1.0 / clip.fps, axis=0)[:, :2], axis=1)
        out[side] = low & (v < speed)
    return out


REG = 1e-3  # regularisation of the corner forces (relative to N, N m)


def joint_torques(clip: LupusClip, contacts: bool = True, return_residual: bool = False):
    """(T, 22) inverse-dynamics joint torques in PHP DoF order.

    ``mj_inverse`` on the floating-base model puts the whole weight on the
    (virtual) root. With ``contacts`` the root wrench is instead carried by
    point forces at the four sole corners of each stance foot, solved per
    frame as a bounded least-squares problem (normal force >= 0, friction
    not limited). The joints then see ``qfrc_inverse[6:] - J^T F``, so the
    ankle torque reflects where the centre of pressure has to be.

    With ``return_residual`` the root wrench the feet cannot provide
    (centre of pressure outside the sole, flight phase) is returned as well,
    ``(T, 2)``: residual force [N] and moment [N m].
    """
    from scipy.optimize import lsq_linear

    kin = kinematics()
    m, d = kin.model, kin.data
    qmj = kin.php_to_mj(clip.qpos)
    T = len(qmj)
    dt = 1.0 / clip.fps
    qvel = np.zeros((T, m.nv))
    for t in range(T):
        a, b = max(t - 1, 0), min(t + 1, T - 1)
        if b == a:
            continue
        mujoco.mj_differentiatePos(m, qvel[t], (b - a) * dt, qmj[a], qmj[b])
    qacc = _smooth_derivative(qvel, dt)
    stance = stance_mask(clip) if contacts else {"left": np.zeros(T, bool), "right": np.zeros(T, bool)}
    # mj_inverse would add MuJoCo's own (soft) contact and joint-limit forces
    # for penetrating frames; the support is modelled explicitly below.
    flags = m.opt.disableflags
    m.opt.disableflags = flags | mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    tau = np.zeros((T, m.nv))
    residual = np.zeros((T, 2))
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    signs = np.array([[sx, sy, -1.0] for sx in (-1, 1) for sy in (-1, 1)])
    for t in range(T):
        d.qpos[:] = qmj[t]
        d.qvel[:] = qvel[t]
        d.qacc[:] = qacc[t]
        mujoco.mj_inverse(m, d)
        f = d.qfrc_inverse.copy()
        cols = []
        for side in ("left", "right"):
            if not stance[side][t]:
                continue
            foot = kin.feet[side]
            rot = d.geom_xmat[foot.geom].reshape(3, 3)
            for sgn in signs:
                p = d.geom_xpos[foot.geom] + rot @ (sgn * foot.half)
                mujoco.mj_jac(m, d, jacp, None, p, foot.body)
                cols.append(jacp.copy())  # (3, nv) per corner
        if cols:
            J = np.vstack(cols)  # (3k, nv), rows = world force components
            k = len(cols)
            lb = np.tile([-np.inf, -np.inf, 0.0], k)
            ub = np.full(3 * k, np.inf)
            # Tikhonov term: the corner forces are redundant (12 per foot for
            # 6 equations); without it internal forces blow up.
            A = np.vstack([J[:, :6].T, REG * np.eye(3 * k)])
            b = np.concatenate([f[:6], np.zeros(3 * k)])
            sol = lsq_linear(A, b, bounds=(lb, ub), method="bvls")
            f = f - J.T @ sol.x
        residual[t] = np.linalg.norm(f[:3]), np.linalg.norm(f[3:6])
        tau[t] = f
    m.opt.disableflags = flags
    tau_php = tau[:, 6:][:, kin.perm_mj_to_php]
    return (tau_php, residual) if return_residual else tau_php


def check(clip: LupusClip, th: Thresholds | None = None, torques: bool = True) -> FeasibilityReport:
    th = th or Thresholds()
    kin = kinematics()
    q = clip.dofs
    below = np.maximum(kin.lower - q, 0.0)
    above = np.maximum(q - kin.upper, 0.0)
    pos_exc = np.maximum(below, above)
    vel = np.abs(joint_velocities(clip)) / kin.vel_limit
    tau = np.abs(joint_torques(clip)) / kin.effort if torques else np.zeros_like(q)
    lowest = kin.clip_fk(clip.qpos)["lowest"]
    penetration = float(max(0.0, -lowest.min()))

    joints, failing = [], []
    for i, name in enumerate(kin.urdf_names):
        jr = JointReport(
            name,
            float(pos_exc[:, i].max()),
            float(vel[:, i].max()),
            float(tau[:, i].max()),
            float(np.mean(pos_exc[:, i] > th.pos_margin)),
            float(np.mean(vel[:, i] > th.vel_ratio)),
            float(np.mean(tau[:, i] > th.torque_ratio)),
        )
        joints.append(jr)
        if max(jr.frac_pos, jr.frac_vel, jr.frac_torque) > th.violation_fraction:
            failing.append(name)
    if penetration > th.max_penetration:
        failing.append("ground_penetration")
    return FeasibilityReport(
        ok=not failing,
        worst_vel_ratio=float(vel.max()),
        worst_torque_ratio=float(tau.max()),
        worst_pos_violation=float(pos_exc.max()),
        max_penetration=penetration,
        failing=failing,
        joints=joints,
    )


def slowest_feasible_scale(
    clip: LupusClip,
    th: Thresholds | None = None,
    candidates: tuple[float, ...] = (1.0, 0.9, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5),
) -> tuple[float | None, FeasibilityReport]:
    """Fastest speed factor among ``candidates`` whose clip passes :func:`check`.

    Position limits and penetration do not improve with slower playback; if
    those fail the clip needs re-retargeting (or rejection), and ``None`` is
    returned with the report of the slowest candidate.
    """
    report = None
    for f in candidates:
        report = check(clip.time_scale(f), th)
        if report.ok:
            return f, report
    return None, report
