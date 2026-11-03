"""Synthetic LUPUS clips for tests and smoke runs (no mocap data needed).

``stand_kick_clip`` builds a kinematic kick from the default stance: quiet
stance, backswing, kick, return, with a ball touching the kick foot at the
moment of peak foot speed and rolling away afterwards. ``walk_clip`` builds
a crude in-place stepping gait. Neither is meant as training data.
"""

from __future__ import annotations

import numpy as np

from reference_data.clip_ops import LupusClip
from reference_data.lupus_kin import kinematics

DEFAULT_POSE = {
    "left_hip_pitch": -0.2, "right_hip_pitch": -0.2,
    "left_knee": 0.4, "right_knee": 0.4,
    "left_ankle_pitch": -0.2, "right_ankle_pitch": -0.2,
    "left_elbow": -0.3, "right_elbow": -0.3,
}


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _keyframes(t: np.ndarray, keys: list[tuple[float, float]]) -> np.ndarray:
    """Piecewise smoothstep interpolation through (time, value) keys."""
    out = np.full_like(t, keys[0][1])
    for (t0, v0), (t1, v1) in zip(keys[:-1], keys[1:]):
        m = (t >= t0) & (t <= t1)
        out[m] = v0 + (v1 - v0) * _smoothstep((t[m] - t0) / (t1 - t0))
        out[t > t1] = v1
    return out


def _base_qpos(T: int) -> np.ndarray:
    kin = kinematics()
    q = np.zeros((T, 7 + len(kin.urdf_names)))
    q[:, 0] = 1.0
    q[:, 6] = 0.5745
    for name, v in DEFAULT_POSE.items():
        q[:, 7 + kin.urdf_names.index(name)] = v
    return q


def stand_kick_clip(
    side: str = "right", fps: float = 30.0, ball_radius: float = 0.1, ball_speed: float = 3.0
) -> LupusClip:
    kin = kinematics()
    t = np.arange(int(3.0 * fps)) / fps
    q = _base_qpos(len(t))
    j = lambda n: 7 + kin.urdf_names.index(f"{side}_{n}")  # noqa: E731
    hip = _keyframes(t, [(1.0, -0.2), (1.35, 0.2), (1.6, -0.9), (2.1, -0.2)])
    knee = _keyframes(t, [(1.0, 0.4), (1.35, 1.1), (1.6, 0.5), (2.1, 0.4)])
    # Sole pitch (toe down) = hip + knee + ankle; 0 = flat.
    sole = _keyframes(t, [(1.0, 0.0), (1.35, 0.3), (1.6, 0.0), (2.1, 0.0)])
    q[:, j("hip_pitch")] = hip
    q[:, j("knee")] = knee
    q[:, j("ankle_pitch")] = np.clip(sole - hip - knee, kin.lower[j("ankle_pitch") - 7], kin.upper[j("ankle_pitch") - 7])
    # Keep the stance sole exactly on the ground.
    stance = "left" if side == "right" else "right"
    q[:, 6] -= kin.clip_fk(q)[f"{stance}_corners"][:, :, 2].min(axis=1)
    clip = LupusClip(q, fps, name=f"synthetic_stand_kick_{side}")

    # Ball touching the front of the kick foot at its fastest frame, then rolling.
    from reference_data.annotate import synthetic_ball

    fk = kin.clip_fk(clip.qpos)
    speed = np.linalg.norm(np.gradient(fk[f"{side}_center"], 1.0 / fps, axis=0), axis=1)
    contact = int(np.argmax(speed))
    ball0 = synthetic_ball(clip, side, contact, ball_radius)
    ball = np.tile(ball0, (len(t), 1))
    after = t > t[contact]
    ball[after, 0] += ball_speed * (t[after] - t[contact])
    clip.ball_pos = ball
    clip.ball_radius = ball_radius
    clip.meta["true_contact"] = contact
    return clip


def walk_clip(steps: int = 6, fps: float = 30.0, step_time: float = 0.5) -> LupusClip:
    """Alternating leg lifts in place (feet leave and touch the ground)."""
    kin = kinematics()
    T = int((steps * step_time + 1.0) * fps)
    t = np.arange(T) / fps
    q = _base_qpos(T)
    for k in range(steps):
        side = "left" if k % 2 == 0 else "right"
        t0 = 0.5 + k * step_time
        keys_h = [(t0, -0.2), (t0 + step_time / 2, -0.6), (t0 + step_time * 0.9, -0.2)]
        keys_k = [(t0, 0.4), (t0 + step_time / 2, 1.0), (t0 + step_time * 0.9, 0.4)]
        m = (t >= t0) & (t <= t0 + step_time)
        q[m, 7 + kin.urdf_names.index(f"{side}_hip_pitch")] = _keyframes(t[m], keys_h)
        q[m, 7 + kin.urdf_names.index(f"{side}_knee")] = _keyframes(t[m], keys_k)
        q[m, 7 + kin.urdf_names.index(f"{side}_ankle_pitch")] = _keyframes(
            t[m], [(t0, -0.2), (t0 + step_time / 2, -0.4), (t0 + step_time * 0.9, -0.2)]
        )
    return LupusClip(q, fps, name="synthetic_walk").place_on_ground()
