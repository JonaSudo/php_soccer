"""Ball-detection model: what LUPUS's vision stack (YOLO + depth net) reports.

The student policy never sees the true ball. It sees what the robot's vision
node publishes: a ball position on the ground plane in the robot frame, a
``visible`` flag and the time since the last sighting. This module reproduces
that signal from the simulator's ground truth so the student is trained on the
same kind of input it gets on the field (plan, Phase 4).

What is modelled, per env:

* **Field of view.** The camera sits in the head (``head_link``), which the
  head-scan generator moves; the ball is only detected inside the camera's
  horizontal/vertical field of view and range. Close to the kick the ball is
  often below the image or the head looks elsewhere, so the policy must time the
  kick from its history.
* **Noise.** Direction from the image position is accurate
  (``bearing_sigma_deg``), depth from the depth net is not: ``sigma_depth = k *
  distance`` plus a per-episode relative bias. Optionally the ball is placed
  by intersecting the noisy viewing ray with the ground plane when it is close
  (``near_ground_projection_range``), which is what the vision stack can do with
  the known camera height and head angle.
* **Timing.** New detections arrive at ``update_hz_range`` (randomised per
  episode, 25-40 Hz against the 50 Hz policy); between detections the last value
  is held. Each detection describes the scene ``latency_s_range`` ago. A fraction
  ``dropout_range`` of detections is lost.

All numbers live in :class:`BallVisionCfg` and are starting values: the camera
mount and field of view follow the Booster K1 head, the noise numbers are
estimates. Measure ``depth_sigma_k`` and ``bearing_sigma_deg`` against ground
truth (tape measure or mocap) and the latency on the robot, then change them
here (see ``wbt_training/BALL_DISTILL.md``).

The functions are pure torch and run on CPU; the holosoma observation term that
feeds them from the simulator is
:class:`wbt_training.config_values.ball_distillation.BallDetectionObsTerm`.
Quaternions are xyzw, as in holosoma's actor-state API.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any

import torch

#: Output layout of :meth:`BallDetector.step` (and the ``ball_detection`` term).
BALL_DETECTION_FIELDS = ("x", "y", "visible", "time_since_seen")


@dataclass(frozen=True)
class BallVisionCfg:
    """Camera, noise and timing of the ball detection. Units: m, s, rad unless noted."""

    # --- camera mount (K1 reference until the LUPUS head exists) -------------
    # Camera optical centre in the head_link frame (x forward, y left, z up at
    # zero head angles, same as the base frame; see robots/lupus/hardware.yaml).
    camera_offset_head: tuple[float, float, float] = (0.06, 0.0, 0.08)
    # Fixed downward tilt of the camera inside the head (positive = down).
    camera_pitch_offset: float = 0.0
    hfov_deg: float = 90.0
    vfov_deg: float = 65.0
    min_range: float = 0.10
    max_range: float = 6.0

    # --- noise ---------------------------------------------------------------
    depth_sigma_k: float = 0.05  # sigma_depth = k * distance
    depth_bias_range: tuple[float, float] = (-0.03, 0.03)  # relative, per episode
    bearing_sigma_deg: float = 1.0  # azimuth and elevation
    # Below this distance the position comes from the viewing ray hitting the
    # ground plane (z = ball_radius) instead of the depth net; 0 disables it.
    near_ground_projection_range: float = 0.0
    ground_projection_pitch_sigma_deg: float = 0.5  # head/IMU angle error
    ball_radius: float = 0.07

    # --- timing --------------------------------------------------------------
    update_hz_range: tuple[float, float] = (25.0, 40.0)
    latency_s_range: tuple[float, float] = (0.04, 0.10)
    dropout_range: tuple[float, float] = (0.05, 0.20)
    time_since_seen_cap: float = 2.0
    policy_dt: float = 0.02  # used when the env does not expose its own dt

    # --- output frame --------------------------------------------------------
    # "heading": gravity-aligned frame of the base yaw (x forward, y left, on
    # the ground plane). "base": full base orientation. Which one the real
    # vision node publishes is an open question in the plan; heading is what
    # the kick needs.
    frame: str = "heading"

    @classmethod
    def from_params(cls, params: dict[str, Any] | None) -> "BallVisionCfg":
        """Build from an obs-term ``params`` dict, ignoring unrelated keys."""
        names = {f.name for f in fields(cls)}
        kwargs = {}
        for k, v in (params or {}).items():
            if k in names:
                kwargs[k] = tuple(v) if isinstance(v, list) else v
        return cls(**kwargs)

    def as_params(self) -> dict[str, Any]:
        """Plain dict for ``ObsTermCfg.params`` (YAML-serialisable)."""
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}

    @property
    def max_latency_steps(self) -> int:
        return int(math.ceil(self.latency_s_range[1] / self.policy_dt)) + 1


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def quat_to_matrix_xyzw(q: torch.Tensor) -> torch.Tensor:
    """Rotation matrices ``[..., 3, 3]`` from xyzw quaternions ``[..., 4]``."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    x, y, z, w = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def yaw_from_quat_xyzw(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _rot_y(angle: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(angle), torch.sin(angle)
    o, z = torch.ones_like(c), torch.zeros_like(c)
    return torch.stack([c, z, s, z, o, z, -s, z, c], dim=-1).reshape(*angle.shape, 3, 3)


def camera_pose_w(
    head_pos_w: torch.Tensor, head_quat_w: torch.Tensor, cfg: BallVisionCfg
) -> tuple[torch.Tensor, torch.Tensor]:
    """Camera position ``[N, 3]`` and rotation ``[N, 3, 3]`` in the world.

    The camera frame is x forward (optical axis), y left, z up, i.e. the head
    frame tilted down by ``camera_pitch_offset`` (a positive rotation about y
    turns +x towards -z).
    """
    r_head = quat_to_matrix_xyzw(head_quat_w)
    offset = torch.tensor(cfg.camera_offset_head, dtype=head_pos_w.dtype, device=head_pos_w.device)
    cam_pos = head_pos_w + r_head @ offset
    tilt = _rot_y(torch.full(head_pos_w.shape[:-1], cfg.camera_pitch_offset, dtype=head_pos_w.dtype, device=head_pos_w.device))
    return cam_pos, r_head @ tilt


def in_field_of_view(p_cam: torch.Tensor, cfg: BallVisionCfg) -> torch.Tensor:
    """True where a camera-frame point ``[N, 3]`` lies inside FOV and range."""
    fwd, left, up = p_cam.unbind(-1)
    dist = p_cam.norm(dim=-1)
    az = torch.atan2(left, fwd)
    el = torch.atan2(up, torch.sqrt(fwd * fwd + left * left))
    return (
        (fwd > 0)
        & (az.abs() <= math.radians(cfg.hfov_deg) / 2)
        & (el.abs() <= math.radians(cfg.vfov_deg) / 2)
        & (dist >= cfg.min_range)
        & (dist <= cfg.max_range)
    )


def _direction(az: torch.Tensor, el: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.cos(el) * torch.cos(az), torch.cos(el) * torch.sin(az), torch.sin(el)], dim=-1)


def noisy_camera_measurement(
    p_cam: torch.Tensor,
    cam_pos_w: torch.Tensor,
    cam_rot_w: torch.Tensor,
    depth_bias: torch.Tensor,
    cfg: BallVisionCfg,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Noisy ball position in the world ``[N, 3]`` from its camera-frame position.

    Depth error grows with distance (``sigma = k * d``, plus the per-episode
    relative ``depth_bias``); azimuth and elevation get a small Gaussian error.
    Within ``near_ground_projection_range`` the noisy viewing ray is intersected
    with the plane ``z = ball_radius`` instead (camera height known, head angle
    with ``ground_projection_pitch_sigma_deg`` error), mimicking the fusion the
    vision stack can do near the ball.
    """
    dist = p_cam.norm(dim=-1).clamp_min(1e-6)
    az = torch.atan2(p_cam[:, 1], p_cam[:, 0])
    el = torch.asin((p_cam[:, 2] / dist).clamp(-1.0, 1.0))

    def randn() -> torch.Tensor:
        return torch.randn(dist.shape, generator=generator, device=dist.device, dtype=dist.dtype)

    bearing = math.radians(cfg.bearing_sigma_deg)
    az_n = az + bearing * randn()
    el_n = el + bearing * randn()
    dist_n = (dist * (1.0 + depth_bias) + cfg.depth_sigma_k * dist * randn()).clamp_min(cfg.min_range * 0.5)
    meas_w = cam_pos_w + (cam_rot_w @ (_direction(az_n, el_n) * dist_n[:, None]).unsqueeze(-1)).squeeze(-1)

    if cfg.near_ground_projection_range > 0:
        el_g = el_n + math.radians(cfg.ground_projection_pitch_sigma_deg) * randn()
        ray_w = (cam_rot_w @ _direction(az_n, el_g).unsqueeze(-1)).squeeze(-1)
        drop = cam_pos_w[:, 2] - cfg.ball_radius
        # Only rays that point down hit the ground in front of the camera.
        t = drop / (-ray_w[:, 2]).clamp_min(1e-6)
        ground_w = cam_pos_w + ray_w * t[:, None]
        use = (dist < cfg.near_ground_projection_range) & (ray_w[:, 2] < -1e-3) & (drop > 0)
        meas_w = torch.where(use[:, None], ground_w, meas_w)
    return meas_w


def world_to_robot_xy(
    p_w: torch.Tensor, base_pos_w: torch.Tensor, base_quat_w: torch.Tensor, frame: str = "heading"
) -> torch.Tensor:
    """``(x, y)`` of a world point in the robot's heading or base frame ``[N, 2]``."""
    d = p_w - base_pos_w
    if frame == "heading":
        yaw = yaw_from_quat_xyzw(base_quat_w)
        c, s = torch.cos(yaw), torch.sin(yaw)
        return torch.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], dim=-1)
    if frame == "base":
        r = quat_to_matrix_xyzw(base_quat_w)
        return (r.transpose(-1, -2) @ d.unsqueeze(-1)).squeeze(-1)[:, :2]
    raise ValueError(f"unknown frame {frame!r}; use 'heading' or 'base'")


# ---------------------------------------------------------------------------
# Stateful detector (timing, latency, dropouts)
# ---------------------------------------------------------------------------


class BallDetector:
    """Per-env ball detection with camera rate, latency, dropouts and hold.

    Call :meth:`step` once per policy step with the simulator ground truth and
    :meth:`reset` for resetting envs. :meth:`step` returns ``[N, 4]`` in
    :data:`BALL_DETECTION_FIELDS` order: held ``(x, y)`` in the robot frame,
    ``visible`` (1 if the newest camera frame contained the ball) and the time
    since the newest sighting was captured, capped at ``time_since_seen_cap``.
    Before the first sighting of an episode ``x = y = 0``.
    """

    def __init__(
        self,
        num_envs: int,
        cfg: BallVisionCfg,
        device: str | torch.device = "cpu",
        dt: float | None = None,
        generator: torch.Generator | None = None,
    ):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dt = float(dt if dt is not None else cfg.policy_dt)
        self.generator = generator
        self.buffer_len = int(math.ceil(cfg.latency_s_range[1] / self.dt)) + 1
        n = num_envs
        # Ring buffer of what the camera captured (measurement xy, valid flag).
        self._cap_xy = torch.zeros(n, self.buffer_len, 2, device=self.device)
        self._cap_valid = torch.zeros(n, self.buffer_len, dtype=torch.bool, device=self.device)
        self._head = 0  # index of the newest slot
        # Per-episode randomisation.
        self.rate_hz = torch.zeros(n, device=self.device)
        self.latency_steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self.dropout_p = torch.zeros(n, device=self.device)
        self.depth_bias = torch.zeros(n, device=self.device)
        self._phase = torch.zeros(n, device=self.device)
        # Published state.
        self.held_xy = torch.zeros(n, 2, device=self.device)
        self.visible = torch.zeros(n, device=self.device)
        self.age = torch.full((n,), cfg.time_since_seen_cap, device=self.device)
        self._steps_since_capture = torch.zeros(n, dtype=torch.long, device=self.device)
        self._fresh = torch.ones(n, dtype=torch.bool, device=self.device)
        self.reset(None)

    # -- helpers ------------------------------------------------------------

    def _uniform(self, lo: float, hi: float, k: int) -> torch.Tensor:
        u = torch.rand(k, generator=self.generator, device=self.device)
        return lo + (hi - lo) * u

    def reset(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        k = env_ids.numel()
        if k == 0:
            return
        c = self.cfg
        self.rate_hz[env_ids] = self._uniform(*c.update_hz_range, k)
        lat = self._uniform(*c.latency_s_range, k)
        self.latency_steps[env_ids] = torch.round(lat / self.dt).long().clamp(0, self.buffer_len - 1)
        self.dropout_p[env_ids] = self._uniform(*c.dropout_range, k)
        self.depth_bias[env_ids] = self._uniform(*c.depth_bias_range, k)
        # Random camera phase so frames are not aligned with the policy clock.
        self._phase[env_ids] = torch.rand(k, generator=self.generator, device=self.device)
        self._cap_xy[env_ids] = 0.0
        self._cap_valid[env_ids] = False
        self.held_xy[env_ids] = 0.0
        self.visible[env_ids] = 0.0
        self.age[env_ids] = c.time_since_seen_cap
        self._steps_since_capture[env_ids] = 0
        self._fresh[env_ids] = True

    def output(self) -> torch.Tensor:
        return torch.cat([self.held_xy, self.visible[:, None], self.age[:, None]], dim=-1)

    # -- main step ----------------------------------------------------------

    def step(
        self,
        ball_pos_w: torch.Tensor,
        head_pos_w: torch.Tensor,
        head_quat_w: torch.Tensor,
        base_pos_w: torch.Tensor,
        base_quat_w: torch.Tensor,
    ) -> torch.Tensor:
        """Advance one policy step and return the published detection ``[N, 4]``."""
        c = self.cfg
        cam_pos, cam_rot = camera_pose_w(head_pos_w, head_quat_w, c)
        p_cam = (cam_rot.transpose(-1, -2) @ (ball_pos_w - cam_pos).unsqueeze(-1)).squeeze(-1)
        valid = in_field_of_view(p_cam, c)
        meas_w = noisy_camera_measurement(p_cam, cam_pos, cam_rot, self.depth_bias, c, self.generator)
        meas_xy = world_to_robot_xy(meas_w, base_pos_w, base_quat_w, c.frame)

        # Push the capture of this step into the ring buffer.
        self._head = (self._head + 1) % self.buffer_len
        self._cap_xy[:, self._head] = meas_xy
        self._cap_valid[:, self._head] = valid
        # A freshly reset env has no past: fill its whole buffer with now.
        if self._fresh.any():
            f = self._fresh
            self._cap_xy[f] = meas_xy[f].unsqueeze(1).expand(-1, self.buffer_len, -1)
            self._cap_valid[f] = valid[f].unsqueeze(1).expand(-1, self.buffer_len)
            self._fresh[:] = False

        # Camera frame events at the per-env rate.
        self._phase += self.rate_hz * self.dt
        event = self._phase >= 1.0
        self._phase = torch.where(event, self._phase - 1.0, self._phase)

        idx = (self._head - self.latency_steps) % self.buffer_len
        rows = torch.arange(self.num_envs, device=self.device)
        cap_xy = self._cap_xy[rows, idx]
        cap_valid = self._cap_valid[rows, idx]
        kept = torch.rand(self.num_envs, generator=self.generator, device=self.device) >= self.dropout_p
        seen = event & cap_valid & kept

        self._steps_since_capture += 1
        self.held_xy = torch.where(seen[:, None], cap_xy, self.held_xy)
        self._steps_since_capture = torch.where(seen, self.latency_steps, self._steps_since_capture)
        # visible follows the newest camera frame; between frames it holds.
        self.visible = torch.where(event, seen.float(), self.visible)
        never = self.age >= c.time_since_seen_cap
        age = (self._steps_since_capture.float() * self.dt).clamp(max=c.time_since_seen_cap)
        self.age = torch.where(seen | ~never, age, self.age)
        return self.output()


__all__ = [
    "BALL_DETECTION_FIELDS",
    "BallDetector",
    "BallVisionCfg",
    "camera_pose_w",
    "in_field_of_view",
    "noisy_camera_measurement",
    "world_to_robot_xy",
]
