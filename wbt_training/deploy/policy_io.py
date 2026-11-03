"""Observation builder, ball/command inputs and ONNX policy for the LUPUS ball student.

Everything here reproduces, with numpy only, what the training env computes for
``actor_obs`` (see ``wbt_training/BALL_DISTILL.md``):

* terms in the order and widths of the ``obs_layout`` ONNX metadata
  (alphabetical, each term's ``history`` frames back to back, oldest first,
  zero-padded after a reset);
* ``dof_pos`` relative to the default pose, ``dof_vel`` raw, ``base_ang_vel`` in
  the base frame, projected gravity of the base, ``actions`` = raw last policy
  output (neck entries included, they are ignored for control);
* ``ball_detection`` = ``[x, y, visible, time_since_seen]`` with the training
  semantics (:class:`BallInput`);
* ``command`` = ``[v, omega, kick_enable]`` (Phase 6: ``kick_dir``, mode one-hot).

``tests/test_deploy.py`` checks the builder against holosoma's
ObservationManager and :class:`BallInput` against the training detector.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- metadata


def parse_list(text: str) -> list:
    """Metadata list in either exporter format: JSON (holosoma) or CSV (PHP)."""
    text = text.strip()
    if text.startswith("["):
        return json.loads(text)
    out: list = []
    for item in filter(None, (t.strip() for t in text.split(","))):
        try:
            out.append(float(item))
        except ValueError:
            out.append(item)
    return out


def parse_layout(text: str) -> list[dict[str, Any]]:
    layout, start = [], 0
    for item in filter(None, text.split(";")):
        name, dim, hist = item.split(":")
        stop = start + int(dim) * int(hist)
        layout.append({"name": name, "dim": int(dim), "history": int(hist), "start": start, "stop": stop})
        start = stop
    return layout


@dataclass
class PolicyMetadata:
    joint_names: list[str]
    default_joint_pos: np.ndarray
    action_scale: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    layout: list[dict[str, Any]]
    passive_dof_names: list[str]
    policy_dt: float
    ball_vision_cfg: dict[str, Any]
    command_obs_cfg: dict[str, Any]

    @property
    def obs_dim(self) -> int:
        return self.layout[-1]["stop"] if self.layout else 0

    @classmethod
    def from_dict(cls, meta: Mapping[str, str]) -> "PolicyMetadata":
        if meta.get("policy_format") != "lupus_ball_student_v1":
            raise ValueError(f"not a LUPUS ball-student ONNX (policy_format={meta.get('policy_format')!r})")
        names = [str(n) for n in parse_list(meta["joint_names"])]
        n = len(names)

        def arr(key: str, default: float | None = None) -> np.ndarray:
            if key not in meta:
                if default is None:
                    raise KeyError(f"ONNX metadata misses {key!r}")
                return np.full(n, default)
            return np.broadcast_to(np.asarray(parse_list(meta[key]), float), (n,)).copy()

        return cls(
            joint_names=names,
            default_joint_pos=arr("default_joint_pos"),
            action_scale=arr("action_scale"),
            kp=arr("kp", 0.0),
            kd=arr("kd", 0.0),
            layout=parse_layout(meta["obs_layout"]),
            passive_dof_names=[str(x) for x in parse_list(meta.get("passive_dof_names", ""))],
            policy_dt=float(meta.get("policy_dt", 0.02)),
            ball_vision_cfg=json.loads(meta.get("ball_vision_cfg", "{}")),
            command_obs_cfg=json.loads(meta.get("command_obs_cfg", "{}")),
        )


# --------------------------------------------------------------------------- math


def quat_rotate_inverse_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, xyz = q[0], q[1:]
    return v * (2.0 * w * w - 1.0) - np.cross(xyz, v) * w * 2.0 + xyz * np.dot(xyz, v) * 2.0


def projected_gravity_wxyz(q: np.ndarray) -> np.ndarray:
    return quat_rotate_inverse_wxyz(np.asarray(q, float), np.array([0.0, 0.0, -1.0]))


# --------------------------------------------------------------------------- inputs


class BallInput:
    """Vision messages -> ``[x, y, visible, time_since_seen]`` like the training detector.

    Call :meth:`on_frame` for every camera frame the vision node processed
    (``found`` False if the ball was not detected), with the frame's *capture*
    time, and :meth:`observe` once per policy step. ``x, y`` must be in the
    frame the policy was trained with (default: heading frame of the base,
    see ``ball_vision_cfg.frame`` in the metadata).
    """

    def __init__(self, cap: float = 2.0):
        self.cap = cap
        self.reset()

    def reset(self) -> None:
        self.xy = np.zeros(2)
        self.visible = 0.0
        self.last_capture: float | None = None

    def on_frame(self, capture_time: float, found: bool, x: float = 0.0, y: float = 0.0) -> None:
        self.visible = 1.0 if found else 0.0
        if found:
            self.xy = np.array([x, y], float)
            self.last_capture = capture_time

    def observe(self, now: float) -> np.ndarray:
        age = self.cap if self.last_capture is None else min(max(now - self.last_capture, 0.0), self.cap)
        return np.array([self.xy[0], self.xy[1], self.visible, age], dtype=np.float32)


def command_vector(
    v: float,
    omega: float,
    kick_enable: float = 0.0,
    *,
    kick_dir: float | None = None,
    mode: int | None = None,
    cfg: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """``command`` term for the layout in ``command_obs_cfg`` (Phase 4 or 6)."""
    cfg = cfg or {}
    parts = [float(v), float(omega)]
    if cfg.get("include_kick_dir", False):
        parts.append(float(kick_dir or 0.0))
    if cfg.get("mode", "enable") == "mode3":
        m = int(mode if mode is not None else (1 if kick_enable else 0))
        parts += [1.0 if m == k else 0.0 for k in range(3)]
    else:
        parts.append(1.0 if kick_enable else 0.0)
    return np.asarray(parts, dtype=np.float32)


# --------------------------------------------------------------------------- observation


class ObsBuilder:
    """Concatenate terms with per-term history exactly like holosoma's ObservationManager."""

    def __init__(self, layout: Sequence[dict[str, Any]]):
        self.layout = list(layout)
        self.reset()

    @property
    def dim(self) -> int:
        return self.layout[-1]["stop"] if self.layout else 0

    def reset(self) -> None:
        self.buffers = {e["name"]: deque(maxlen=e["history"]) for e in self.layout}

    def build(self, terms: Mapping[str, np.ndarray]) -> np.ndarray:
        out = []
        for e in self.layout:
            x = np.asarray(terms[e["name"]], dtype=np.float32).reshape(-1)
            if x.shape[0] != e["dim"]:
                raise ValueError(f"term {e['name']!r} has {x.shape[0]} values, policy expects {e['dim']}")
            buf = self.buffers[e["name"]]
            buf.append(x)
            pad = [np.zeros(e["dim"], np.float32)] * (e["history"] - len(buf))
            out.extend(pad + list(buf))
        return np.concatenate(out).astype(np.float32)[None]


@dataclass
class RobotState:
    """What the firmware reports, joints in the policy's joint order."""

    joint_pos: np.ndarray
    joint_vel: np.ndarray
    base_quat_wxyz: np.ndarray  # IMU orientation of the base
    base_ang_vel: np.ndarray  # rad/s in the base frame


class StudentPolicy:
    """ONNX student + observation assembly + PD targets for the leg/arm joints."""

    def __init__(self, onnx_path: str | Path, providers: Sequence[str] | None = None):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(onnx_path), providers=list(providers or ["CPUExecutionProvider"]))
        meta = self.session.get_modelmeta().custom_metadata_map
        self.meta = PolicyMetadata.from_dict(meta)
        self.obs = ObsBuilder(self.meta.layout)
        if self.obs.dim != self.session.get_inputs()[0].shape[1]:
            raise ValueError("obs_layout does not match the ONNX input width")
        self.input_name = self.session.get_inputs()[0].name
        n = len(self.meta.joint_names)
        self.passive = np.array([self.meta.joint_names.index(j) for j in self.meta.passive_dof_names], dtype=int)
        self.active = np.setdiff1d(np.arange(n), self.passive)
        self.reset()

    def reset(self) -> None:
        self.last_action = np.zeros(len(self.meta.joint_names), np.float32)
        self.obs.reset()

    def terms(self, state: RobotState, command: np.ndarray, ball: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "actions": self.last_action,
            "ball_detection": ball,
            "base_ang_vel": state.base_ang_vel,
            "command": command,
            "dof_pos": np.asarray(state.joint_pos) - self.meta.default_joint_pos,
            "dof_vel": state.joint_vel,
            "robot_anchor_projected_gravity": projected_gravity_wxyz(state.base_quat_wxyz),
        }

    def step(self, state: RobotState, command: np.ndarray, ball: np.ndarray) -> np.ndarray:
        """Return joint position targets (all joints; neck entries = default, overwritten by the behavior)."""
        obs = self.obs.build(self.terms(state, command, ball))
        action = self.session.run(None, {self.input_name: obs})[0][0].astype(np.float32)
        self.last_action = action
        a = action.copy()
        a[self.passive] = 0.0
        return self.meta.default_joint_pos + a * self.meta.action_scale


__all__ = [
    "BallInput",
    "ObsBuilder",
    "PolicyMetadata",
    "RobotState",
    "StudentPolicy",
    "command_vector",
    "parse_layout",
    "parse_list",
    "projected_gravity_wxyz",
]
