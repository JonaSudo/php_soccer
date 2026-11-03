"""Kick annotation (Phase 1, steps 1 and 6).

For every kick clip this finds

- the **kick foot** and the **contact frame** (ball starts to move next to the
  foot; without a ball trajectory: peak speed of the kick foot),
- **skill start** (last lift-off of the kick foot before contact),
- **skill end** (kick foot back on the ground and settled),
- the start of the **entry window**: two run-up steps before skill start for
  kicks from the run, the start of the quiet stance for kicks from standing,
- the **ball position relative to the kick foot** at contact,

and sorts the clip into ``walk_kick`` (at least two run-up steps),
``stand_kick`` (quiet pose before the kick) or ``reject`` (e.g. a single
lunge step, which fits neither group). Every field can be overridden by
hand (see ``catalog.yaml``); the detector only proposes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from reference_data.clip_ops import LupusClip
from reference_data.lupus_kin import SIDES, box_point_distance, kinematics


@dataclass
class KickParams:
    ball_move_speed: float = 0.5      # m/s, ball counts as kicked above this
    ball_move_frames: int = 3         # ... for this many frames in a row
    contact_search_before: float = 0.2  # s before ball movement to look for the touch
    contact_tolerance: float = 0.01   # m, foot-ball gap that counts as touching
    stance_height: float = 0.02       # m, sole lower than this = may be in stance
    stance_speed: float = 0.25        # m/s, horizontal foot speed below = stance
    min_stance_frames: int = 3        # shorter ground phases are ignored
    quiet_speed: float = 0.10         # m/s pelvis speed of a quiet stance
    quiet_min_duration: float = 0.5   # s
    quiet_gap: float = 0.3            # s, quiet window must end this close to skill start
    runup_steps: int = 2              # touchdowns required for walk_kick
    runup_horizon: float = 2.5        # s before skill start that count as run-up
    settle: float = 0.2               # s after the kick foot lands = skill end


@dataclass
class KickAnnotation:
    group: str                       # walk_kick | stand_kick | reject
    kick_foot: str                   # left | right
    entry_start: int
    skill_start: int
    contact_frame: int
    skill_end: int
    fps: float
    num_frames: int
    ball_radius: float
    ball_source: str                 # data | synthetic
    ball_pos_contact: list[float]    # world, at contact
    ball_in_foot: list[float]        # ball centre in the foot-box frame (fwd, left, up)
    ball_in_heading: list[float]     # ball centre relative to the sole centre, yaw-aligned
    contact_distance: float          # foot box to ball surface at contact (m, <0 = overlap)
    runup_touchdowns: int
    reason: str = ""
    notes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def search_range(self) -> tuple[int, int, int]:
        """(entry window start, skill start, skill end) as skill_registry uses it."""
        return self.entry_start, self.skill_start, self.skill_end


def _runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """[start, stop) of True runs with at least ``min_len`` frames."""
    out, start = [], None
    for i, v in enumerate(np.append(mask, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    return out


def foot_tracks(clip: LupusClip, p: KickParams) -> dict:
    fk = kinematics().clip_fk(clip.qpos)
    dt = 1.0 / clip.fps
    out = {"fk": fk}
    for s in SIDES:
        c = fk[f"{s}_center"]
        v = np.gradient(c, dt, axis=0)
        low = fk[f"{s}_corners"][:, :, 2].min(axis=1) < p.stance_height
        stance = low & (np.linalg.norm(v[:, :2], axis=1) < p.stance_speed)
        runs = _runs(stance, p.min_stance_frames)
        clean = np.zeros_like(stance)
        for a, b in runs:
            clean[a:b] = True
        out[s] = {"center": c, "vel": v, "speed": np.linalg.norm(v, axis=1), "stance": clean, "runs": runs}
    pv = np.gradient(fk["pelvis"], dt, axis=0)
    out["pelvis_speed"] = np.linalg.norm(pv[:, :2], axis=1)
    return out


def _heading(rot: np.ndarray) -> np.ndarray:
    fwd = rot[:, 0].copy()
    fwd[2] = 0.0
    n = np.linalg.norm(fwd)
    return fwd / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])


def _ball_moving_frame(ball: np.ndarray, fps: float, p: KickParams) -> int | None:
    speed = np.linalg.norm(np.gradient(ball, 1.0 / fps, axis=0), axis=1)
    runs = _runs(speed > p.ball_move_speed, p.ball_move_frames)
    return runs[0][0] if runs else None


def synthetic_ball(clip: LupusClip, side: str, frame: int, radius: float) -> np.ndarray:
    """Ball resting on the ground in front of the foot, just touching the foot
    box at ``frame`` (found by bisection along the foot heading)."""
    kin = kinematics()
    kin.forward(clip.qpos[frame])
    c, rot = kin.foot_frame(side)
    half = kin.feet[side].half
    fwd = _heading(rot)
    start = np.array([c[0], c[1], radius])

    def gap(s: float) -> float:
        return box_point_distance(c, rot, half, start + s * fwd) - radius

    lo, hi = -half[0], half[0] + 3 * radius
    if gap(hi) <= 0:
        return start + hi * fwd
    if gap(lo) > 0:  # foot entirely above/behind: put the ball under the toe
        lo = 0.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if gap(mid) < 0 else (lo, mid)
    return start + hi * fwd


def annotate(
    clip: LupusClip,
    params: KickParams | None = None,
    overrides: dict | None = None,
    ball_radius: float | None = None,
) -> KickAnnotation:
    """Detect the kick events. ``overrides`` replaces any detected field."""
    p = params or KickParams()
    ov = dict(overrides or {})
    tr = foot_tracks(clip, p)
    fk = tr["fk"]
    kin = kinematics()
    T, fps = clip.num_frames, clip.fps
    radius = float(ball_radius or clip.ball_radius or 0.1)
    notes: dict = {}

    # ---- kick foot and contact ------------------------------------------------
    kick_foot, contact = ov.get("kick_foot"), ov.get("contact_frame")
    have_ball = clip.ball_pos is not None
    if have_ball and (kick_foot is None or contact is None):
        move = _ball_moving_frame(clip.ball_pos, fps, p)
        notes["ball_move_frame"] = move
        if move is not None:
            lo = max(0, move - int(round(p.contact_search_before * fps)))
            hi = min(T, move + 2)
            # First touch (gap below contact_tolerance) in the window; if the
            # foot never gets that close, the closest approach.
            best, first = (np.inf, None, None), None
            for t in range(lo, hi):
                for s in SIDES:
                    if kick_foot is not None and s != kick_foot:
                        continue
                    dist = box_point_distance(
                        fk[f"{s}_center"][t], fk[f"{s}_rot"][t], kin.feet[s].half, clip.ball_pos[t]
                    ) - radius
                    if dist < best[0]:
                        best = (dist, s, t)
                    if first is None and dist <= p.contact_tolerance:
                        first = (dist, s, t)
            pick = first or best
            kick_foot = kick_foot or pick[1]
            contact = contact if contact is not None else pick[2]
    if kick_foot is None:
        kick_foot = max(SIDES, key=lambda s: tr[s]["speed"].max())
    if contact is None:
        contact = int(np.argmax(tr[kick_foot]["speed"]))
        notes["contact_from"] = "peak_foot_speed"
    contact = int(contact)

    # ---- skill start / end ------------------------------------------------------
    k = tr[kick_foot]
    before = [r for r in k["runs"] if r[0] < contact]
    skill_start = ov.get("skill_start", before[-1][1] if before else 0)
    after = [r for r in k["runs"] if r[0] > contact]
    if "skill_end" in ov:
        skill_end = ov["skill_end"]
    elif after:
        skill_end = min(T - 1, after[0][0] + int(round(p.settle * fps)))
    else:
        skill_end = T - 1
        notes["skill_end"] = "kick foot never lands again"
    skill_start, skill_end = int(skill_start), int(skill_end)

    # ---- entry window and group -------------------------------------------------
    horizon = max(0, skill_start - int(round(p.runup_horizon * fps)))
    touchdowns = sorted(
        a for s in SIDES for a, _ in tr[s]["runs"] if horizon < a < skill_start
    )
    quiet = (tr["pelvis_speed"] < p.quiet_speed) & tr["left"]["stance"] & tr["right"]["stance"]
    quiet_runs = [r for r in _runs(quiet[: skill_start + 1], 1) if r[1] - r[0] >= p.quiet_min_duration * fps]
    quiet_ok = bool(quiet_runs) and skill_start - quiet_runs[-1][1] <= p.quiet_gap * fps
    reason = ""
    if "group" in ov:
        group = ov["group"]
    elif quiet_ok and len([a for a in touchdowns if a >= quiet_runs[-1][1]]) == 0:
        group = "stand_kick"
    elif len(touchdowns) >= p.runup_steps:
        group = "walk_kick"
    else:
        group = "reject"
        reason = f"{len(touchdowns)} run-up touchdown(s) and no quiet stance before the kick"
    if "entry_start" in ov:
        entry = int(ov["entry_start"])
    elif group == "stand_kick":
        entry = quiet_runs[-1][0]
    elif len(touchdowns) >= p.runup_steps:
        entry = touchdowns[-p.runup_steps]
    else:
        entry = horizon
    entry = int(entry)

    # ---- ball relative to kick foot --------------------------------------------
    if have_ball and not ov.get("synthetic_ball", False):
        ball = clip.ball_pos[contact].copy()
        source = "data"
    else:
        ball = synthetic_ball(clip, kick_foot, contact, radius)
        source = "synthetic"
    c, rot = fk[f"{kick_foot}_center"][contact], fk[f"{kick_foot}_rot"][contact]
    in_foot = rot.T @ (ball - c)
    fwd = _heading(rot)
    left = np.cross([0.0, 0.0, 1.0], fwd)
    sole = c.copy()
    sole[2] = 0.0
    rel = ball - sole
    in_heading = np.array([rel @ fwd, rel @ left, rel[2]])
    dist = box_point_distance(c, rot, kin.feet[kick_foot].half, ball) - radius

    return KickAnnotation(
        group=group,
        kick_foot=kick_foot,
        entry_start=entry,
        skill_start=skill_start,
        contact_frame=contact,
        skill_end=skill_end,
        fps=fps,
        num_frames=T,
        ball_radius=radius,
        ball_source=source,
        ball_pos_contact=[float(v) for v in ball],
        ball_in_foot=[float(v) for v in in_foot],
        ball_in_heading=[float(v) for v in in_heading],
        contact_distance=float(dist),
        runup_touchdowns=len(touchdowns),
        reason=reason,
        notes=notes,
    )


def ball_track(clip: LupusClip, ann: KickAnnotation) -> np.ndarray:
    """Ball centre per frame for export.

    Data ball: the retargeted trajectory. Synthetic ball: resting at the
    annotated contact position (valid up to the contact frame; afterwards the
    ball follows physics in training, not a reference).
    """
    if ann.ball_source == "data" and clip.ball_pos is not None:
        return clip.ball_pos.copy()
    return np.tile(ann.ball_pos_contact, (clip.num_frames, 1))


def scale_frames(fields: dict, factor: float) -> dict:
    """Map frame-valued override fields to a clip played at ``factor`` x speed
    (same fps). Used for hand annotations given on the unscaled clip."""
    out = dict(fields)
    for key in ("entry_start", "skill_start", "contact_frame", "skill_end"):
        if key in out and out[key] is not None:
            out[key] = int(round(out[key] / factor))
    return out


def mirror_fields(fields: dict) -> dict:
    out = dict(fields)
    if out.get("kick_foot") in SIDES:
        out["kick_foot"] = "right" if out["kick_foot"] == "left" else "left"
    return out
