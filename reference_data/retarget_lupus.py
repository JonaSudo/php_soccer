"""Retarget a human clip onto LUPUS with holosoma's interaction-mesh retargeter.

    # in the holosoma_retargeting environment (numpy 2.3, cvxpy, torch)
    python -m reference_data.retarget_lupus clip.human.npz --out clip.lupus_raw.npz

Clips with a ball (OmniContact kicks) are retargeted with the ball as a free
object, so the kick foot keeps its spatial relation to the ball (the
OmniRetarget principle the plan asks for). Clips without a ball use the plain
ground task.

holosoma's own entry point (``examples/robot_retarget.py``) is hard-wired to
its demo formats (InterMimic ``.pt``, a height table, G1/T1 constants), so this
module drives ``InteractionMeshRetargeter`` directly with the LUPUS model from
:mod:`reference_data.retarget_model`. The submodule stays untouched.

Output (``*.lupus_raw.npz``)::

    qpos_mj      (T, 7 + 22)  MuJoCo order [root_pos, root_quat_wxyz, joints]
    joint_names  (22,)        MJCF joint order of qpos_mj[:, 7:]
    fps          ()           frame rate of the retargeted clip
    ball_pos     (T, 3)       robot-scale ball centre (only clips with a ball)
    ball_radius  ()           ball used during retargeting
    human_joints (T, J, 3)    scaled human joints the robot was fitted to
    scale        ()           robot height / human height
    meta         ()           JSON: source, mapping, settings
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from reference_data.human_clip import HumanClip
from reference_data.retarget_model import build, load_config

REPO = Path(__file__).resolve().parents[1]
HOLOSOMA_RT = REPO / "thirdparty/holosoma/src/holosoma_retargeting"


def _import_holosoma():
    """Import the retargeter; it imports its helpers as top-level modules."""
    for p in (HOLOSOMA_RT, HOLOSOMA_RT / "holosoma_retargeting/src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter
    from holosoma_retargeting.src import utils as rt_utils

    return InteractionMeshRetargeter, rt_utils


def sphere_points(radius: float, n: int = 100) -> np.ndarray:
    """Evenly spread points on a sphere (Fibonacci lattice)."""
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    theta = np.pi * (1 + 5**0.5) * k
    return radius * np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def task_constants(cfg: dict, clip: HumanClip, model, joint_names: list[str], with_ball: bool) -> SimpleNamespace:
    mapping_all = cfg["joint_mappings"][clip.fmt]
    mapping = {h: r for h, r in mapping_all.items() if h in clip.joint_names}
    missing = sorted(set(mapping_all) - set(mapping))
    if missing:
        raise ValueError(f"clip lacks mapped joints {missing}")
    lb = {str(i): -1.0 for i in range(3, 7)}
    ub = {str(i): 1.0 for i in range(3, 7)}
    for name in cfg["robot"]["locked_joints"]:
        idx = str(7 + joint_names.index(name))
        lb[idx] = ub[idx] = 0.0
    return SimpleNamespace(
        ROBOT_URDF_FILE=str(model.urdf),
        ROBOT_DOF=len(joint_names),
        ROBOT_HEIGHT=float(cfg["robot"]["height"]),
        ROBOT_NAME="lupus_22dof",
        OBJECT_NAME="ball" if with_ball else "ground",
        FOOT_STICKING_LINKS=list(model.foot_sticking_links),
        DEMO_JOINTS=list(clip.joint_names),
        JOINTS_MAPPING=mapping,
        TOE_NAMES=list(cfg["toe_joints"][clip.fmt]),
        MANUAL_LB=lb,
        MANUAL_UB=ub,
        MANUAL_COST={},
        # root (7) + 12 leg joints, as for G1/T1
        NOMINAL_TRACKING_INDICES=np.arange(19),
    )


def retarget(
    clip: HumanClip,
    out_path: Path | str,
    *,
    cfg: dict | None = None,
    work_dir: Path | str | None = None,
    use_ball: bool | None = None,
    visualize: bool = False,
    max_frames: int | None = None,
) -> Path:
    cfg = cfg or load_config()
    rt_cfg = cfg["retargeting"]
    InteractionMeshRetargeter, rt_utils = _import_holosoma()

    clip = clip.resample(float(rt_cfg["fps"]))
    if max_frames is not None:
        clip = clip.trim(0, max_frames)
    with_ball = clip.has_ball if use_ball is None else (use_ball and clip.has_ball)
    r_robot = float(cfg["ball"]["robot_diameter"]) / 2
    r_human = float(clip.ball_diameter or cfg["ball"]["human_diameter"]) / 2

    work = Path(work_dir or tempfile.mkdtemp(prefix="lupus_rt_"))
    model = build(work, cfg, r_robot if with_ball else None)
    joint_names = list(model.joint_names)
    consts = task_constants(cfg, clip, model, joint_names, with_ball)

    retargeter = InteractionMeshRetargeter(
        task_constants=consts,
        object_urdf_path=None,
        q_a_init_idx=-7,
        activate_joint_limits=True,
        activate_obj_non_penetration=True,
        activate_foot_sticking=True,
        penetration_tolerance=1e-3,
        foot_sticking_tolerance=1e-3,
        step_size=float(rt_cfg["step_size"]),
        visualize=visualize,
        debug=False,
        w_nominal_tracking_init=5.0,
        nominal_tracking_tau=1e6,
    )

    scale = consts.ROBOT_HEIGHT / clip.human_height
    toe = consts.TOE_NAMES
    joints = clip.joints.copy()
    if with_ball:
        obj = np.concatenate([np.tile([1.0, 0.0, 0.0, 0.0], (clip.num_frames, 1)), clip.object_pos], axis=1)
        joints, obj, _ = rt_utils.preprocess_motion_data(joints, retargeter, toe, scale=scale, object_poses=obj)
        obj_mj = obj[:, [4, 5, 6, 0, 1, 2, 3]]
        # Robot-side ball: same path, resting on the ground with its own radius.
        obj_robot = obj_mj.copy()
        obj_robot[:, 2] += r_robot - obj_mj[0, 2]
        pts_demo, pts_robot = sphere_points(r_human * scale), sphere_points(r_robot)
    else:
        joints = rt_utils.preprocess_motion_data(joints, retargeter, toe, scale=scale)
        obj_mj = np.tile([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], (clip.num_frames, 1))
        g = np.linspace(*rt_cfg["ground_range"], int(rt_cfg["ground_size"]))
        gx, gy = np.meshgrid(g, g)
        pts_demo = pts_robot = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)

    pelvis = joints[0, clip.joint_names.index({"lafan": "Hips"}.get(clip.fmt, "Pelvis"))]
    heading = HumanClip(joints, clip.joint_names, clip.fps, clip.fmt, clip.human_height).heading_quat(0)
    q_init = np.concatenate([pelvis, heading, np.zeros(len(joint_names))])
    if not with_ball:
        # The retargeter writes object_poses_augmented into the last 7 qpos
        # entries as the initial guess; for ground-only clips those are robot
        # joints, so hand it the initial joint values instead.
        obj_robot = np.tile(q_init[-7:], (clip.num_frames, 1))

    sticking = rt_utils.extract_foot_sticking_sequence_velocity(
        joints, clip.joint_names, toe, velocity_threshold=float(rt_cfg["foot_sticking_velocity"])
    )
    # Warm-up: the solver starts from the zero pose and moves at most
    # step_size per iteration, so the first frames are not converged yet
    # (jumps of 0.3 rad between frames). Hold the first frame for a while
    # and drop those frames afterwards.
    w = int(rt_cfg.get("warmup_frames", 0))
    pad = lambda a: np.concatenate([np.repeat(a[:1], w, axis=0), a], axis=0)  # noqa: E731
    raw_path = work / "holosoma_out.npz"
    retargeter.retarget_motion(
        human_joint_motions=pad(joints),
        object_poses=pad(obj_mj),
        object_poses_augmented=pad(obj_robot),
        object_points_local_demo=pts_demo,
        object_points_local=pts_robot,
        foot_sticking_sequences=[sticking[0]] * w + list(sticking),
        q_a_init=q_init,
        q_nominal_list=None,
        original=True,
        dest_res_path=str(raw_path),
    )
    with np.load(raw_path) as d:
        qpos = np.asarray(d["qpos"], dtype=np.float64)[w:]
    n_robot = 7 + len(joint_names)
    out = dict(
        qpos_mj=qpos[:, :n_robot],
        joint_names=np.array(joint_names),
        fps=float(clip.fps),
        human_joints=joints,
        scale=scale,
        ball_radius=r_robot if with_ball else 0.0,
        meta=json.dumps(
            {
                "source": clip.source,
                "fmt": clip.fmt,
                "human_height": clip.human_height,
                "with_ball": bool(with_ball),
                "mapping": consts.JOINTS_MAPPING,
                "clip_meta": clip.meta,
            }
        ),
    )
    if with_ball:
        out["ball_pos"] = qpos[:, n_robot : n_robot + 3]
    out_path = Path(out_path)
    np.savez(out_path, **out)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path, help="*.human.npz from reference_data.human_clip")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-ball", action="store_true", help="ignore the ball even if the clip has one")
    ap.add_argument("--visualize", action="store_true", help="viser view of the result")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--work-dir", type=Path)
    args = ap.parse_args()
    clip = HumanClip.load(args.clip)
    path = retarget(
        clip, args.out, use_ball=not args.no_ball, visualize=args.visualize,
        max_frames=args.max_frames, work_dir=args.work_dir,
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
