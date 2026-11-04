#!/usr/bin/env python3
"""Print (or write) launcher commands for a dribble reward sweep (Phase 6 tuning).

The dribble rewards (``ball_near_foot``, ``ball_along_heading``) start from
estimated weights (:class:`wbt_training.config_values.lupus_phase6.DribbleRewardCfg`).
This script builds one ``run_lupus_kick_teacher.sh`` call per combination, with
the weights as CLI overrides, so a sweep needs no code change::

    python -m wbt_training.training_runs.dribble_sweep \\
        --registry file:///data/lupus_phase6_bundle \\
        --near-foot 0.25 0.5 1.0 --along-heading 0.5 1.0 --iterations 5000 > sweep.sh
    bash sweep.sh   # runs one after another; or split the lines across machines

Compare the runs with ``eval_teacher.py --kick-metrics.enabled True``
(``dribble_ball_dist_mean``, ``dribble_loss_rate``, ``dribble_speed_err_rms``).
"""

from __future__ import annotations

import argparse
import itertools
import json
import shlex
import sys
from collections.abc import Sequence

from wbt_training.config_values.lupus_phase6 import DRIBBLE_REWARDS


def sweep_commands(
    registry: str,
    near_foot: Sequence[float],
    along_heading: Sequence[float],
    near_foot_sigma: Sequence[float] = (DRIBBLE_REWARDS.near_foot_sigma,),
    preset: str = "exp:lupus-dribble-teacher",
    iterations: int | None = None,
    ngpus: int = 2,
    extra: Sequence[str] = (),
) -> list[str]:
    cmds = []
    for wn, wa, sn in itertools.product(near_foot, along_heading, near_foot_sigma):
        name = f"dribble_nf{wn:g}_ah{wa:g}_s{sn:g}"
        args = [
            "--reward.terms.ball-near-foot.weight", f"{wn:g}",
            "--reward.terms.ball-near-foot.params", json.dumps({"sigma": sn}),
            "--reward.terms.ball-along-heading.weight", f"{wa:g}",
        ]
        if iterations:
            args += ["--algo.config.num-learning-iterations", str(iterations)]
        args += list(extra)
        env = f"EXP_PRESET={preset} REGISTRY={shlex.quote(registry)} NGPUS={ngpus} RUN_NAME={name}"
        cmds.append(f"{env} bash wbt_training/training_runs/run_lupus_kick_teacher.sh " + " ".join(shlex.quote(a) for a in args))
    return cmds


def main(argv: Sequence[str] | None = None) -> int:
    d = DRIBBLE_REWARDS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--registry", required=True)
    ap.add_argument("--near-foot", type=float, nargs="+", default=[d.near_foot_weight])
    ap.add_argument("--along-heading", type=float, nargs="+", default=[d.along_heading_weight])
    ap.add_argument("--near-foot-sigma", type=float, nargs="+", default=[d.near_foot_sigma])
    ap.add_argument("--preset", default="exp:lupus-dribble-teacher")
    ap.add_argument("--iterations", type=int, default=None, help="shorter runs for the sweep")
    ap.add_argument("--ngpus", type=int, default=2)
    args, extra = ap.parse_known_args(argv)
    for c in sweep_commands(
        args.registry, args.near_foot, args.along_heading, args.near_foot_sigma, args.preset, args.iterations, args.ngpus, extra
    ):
        print(c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
