"""Build the Phase-2 soccer training bundle: all scenarios, plan shares, one directory.

Generates every Phase-2 scenario of :mod:`motion_matching.scenarios.soccer`
with the same per-motion code path as ``run --mode generate``
(:func:`motion_matching.run.process_scenario`), converts the motions into
holosoma's native NPZ layout (``wbt_training/training_runs/motion_convert.py``)
while keeping the kick fields, and writes a bundle that the teacher reads
directly::

    python -m motion_matching.build_soccer_bundle --robot lupus --out ~/soccer_bundle
    python wbt_training/train_agent.py exp:lupus-kick-teacher \
        --training.registry-name file://$HOME/soccer_bundle

Default amounts follow the plan (Phase 2, "Szenarien"): ``--total`` motions
split by :data:`PHASE2_SHARES` (walk_kick 40 %, stand_kick 15 %,
stand_ball_outside_zone 5 %, walk_ball_offpath 20 %, walk_no_ball 20 %); the
default total of 750 gives 300 walk kicks and ~110 stand kicks, inside the
plan's 200-400 / ~100. ``--counts walk_kick=200,stand_kick=100`` overrides
single scenarios; ``--phase6`` adds the Phase-6 scenarios.

Layout::

    <out>/manifest.json                        counts, shares, per-motion records, acceptance report
    <out>/<scenario>/<name>_motion.npz         holosoma motion + kick fields
    <out>/<scenario>/<name>_terrain.npy        (0, variants, 10): flat floor, no boxes
    <out>/<scenario>/<name>_terrain.obj        ground plane only (``--no-obj`` drops it)
    <out>.raw/<scenario>/...                   raw motion-matching output (kept for the visualizer)

Bundle motion files differ from the raw output only in two ways: the
motion part is in holosoma layout (root in ``joint_pos``, MJCF order,
``joint_names`` / ``body_names``), and the scalar ``kick_side`` / ``scenario``
are replaced by a per-frame ``kick_side`` (int8) - holosoma's
``combine_motions`` concatenates every key along frames and cannot take
0-d arrays. The scenario name is in ``manifest.json``.

``check <bundle>`` re-runs the Phase-2 acceptance (:mod:`motion_matching.soccer_acceptance`).

Without real kick databases, ``--placeholder-db`` bakes the synthetic
placeholder databases (:mod:`motion_matching.placeholder_clips`) into a
temporary directory first - for smoke tests only.
"""

from __future__ import annotations

import argparse
import inspect
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from motion_matching.io.kick_io import KICK_NPZ_FIELDS, KICK_SIDE, load_ball_radius
from motion_matching.kinematics.robot_spec import DEFAULT_ROBOT, ROBOT_SPECS
from motion_matching.scenarios import soccer as S

#: Plan shares of the Phase-2 scenarios ("Anteil" column of the plan table).
PHASE2_SHARES: dict[str, float] = {
    "walk_kick": 0.40,
    "stand_kick": 0.15,
    "stand_ball_outside_zone": 0.05,
    "walk_ball_offpath": 0.20,
    "walk_no_ball": 0.20,
}
#: Phase-6 scenarios, as a share of ``--total`` on top of Phase 2 (``--phase6``).
PHASE6_SHARES: dict[str, float] = {
    "walk_kick_dir": 0.20,
    "stand_kick_dir": 0.07,
    "walk_dribble": 0.13,
}
DEFAULT_TOTAL = 750

ALL_SCENARIOS: dict[str, Callable] = {**S.SOCCER_SCENARIOS, **S.PHASE6_SCENARIOS}


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


def plan_counts(
    total: int = DEFAULT_TOTAL,
    overrides: Optional[dict[str, int]] = None,
    phase6: bool = False,
) -> dict[str, int]:
    """Per-scenario motion counts: ``total`` split by the plan shares, then ``overrides``."""
    shares = dict(PHASE2_SHARES)
    if phase6:
        shares.update(PHASE6_SHARES)
    counts = {name: int(round(share * total)) for name, share in shares.items()}
    for name, n in (overrides or {}).items():
        if name not in ALL_SCENARIOS:
            raise ValueError(f"unknown soccer scenario {name!r}; known: {sorted(ALL_SCENARIOS)}")
        counts[name] = int(n)
    return {k: v for k, v in counts.items() if v > 0}


def parse_counts(text: str) -> dict[str, int]:
    out = {}
    for item in filter(None, (s.strip() for s in text.split(","))):
        name, _, value = item.partition("=")
        if not value:
            raise ValueError(f"--counts entry {item!r} must be scenario=N")
        out[name.strip()] = int(value)
    return out


def scenario_configs(name: str, count: int, seed_offset: int = 0) -> tuple[list[dict], float]:
    """Configs of one scenario; ``seed_offset`` shifts its default seed (new, disjoint draw)."""
    fn = ALL_SCENARIOS[name]
    kwargs: dict[str, Any] = {"num_configs": count}
    if seed_offset:
        default_seed = inspect.signature(fn).parameters["seed"].default
        kwargs["seed"] = int(default_seed) + int(seed_offset)
    return fn(**kwargs)


# --------------------------------------------------------------------------
# Generation (same per-motion path as run --mode generate)
# --------------------------------------------------------------------------


def _process_safe(cfg: dict, folder: str, force: bool, speed_up: float) -> tuple[str, Optional[str]]:
    from motion_matching import run as mm_run

    try:
        mm_run.process_scenario(cfg, folder, force, speed_up)
    except Exception as e:  # noqa: BLE001 - one bad motion must not kill the bundle
        return cfg["name"], f"{type(e).__name__}: {e}"
    return cfg["name"], None


def generate_motions(
    tasks: list[tuple[dict, str, bool, float]], robot: str, jobs: int = 0
) -> dict[str, Optional[str]]:
    """Run ``process_scenario`` for every task; returns ``{name: error or None}``."""
    from motion_matching import run as mm_run

    required = mm_run.collect_required_skills([t[0] for t in tasks])
    requested = jobs if jobs > 0 else multiprocessing.cpu_count()
    num_jobs = min(requested, len(tasks))
    results: dict[str, Optional[str]] = {}
    if num_jobs <= 1:
        mm_run.init_worker(required_skills=required, robot=robot)
        for task in tasks:
            name, err = _process_safe(*task)
            results[name] = err
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=num_jobs, initializer=mm_run.init_worker, initargs=(required, robot)) as pool:
            for name, err in pool.starmap(_process_safe, tasks):
                results[name] = err
    return results


# --------------------------------------------------------------------------
# Conversion into the bundle
# --------------------------------------------------------------------------


def _motion_convert():
    try:
        from wbt_training.training_runs.motion_convert import convert
    except ImportError:
        from motion_matching.utils import PKG_ROOT

        sys.path.insert(0, str(PKG_ROOT.parents[1]))
        from wbt_training.training_runs.motion_convert import convert
    return convert


def convert_into_bundle(raw_motion: Path, out_motion: Path) -> dict[str, Any]:
    """holosoma layout + kick fields (+ per-frame ``kick_side``); returns record metadata."""
    convert = _motion_convert()
    out_motion.parent.mkdir(parents=True, exist_ok=True)
    if not convert(str(raw_motion), str(out_motion)):
        raise ValueError(f"{raw_motion} has fewer than two frames")
    with np.load(raw_motion, allow_pickle=False) as raw:
        extra = {k: raw[k] for k in KICK_NPZ_FIELDS if k in raw.files}
        side = int(raw["kick_side"]) if "kick_side" in raw.files else -1
        scenario = str(raw["scenario"]) if "scenario" in raw.files else ""
        if "vel_cmd" in raw.files:
            extra["vel_cmd"] = raw["vel_cmd"]
    with np.load(out_motion, allow_pickle=False) as conv:
        out = {k: conv[k] for k in conv.files}
    T = len(out["joint_pos"])
    for key, value in extra.items():
        if len(value) != T:
            raise ValueError(f"{raw_motion.name}: {key} has {len(value)} frames, motion {T}")
        out[key] = value
    out["kick_side"] = np.full(T, side, dtype=np.int8)
    np.savez(out_motion, **out)
    return {"frames": T, "fps": int(np.asarray(out["fps"]).reshape(-1)[0]), "kick_side_code": side, "scenario": scenario}


def _git_commit() -> Optional[str]:
    try:
        root = Path(__file__).resolve().parents[2]
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def validate_with_registry(bundle: Path) -> Optional[str]:
    """``wbt_training.utils.registry.validate_bundle``; error text or ``None``."""
    try:
        try:
            from wbt_training.utils.registry import validate_bundle
        except ImportError:
            _motion_convert()  # puts the repo root on sys.path
            from wbt_training.utils.registry import validate_bundle
        validate_bundle(bundle)
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"
    return None


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def build_bundle(
    out: os.PathLike,
    *,
    robot: str = DEFAULT_ROBOT,
    counts: Optional[dict[str, int]] = None,
    total: int = DEFAULT_TOTAL,
    phase6: bool = False,
    raw_dir: Optional[os.PathLike] = None,
    jobs: int = 0,
    seed_offset: int = 0,
    keep_obj: bool = True,
    force: bool = False,
    check: bool = True,
) -> dict[str, Any]:
    """Generate, convert, validate and check a soccer bundle; returns the manifest dict."""
    from motion_matching.soccer_acceptance import check_bundle

    out = Path(out).expanduser().resolve()
    raw = Path(raw_dir).expanduser().resolve() if raw_dir else out.with_name(out.name + ".raw")
    if out == raw or raw.is_relative_to(out):
        raise ValueError("the raw directory must lie outside the bundle (the registry globs every *motion*.npz)")
    if out.exists() and any(out.iterdir()):
        if not force:
            raise FileExistsError(f"{out} is not empty; pass --force to rebuild it")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    plan = counts if counts is not None else plan_counts(total, phase6=phase6)
    t0 = time.time()
    tasks, cfg_by_name = [], {}
    for scenario, n in plan.items():
        configs, speed_up = scenario_configs(scenario, n, seed_offset)
        folder = raw / scenario
        folder.mkdir(parents=True, exist_ok=True)
        for cfg in configs:
            tasks.append((cfg, str(folder), force, speed_up))
            cfg_by_name[cfg["name"]] = (scenario, cfg)
    print(f"Generating {len(tasks)} soccer motions for {robot}: {plan}")
    errors = generate_motions(tasks, robot, jobs)

    records, failed = [], []
    for name, (scenario, cfg) in cfg_by_name.items():
        err = errors.get(name)
        raw_motion = raw / scenario / f"{name}_motion.npz"
        if err is None and not raw_motion.is_file():
            err = "no output written"
        if err is not None:
            failed.append({"name": name, "scenario": scenario, "error": err})
            continue
        dst = out / scenario / raw_motion.name
        try:
            meta = convert_into_bundle(raw_motion, dst)
        except Exception as e:  # noqa: BLE001
            failed.append({"name": name, "scenario": scenario, "error": f"convert: {e}"})
            continue
        npy = raw_motion.with_name(f"{name}_terrain.npy")
        if npy.is_file():
            shutil.copy2(npy, dst.with_name(npy.name))
        obj = npy.with_suffix(".obj")
        if keep_obj and obj.is_file():
            shutil.copy2(obj, dst.with_name(obj.name))
        soccer = cfg["soccer"]
        records.append({
            "file": dst.relative_to(out).as_posix(),
            "name": name,
            "scenario": scenario,
            "seed": soccer.get("seed"),
            "ball_mode": soccer["ball"]["mode"],
            "kick_side": soccer.get("kick_side"),
            "kick_dir": soccer.get("kick_dir"),
            "frames": meta["frames"],
            "duration_s": round(meta["frames"] / meta["fps"], 3),
        })

    generated = {s: sum(r["scenario"] == s for r in records) for s in plan}
    n_ok = max(len(records), 1)
    manifest: dict[str, Any] = {
        "format": "php_soccer_bundle/1",
        "robot": robot,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "ball_radius": load_ball_radius(),
        "planned_counts": plan,
        "counts": generated,
        "shares": {s: round(n / n_ok, 4) for s, n in generated.items()},
        "plan_shares": {**PHASE2_SHARES, **(PHASE6_SHARES if phase6 else {})},
        "total_frames": int(sum(r["frames"] for r in records)),
        "total_duration_s": round(sum(r["duration_s"] for r in records), 1),
        "seed_offset": seed_offset,
        "raw_dir": str(raw),
        "kick_side_codes": {k or "none": v for k, v in KICK_SIDE.items()},
        "failed": failed,
        "motions": records,
        "build_seconds": round(time.time() - t0, 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    manifest["registry_validation"] = validate_with_registry(out) or "ok"
    if check:
        report = check_bundle(out)
        if manifest["registry_validation"] != "ok":
            report.fail(f"registry validate_bundle: {manifest['registry_validation']}")
        manifest["acceptance"] = report.to_dict()
        print(report.summary())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Bundle: {len(records)} motions in {out} ({len(failed)} failed); use --training.registry-name file://{out}")
    return manifest


def _use_placeholder_databases(robot: str) -> Path:
    """Bake placeholder DBs into a temp dir and point this process and its workers at it."""
    import motion_matching.utils as mm_utils
    from motion_matching.placeholder_clips import build_placeholder_databases

    db_dir = Path(tempfile.mkdtemp(prefix="php_soccer_placeholder_db_"))
    build_placeholder_databases(robot, db_dir / "clips", db_dir)
    os.environ["PHP_MOTION_DATABASE_DIR"] = str(db_dir)
    mm_utils.DATABASE_DIR = db_dir
    print(f"Using PLACEHOLDER databases in {db_dir} (synthetic clips, smoke test only)")
    return db_dir


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "check":
        from motion_matching.soccer_acceptance import AcceptanceConfig, check_bundle

        p = argparse.ArgumentParser(prog="build_soccer_bundle check", description="Phase-2 acceptance of a bundle")
        p.add_argument("bundle")
        p.add_argument("--max-ball-foot-distance", type=float, default=AcceptanceConfig.max_ball_foot_distance)
        p.add_argument("--json", action="store_true", help="print the full report as JSON")
        a = p.parse_args(argv[1:])
        report = check_bundle(Path(a.bundle), AcceptanceConfig(max_ball_foot_distance=a.max_ball_foot_distance))
        err = validate_with_registry(Path(a.bundle))
        if err:
            report.fail(f"registry validate_bundle: {err}")
        print(json.dumps(report.to_dict(), indent=2) if a.json else report.summary())
        return 0 if report.passed else 1

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True, help="bundle directory (use as file://<out>)")
    p.add_argument("--robot", default=DEFAULT_ROBOT, choices=sorted(ROBOT_SPECS))
    p.add_argument("--total", type=int, default=DEFAULT_TOTAL, help="motions split by the plan shares")
    p.add_argument("--counts", default="", help="overrides, e.g. walk_kick=200,stand_kick=100")
    p.add_argument("--phase6", action="store_true", help="add walk_kick_dir, stand_kick_dir, walk_dribble")
    p.add_argument("--raw-dir", default=None, help="raw motion-matching output (default <out>.raw)")
    p.add_argument("--jobs", type=int, default=0, help="parallel workers (0 = all CPUs)")
    p.add_argument("--seed-offset", type=int, default=0, help="shift every scenario seed (another draw)")
    p.add_argument("--no-obj", action="store_true", help="leave the ground-plane OBJ out of the bundle")
    p.add_argument("--no-check", action="store_true", help="skip the Phase-2 acceptance check")
    p.add_argument("--force", action="store_true", help="rebuild a non-empty bundle / overwrite raw output")
    p.add_argument("--placeholder-db", action="store_true", help="synthetic databases (smoke test only)")
    a = p.parse_args(argv)
    counts = plan_counts(a.total, parse_counts(a.counts), a.phase6)
    if a.placeholder_db:
        _use_placeholder_databases(a.robot)
    manifest = build_bundle(
        a.out, robot=a.robot, counts=counts, raw_dir=a.raw_dir, jobs=a.jobs, seed_offset=a.seed_offset,
        keep_obj=not a.no_obj, force=a.force, check=not a.no_check, phase6=a.phase6,
    )
    ok = manifest["registry_validation"] == "ok" and manifest.get("acceptance", {}).get("passed", True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
