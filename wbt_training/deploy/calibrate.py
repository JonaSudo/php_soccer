"""Turn measurements on the robot / field into randomisation ranges (plan, Phase 3-5).

Three calibrations, each a function plus a CLI subcommand that prints the
suggested values and where they go:

``ball``
    Roll-out of the real ball on the turf. Input CSV with either a ball track
    (columns ``t, x, y``, one roll, e.g. from a video) or one row per roll
    (``v0, distance``). Fits the decay ``v' = -c v`` that the simulators model
    (PhysX and MuJoCo have no rolling friction) and also a constant rolling
    deceleration for comparison. Output: ``linear_damping``/``angular_damping``
    for ``robots/ball/ball.yaml`` and the damping DR range in ``lupus_kick.py``.
    With equal linear and angular damping ``c`` a rolling shell slows down with
    exactly ``c`` (``0.6 c_lin + 0.4 c_ang``).

``vision``
    Ball detections against ground truth (tape measure or mocap). CSV columns
    ``true_x, true_y`` (ball on the ground in the vision output frame) and
    ``det_x, det_y`` (empty if the ball was not detected), optional
    ``cam_x, cam_y`` (camera position in that frame, default 0) and
    ``gp_x, gp_y`` (position from the ground-plane projection, if the vision
    stack computes it). Fits ``depth_sigma_k`` (sigma = k * distance),
    ``depth_bias_range``, ``bearing_sigma_deg``, the usable ``max_range`` and,
    with ``gp_*``, the distance below which the ground projection beats the
    depth net (``near_ground_projection_range``). Optional ``capture_time``
    column (s) gives ``update_hz_range``. Values go into ``BallVisionCfg``.

``latency``
    One or more ``PolicyNode.stats.as_dict()`` JSON files (what
    ``lupus_kick_sim2sim --out`` writes under ``latency`` and what the robot
    loop should dump). Gives ``latency_s_range`` (camera -> policy) and, with
    the measured policy -> motor latency, ``ctrl_delay_step_range`` for the
    teacher preset.

All functions are numpy-only so they also run on the Jetson.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- io


def read_csv(path: str | Path) -> dict[str, np.ndarray]:
    """Columns of a CSV as float arrays; empty cells become NaN."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path}: no rows")
    out = {}
    for k in rows[0]:
        out[k.strip()] = np.array([float(r[k]) if r[k] not in ("", None) else np.nan for r in rows])
    return out


def _range(lo: float, hi: float, margin: float, floor: float = 0.0) -> list[float]:
    """[lo, hi] widened by ``margin`` (relative), rounded, not below ``floor``."""
    span = max(hi - lo, abs(hi) * 0.05, 1e-6)
    return [round(max(floor, lo - margin * span), 4), round(hi + margin * span, 4)]


# --------------------------------------------------------------------------- ball


def fit_ball_track(t: np.ndarray, x: np.ndarray, y: np.ndarray, min_speed: float = 0.15) -> dict[str, float]:
    """Fit ``v(t) = v0 exp(-c t)`` (and ``v0 - a t``) to one roll-out track."""
    t, x, y = (np.asarray(a, float) for a in (t, x, y))
    order = np.argsort(t)
    t, x, y = t[order], x[order], y[order]
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])  # path length
    tm = 0.5 * (t[1:] + t[:-1])
    v = np.diff(s) / np.diff(t)
    keep = v > min_speed  # end of the roll: speed noise dominates
    if keep.sum() < 3:
        raise ValueError("track too short: need at least 3 samples above min_speed")
    tm, v = tm[keep], v[keep]
    # exponential: log v = log v0 - c t
    slope, icpt = np.polyfit(tm, np.log(v), 1)
    c, v0 = float(-slope), float(math.exp(icpt))
    # constant deceleration: v = v0 - a t
    a_slope, a_icpt = np.polyfit(tm, v, 1)
    res_exp = float(np.sqrt(np.mean((v0 * np.exp(-c * tm) - v) ** 2)))
    res_lin = float(np.sqrt(np.mean((a_icpt + a_slope * tm - v) ** 2)))
    return {
        "c": c,
        "v0": v0,
        "distance": float(s[-1]),
        "decel": float(-a_slope),
        "rms_exp": res_exp,
        "rms_const_decel": res_lin,
    }


def fit_ball_rolls(v0: np.ndarray, distance: np.ndarray) -> dict[str, float]:
    """``c = v0 / distance`` per roll (exponential decay rolls ``v0 / c`` far)."""
    v0, distance = np.asarray(v0, float), np.asarray(distance, float)
    ok = (v0 > 0) & (distance > 0)
    c = v0[ok] / distance[ok]
    decel = v0[ok] ** 2 / (2 * distance[ok])
    # if c depends strongly on v0, a constant deceleration fits better than damping
    trend = float(np.corrcoef(v0[ok], c)[0, 1]) if ok.sum() > 2 else float("nan")
    return {"c": float(np.median(c)), "c_min": float(c.min()), "c_max": float(c.max()), "decel": float(np.median(decel)), "c_vs_v0_corr": trend, "n": int(ok.sum())}


def ball_suggestion(fits: Sequence[float], margin: float = 0.25) -> dict:
    cs = np.asarray(fits, float)
    c = float(np.median(cs))
    lo, hi = float(cs.min()), float(cs.max())
    return {
        "robots/ball/ball.yaml": {
            "linear_damping": {"value": round(c, 4), "status": "measured"},
            "angular_damping": {"value": round(c, 4), "status": "measured"},
        },
        "lupus_kick.py randomize_object_linear_damping_startup.damping_range": _range(lo, hi, margin),
        "lupus_kick.py randomize_object_angular_damping_startup.damping_range": _range(lo, hi, margin),
        "note": "equal linear and angular damping c give a rolling ball exactly v' = -c v in both simulators",
    }


# --------------------------------------------------------------------------- vision


def fit_vision(cols: dict[str, np.ndarray], bins: int = 6) -> dict:
    tx, ty = cols["true_x"], cols["true_y"]
    cx = cols.get("cam_x", np.zeros_like(tx))
    cy = cols.get("cam_y", np.zeros_like(tx))
    cx, cy = np.nan_to_num(cx), np.nan_to_num(cy)
    d_true = np.hypot(tx - cx, ty - cy)
    found = ~np.isnan(cols["det_x"]) & ~np.isnan(cols["det_y"])
    if found.sum() < 5:
        raise ValueError("need at least 5 detections")
    dx, dy = cols["det_x"][found] - cx[found], cols["det_y"][found] - cy[found]
    dt_ = d_true[found]
    d_det = np.hypot(dx, dy)
    rel = (d_det - dt_) / dt_
    bearing = np.degrees(np.angle(np.exp(1j * (np.arctan2(dy, dx) - np.arctan2(ty[found] - cy[found], tx[found] - cx[found])))))
    bias = float(np.mean(rel))
    k = float(np.std(rel))
    out: dict = {
        "n": int(len(tx)),
        "detections": int(found.sum()),
        "depth_sigma_k": round(k, 4),
        "depth_bias_mean": round(bias, 4),
        "bearing_sigma_deg": round(float(np.std(bearing)), 3),
        "bearing_bias_deg": round(float(np.mean(bearing)), 3),
    }
    # per-distance table: error and detection rate
    edges = np.quantile(d_true, np.linspace(0, 1, bins + 1))
    table = []
    gp = "gp_x" in cols and "gp_y" in cols
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d_true >= lo) & (d_true <= hi)
        mf = m & found
        row = {"d_lo": round(float(lo), 3), "d_hi": round(float(hi), 3), "n": int(m.sum()), "detect_rate": float(mf.sum() / max(m.sum(), 1))}
        if mf.any():
            err = np.hypot(cols["det_x"][mf] - tx[mf], cols["det_y"][mf] - ty[mf])
            row["depth_err_rms"] = float(np.sqrt(np.mean(err**2)))
            if gp:
                g = mf & ~np.isnan(cols["gp_x"])
                if g.any():
                    eg = np.hypot(cols["gp_x"][g] - tx[g], cols["gp_y"][g] - ty[g])
                    row["ground_err_rms"] = float(np.sqrt(np.mean(eg**2)))
        table.append(row)
    out["table"] = table
    good = [r for r in table if r["detect_rate"] >= 0.5]
    out["max_range"] = round(max((r["d_hi"] for r in good), default=float(edges[-1])), 3)
    out["dropout_near"] = round(1.0 - (table[0]["detect_rate"] if table else 1.0), 3)
    if gp:
        better = [r["d_hi"] for r in table if r.get("ground_err_rms", math.inf) < r.get("depth_err_rms", -1)]
        # contiguous from the nearest bin only
        near = 0.0
        for r in table:
            if r.get("ground_err_rms", math.inf) < r.get("depth_err_rms", -1):
                near = r["d_hi"]
            else:
                break
        out["near_ground_projection_range"] = round(near, 3)
        out["ground_better_bins"] = better
    if "capture_time" in cols:
        ts = np.sort(cols["capture_time"][~np.isnan(cols["capture_time"])])
        dts = np.diff(ts)
        dts = dts[dts > 0]
        if len(dts) > 2:
            hz = 1.0 / dts
            out["update_hz_p5_p95"] = [round(float(np.percentile(hz, 5)), 2), round(float(np.percentile(hz, 95)), 2)]
    return out


def vision_suggestion(fit: dict, margin: float = 0.25) -> dict:
    b = abs(fit["depth_bias_mean"])
    sug = {
        "depth_sigma_k": round(fit["depth_sigma_k"] * (1 + margin), 4),
        "depth_bias_range": [round(-(b + 0.01), 4), round(b + 0.01, 4)],
        "bearing_sigma_deg": round(fit["bearing_sigma_deg"] * (1 + margin), 3),
        "max_range": fit["max_range"],
    }
    if "near_ground_projection_range" in fit:
        sug["near_ground_projection_range"] = fit["near_ground_projection_range"]
    if "update_hz_p5_p95" in fit:
        sug["update_hz_range"] = fit["update_hz_p5_p95"]
    return {"BallVisionCfg (wbt_training/utils/ball_detection.py)": sug}


# --------------------------------------------------------------------------- latency


def _merge_series(stats: Sequence[dict], key: str) -> dict[str, float]:
    s = [st[key] for st in stats if st.get(key, {}).get("n", 0)]
    if not s:
        return {}
    n = sum(x["n"] for x in s)
    return {
        "n": n,
        "mean": sum(x["mean"] * x["n"] for x in s) / n,
        "p50": float(np.median([x["p50"] for x in s])),
        "p99": max(x["p99"] for x in s),
        "max": max(x["max"] for x in s),
    }


def latency_suggestion(stats: Sequence[dict], policy_dt: float = 0.02, actuation_latency_s: float | None = None, margin: float = 0.25) -> dict:
    """Randomisation ranges from measured loop statistics.

    ``actuation_latency_s``: measured time from sending a target to the motor
    acting on it (firmware + bus), not visible to the policy process.
    """
    vis = _merge_series(stats, "vision_latency_s")
    comp = _merge_series(stats, "compute_s")
    period = _merge_series(stats, "loop_period_s")
    out: dict = {"measured": {"vision_latency_s": vis, "compute_s": comp, "loop_period_s": period, "overruns": sum(int(s.get("overruns", 0)) for s in stats)}}
    sug: dict = {}
    if vis:
        sug["BallVisionCfg.latency_s_range"] = _range(vis["p50"] * 0.8, vis["p99"], margin)
    if comp:
        act = actuation_latency_s or 0.0
        lo = (comp["p50"] + act) / policy_dt
        hi = (comp["p99"] + act) / policy_dt
        sug["lupus_kick.py setup_action_delay_buffers.ctrl_delay_step_range"] = [int(math.floor(lo)), max(int(math.ceil(hi)), 1)]
        if actuation_latency_s is None:
            out["note"] = "actuation latency (target -> motor) not given; ctrl_delay_step_range covers compute only"
    if period:
        out["loop_jitter_ms_p99"] = round(1000 * (period["p99"] - policy_dt), 3)
    out["suggested"] = sug
    return out


# --------------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("ball", help="ball roll-out -> damping")
    b.add_argument("csv", nargs="+", help="track CSVs (t,x,y) or one CSV with v0,distance rows")
    v = sub.add_parser("vision", help="detections vs ground truth -> BallVisionCfg")
    v.add_argument("csv")
    v.add_argument("--bins", type=int, default=6)
    lt = sub.add_parser("latency", help="PolicyNode stats JSON -> latency / delay ranges")
    lt.add_argument("json", nargs="+")
    lt.add_argument("--policy-dt", type=float, default=0.02)
    lt.add_argument("--actuation-latency-ms", type=float, default=None)
    for p in (b, v, lt):
        p.add_argument("--margin", type=float, default=0.25, help="widen ranges by this fraction")
        p.add_argument("--out", default=None, help="also write the result as JSON")
    args = ap.parse_args(argv)

    if args.cmd == "ball":
        fits, report = [], {"fits": []}
        for path in args.csv:
            cols = read_csv(path)
            if {"t", "x", "y"} <= cols.keys():
                f = fit_ball_track(cols["t"], cols["x"], cols["y"])
                fits.append(f["c"])
            elif {"v0", "distance"} <= cols.keys():
                f = fit_ball_rolls(cols["v0"], cols["distance"])
                fits += [f["c_min"], f["c"], f["c_max"]]
            else:
                raise SystemExit(f"{path}: need columns t,x,y or v0,distance")
            report["fits"].append({"file": str(path), **f})
        report["suggested"] = ball_suggestion(fits, args.margin)
    elif args.cmd == "vision":
        fit = fit_vision(read_csv(args.csv), args.bins)
        report = {"fit": fit, "suggested": vision_suggestion(fit, args.margin)}
    else:
        stats = []
        for path in args.json:
            d = json.loads(Path(path).read_text())
            stats.append(d.get("latency", d))
        act = None if args.actuation_latency_ms is None else args.actuation_latency_ms / 1000
        report = latency_suggestion(stats, args.policy_dt, act, args.margin)
    text = json.dumps(report, indent=1, default=float)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "ball_suggestion",
    "fit_ball_rolls",
    "fit_ball_track",
    "fit_vision",
    "latency_suggestion",
    "read_csv",
    "vision_suggestion",
]
