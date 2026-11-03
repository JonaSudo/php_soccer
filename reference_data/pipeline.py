"""Batch run of Phase 1 over ``catalog.yaml``.

Three stages, because retargeting runs in the holosoma_retargeting
environment and the rest in the motion_matching environment::

    # 1) source clips -> neutral human clips (either env; AMASS needs nothing extra)
    python -m reference_data.pipeline human --root lafan=~/data/lafan1 \\
        --root omnicontact=~/data/OmniContact-Dataset --root amass=~/data/amass_rt --work work/

    # 2) retargeting (holosoma_retargeting env, CPU, about 0.1-0.3 s per frame)
    python -m reference_data.pipeline retarget --work work/ --jobs 8

    # 3) PHP export, annotation, acceptance check (motion_matching env)
    python -m reference_data.pipeline post --work work/ --out reference_out/

Stages skip clips whose output already exists (``--force`` redoes them), so
a batch can be stopped and resumed. ``--only <id>`` limits a stage to
catalog ids starting with that prefix.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CATALOG = HERE / "catalog.yaml"


@dataclass
class Entry:
    id: str
    source: str
    kind: str
    path: Path
    group: str
    trim: list | None = None
    speed: float | str = "auto"
    speeds: list[float] = field(default_factory=list)
    mirror: bool = True
    annotation: dict = field(default_factory=dict)
    npz: Path | None = None   # omnicontact: processed npz with the ball
    notes: str = ""


def load_catalog(path: Path | str = CATALOG) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _omnicontact_npz(npz_dir: Path, bvh: Path) -> Path | None:
    """Processed npz of a raw capture: file names start with the capture id."""
    capture = bvh.parent.name
    hits = sorted(npz_dir.glob(f"{capture}*.npz"))
    if not hits:
        hits = sorted(p for p in npz_dir.glob("*.npz") if capture in p.name)
    return hits[0] if hits else None


def expand(catalog: dict, roots: dict[str, Path], only: str | None = None) -> list[Entry]:
    entries = []
    for c in catalog["clips"]:
        if only and not c["id"].startswith(only):
            continue
        src = c["source"]
        if src not in roots:
            print(f"skip {c['id']}: no --root {src}=...", file=sys.stderr)
            continue
        kind = catalog["sources"][src]["kind"]
        d = {**catalog.get("defaults", {}).get(c["group"], {}), **c}
        root = roots[src]
        matches = sorted(root.glob(c["path"])) if any(ch in c["path"] for ch in "*?[") else [root / c["path"]]
        for m in matches:
            if not m.exists():
                print(f"skip {c['id']}: {m} not found", file=sys.stderr)
                continue
            many = len(matches) > 1
            stem = m.parent.name if m.name == "motion_actor.bvh" else m.name.split(".")[0]
            e = Entry(
                id=f"{c['id']}_{stem}" if many else c["id"],
                source=src,
                kind=kind,
                path=m,
                group=d["group"],
                trim=d.get("trim"),
                speed=d.get("speed", "auto"),
                speeds=list(d.get("speeds", [])),
                mirror=d.get("mirror", True),
                annotation=dict(d.get("annotation") or {}),
                notes=d.get("notes", ""),
            )
            if kind == "omnicontact":
                e.npz = _omnicontact_npz(root / c["npz_dir"], m)
                if e.npz is None:
                    print(f"skip {e.id}: no processed npz in {c['npz_dir']}", file=sys.stderr)
                    continue
            entries.append(e)
    return entries


# ---- stages -------------------------------------------------------------------


def stage_human(entries: list[Entry], work: Path, force: bool) -> None:
    from reference_data import human_clip as hc

    out = work / "human"
    out.mkdir(parents=True, exist_ok=True)
    for e in entries:
        dst = out / f"{e.id}.human.npz"
        if dst.exists() and not force:
            continue
        if e.kind == "bvh":
            clip = hc.from_bvh(e.path)
        elif e.kind == "omnicontact":
            clip = hc.from_omnicontact(e.path, e.npz)
        elif e.kind == "amass":
            clip = hc.from_amass_prepped(e.path)
        else:
            raise ValueError(f"unknown source kind {e.kind!r}")
        if e.trim:
            a, b = e.trim
            clip = clip.trim(int(round(a * clip.fps)), None if b is None else int(round(b * clip.fps)))
        clip.meta["catalog"] = {"id": e.id, "group": e.group, "source": e.source}
        clip.save(dst)
        print(f"human  {e.id}: {clip.num_frames} frames @ {clip.fps:g} fps{' + ball' if clip.has_ball else ''}")


def _retarget_one(src: str, dst: str) -> str:
    from reference_data.human_clip import HumanClip
    from reference_data.retarget_lupus import retarget

    retarget(HumanClip.load(src), dst)
    return dst


def stage_retarget(work: Path, force: bool, jobs: int, only: str | None) -> None:
    src_dir, out = work / "human", work / "raw"
    out.mkdir(parents=True, exist_ok=True)
    todo = []
    for src in sorted(src_dir.glob("*.human.npz")):
        cid = src.name[: -len(".human.npz")]
        if only and not cid.startswith(only):
            continue
        dst = out / f"{cid}.lupus_raw.npz"
        if force or not dst.exists():
            todo.append((str(src), str(dst)))
    if jobs <= 1:
        for s, d in todo:
            print("retarget", _retarget_one(s, d))
        return
    with ProcessPoolExecutor(jobs) as pool:
        futs = {pool.submit(_retarget_one, s, d): s for s, d in todo}
        for f in as_completed(futs):
            try:
                print("retarget", f.result())
            except Exception as exc:  # keep the batch going
                print(f"retarget FAILED {futs[f]}: {exc}", file=sys.stderr)


def stage_post(entries: list[Entry], work: Path, out: Path, force: bool) -> int:
    from reference_data.clip_ops import LupusClip
    from reference_data.postprocess import ClipSpec, kick_json_path, process, write_manifest
    from reference_data.verify import verify_clip

    out.mkdir(parents=True, exist_ok=True)
    exported = []
    for e in entries:
        raw = work / "raw" / f"{e.id}.lupus_raw.npz"
        if not raw.exists():
            print(f"post   {e.id}: not retargeted yet", file=sys.stderr)
            continue
        clip = LupusClip.from_retarget(raw, e.id)
        fps = clip.fps
        ann = {
            k: (int(round(v * fps)) if k in ("contact_frame", "skill_start", "skill_end", "entry_start") else v)
            for k, v in e.annotation.items()
        }
        spec = ClipSpec(
            id=e.id, group=e.group, speed=e.speed, speeds=tuple(e.speeds) or ClipSpec.speeds,
            mirror=e.mirror, annotation=ann, notes=e.notes,
        )
        done = sorted(p for p in (out / e.group).glob(f"{e.id}_*.json") if not p.name.endswith(".kick.json"))
        if done and not force:
            continue
        for x in process(clip, spec, out):
            exported.append(x)
            v = verify_clip(x.npz)
            rec = json.loads(x.json.read_text())
            rec["verify"] = {"ok": v.ok, "lowest": v.lowest, "contact_gap": v.contact_gap, "problems": v.problems}
            x.json.write_text(json.dumps(rec, indent=2))
            if not v.ok:
                kick_json_path(x.npz).unlink(missing_ok=True)
            status = "ok" if (x.accepted and v.ok) else "REJECT " + "; ".join(x.reasons + v.problems)
            print(f"post   {x.name}: {status}")
    # Manifest over everything in the output folder, not just this run.
    from reference_data.postprocess import Exported

    allx = []
    for js in sorted(out.rglob("*.json")):
        if js.name == "manifest.json" or js.name.endswith(".kick.json"):
            continue
        rec = json.loads(js.read_text())
        ok = rec.get("accepted", False) and rec.get("verify", {}).get("ok", False)
        allx.append(Exported(rec["name"], js.with_suffix(".npz"), js, ok, rec.get("reasons", []) + rec.get("verify", {}).get("problems", [])))
    write_manifest(out, allx)
    return sum(not x.accepted for x in allx)


def summary(out: Path) -> dict:
    """Seconds of accepted material per group (plan targets in the README)."""
    res: dict = {}
    for js in sorted(Path(out).rglob("*.json")):
        if js.name == "manifest.json" or js.name.endswith(".kick.json"):
            continue
        rec = json.loads(js.read_text())
        g = res.setdefault(rec["group"], {"clips": 0, "accepted": 0, "seconds": 0.0})
        g["clips"] += 1
        if rec.get("accepted") and rec.get("verify", {}).get("ok"):
            g["accepted"] += 1
            g["seconds"] += (rec["num_frames"] - 1) / rec["fps"]
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["human", "retarget", "post", "summary"])
    ap.add_argument("--catalog", type=Path, default=CATALOG)
    ap.add_argument("--root", action="append", default=[], help="<source>=<dir>")
    ap.add_argument("--work", type=Path, default=Path("work"))
    ap.add_argument("--out", type=Path, default=Path("reference_out"))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--only")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    roots = {}
    for r in args.root:
        k, _, v = r.partition("=")
        roots[k] = Path(v).expanduser()
    if args.stage == "retarget":
        stage_retarget(args.work, args.force, args.jobs, args.only)
        return 0
    if args.stage == "summary":
        print(json.dumps(summary(args.out), indent=2))
        return 0
    if args.stage == "human":
        stage_human(expand(load_catalog(args.catalog), roots, args.only), args.work, args.force)
        return 0
    # post needs no data roots: entries are rebuilt from the human clips' catalog ids
    entries = _entries_from_work(load_catalog(args.catalog), args.work, args.only)
    stage_post(entries, args.work, args.out, args.force)
    print(json.dumps(summary(args.out), indent=2))
    return 0


def _entries_from_work(catalog: dict, work: Path, only: str | None) -> list[Entry]:
    """Catalog entries for the clips present in work/raw (no data roots needed)."""
    by_id = {c["id"]: c for c in catalog["clips"]}
    entries = []
    for raw in sorted((work / "raw").glob("*.lupus_raw.npz")):
        cid = raw.name[: -len(".lupus_raw.npz")]
        if only and not cid.startswith(only):
            continue
        base = by_id.get(cid) or next(
            (c for k, c in sorted(by_id.items(), key=lambda kv: -len(kv[0])) if cid.startswith(k + "_")), None
        )
        if base is None:
            print(f"post   {cid}: not in the catalog", file=sys.stderr)
            continue
        d = {**catalog.get("defaults", {}).get(base["group"], {}), **base}
        entries.append(
            Entry(
                id=cid, source=d["source"], kind=catalog["sources"][d["source"]]["kind"], path=raw,
                group=d["group"], trim=d.get("trim"), speed=d.get("speed", "auto"),
                speeds=list(d.get("speeds", [])), mirror=d.get("mirror", True),
                annotation=dict(d.get("annotation") or {}), notes=d.get("notes", ""),
            )
        )
    return entries


if __name__ == "__main__":
    sys.exit(main())
