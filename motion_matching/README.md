# Motion matching

[Project overview](../README.md)

Generate parkour motion/terrain datasets for the Unitree G1.

To skip generation, use the [prepared example motion/terrain bundles](../wbt_training/README.md#example-motionterrain-datasets)
and follow the [teacher-training instructions](../wbt_training/README.md#train-a-teacher).

## Setup

From the PHP checkout root, with Conda installed:

```bash
bash motion_matching/setup_env.sh
conda activate php-motion-matching
python scripts/download_assets.py databases
cd motion_matching
```

Run the remaining commands from `motion_matching/`. Databases default to
`~/.cache/php-parkour/databases`; set `PHP_MOTION_DATABASE_DIR` before downloading
and generating to choose another writable directory.

The downloader checks the archive and every extracted file against the SHA-256
manifest in [release-assets.json](../release-assets.json).

The download provides the motion-matching databases and source terrains.
Generation uses these to create the motion/terrain datasets for RL training.
Robot models and required meshes are included in the Python package.

## Generate and inspect

```bash
python -m motion_matching.run --mode generate --scenario high_speed_climb_76
python -m motion_matching.motion_visualizer \
    --motion-file motion_matching_results/high_speed_climb_76
```

Generated motions are exported at 50 FPS. Each motion has paired terrain files
in `motion_matching_results/<scenario>/`:

| File | Contents |
|---|---|
| `*_motion.npz` | Reference motion and per-frame direction commands, when supplied |
| `*_terrain.npy` | Box obstacles with randomized terrain variants |
| `*_terrain.obj` | Terrain mesh for visualization |

Choose a scenario from [builtin.py](motion_matching/scenarios/builtin.py), such
as `low_speed_step`, `high_speed_climb_76`, or `random_locomotion`. Open the
visualizer's URL to inspect a generated motion.

For interactive motion matching, run `python -m motion_matching.run`.
Interactive keyboard control needs Tkinter.

## Upload to a W&B registry

Create a W&B registry named `terrains-motions` that accepts `terrains-motions`
artifacts (or all types), then upload:

```bash
python -m pip install -e '.[registry]'
wandb login
python tracking_utils/upload_npz_newformat.py \
    --registry_name high-speed-climb-76 \
    --file_folder motion_matching_results/high_speed_climb_76 \
    --entity your-team
```

`--registry_name` names the collection. The uploader prints a `REGISTRY`
reference such as the one below; its organization may differ from your upload
team. You can select any available version or `:latest` for
[training](../wbt_training/README.md#train-a-teacher).

```bash
REGISTRY=your-org/wandb-registry-terrains-motions/high-speed-climb-76:v0
```

## Add your own motions or scenarios

For a 30 FPS G1 source clip (`qpos` shaped `(T, 36)` in
`[root_quat_wxyz, root_pos, dof29]` order), build a database with its terrain:

```bash
python -m motion_matching.generate_database \
    --motion-file /path/to/source.npz \
    --terrain-file /path/to/box.obj \
    --fps 30 --output database_g1_skill_name.bin
python -m motion_matching.play_database --database database_g1_skill_name.bin
```

Set `--fps` to the source clip's frame rate. The builder resamples the motion to
60 FPS and writes the database to the cache.

Register new skills in
[skill_registry.py](motion_matching/scenarios/skill_registry.py), compose
scenarios in [builtin.py](motion_matching/scenarios/builtin.py), and run them
with `--mode generate --scenario <name>`.

## Robots

Robot-specific data (URDF, DoF count, foot and hip bones, joint axes, mirror
rules, simulation-bone joints) lives in a `RobotSpec` in
[robot_spec.py](motion_matching/kinematics/robot_spec.py). G1 is the default
(`G1_SPEC`). To add a robot, create a `RobotSpec`, register it in
`ROBOT_SPECS`, and pass `--robot <name>` to `generate_database`, `run` and
`play_database`. Joint axes and bone indices are derived from the URDF when the
spec is first used.

Use `joint_axes_frame="zero_pose"` for URDFs whose joint origins carry a
rotation (`rpy`), such as CAD exports where every axis is `0 0 1`. G1 uses
`"local"`, which ignores the origin rotation and is what its released
databases were built with.

`lupus` (`LUPUS_SPEC`) is registered too: 22 DoF (12 leg, 8 arm, 2 neck) from
`robots/lupus/lupus.urdf` in the repository (override the folder with the
`LUPUS_DIR` environment variable), `zero_pose` axes, feet on the ankle-roll
joints. In the base frame left and right joints share their axes, so mirroring
flips all roll and yaw joints (including `head_yaw`). There are no LUPUS
databases yet; they follow once retargeted clips exist (Phase 1 of the plan).

## Soccer: walk, kick, walk

The soccer scenarios in [soccer.py](motion_matching/scenarios/soccer.py)
replace the parkour one-hot with a continuous command: `vel_cmd` is `(T, 2)`
float32 `[v (m/s), omega (rad/s)]`, held at `(v_before, 0)` during a kick. The
ball replaces the terrain box: when a kick skill is entered, its annotated
resting ball is placed with the same clip-to-world transform a terrain box gets.

| Scenario | Content | `kick_enable` |
| --- | --- | --- |
| `walk_kick` | stand, random walk 0.1-3 s, straight run-up, kick, 2 s walk, stand | 1 |
| `stand_kick` | stand at the ball 0.5-3 s, rising edge, kick, stand or walk off | 0 then 1 |
| `stand_ball_outside_zone` | stand, ball just outside the kick zone | 0, 1 or 0 then 1 |
| `walk_ball_offpath` | random walk, ball 0.25-1.0 m beside the path | 0 or 1 |
| `walk_no_ball` | random walk, ball parked at least 5 m away | 0 or 1 |

Command ranges, corridor (kick under 0.10 m, ignore from 0.25 m) and the stand
kick zone are constants at the top of `soccer.py` (placeholders until LUPUS'
limits are known). Soccer motions carry these extra fields, checked by
`validate_kick_npz` in [kick_io.py](motion_matching/io/kick_io.py):
`object_pos_w`, `object_quat_w`, `object_lin_vel_w`, `kick_enable`,
`kick_phase` (0 walk/stand, 1 commit, 2 after contact), `contact_frame`,
`kick_dir` (rad, relative to the heading) and `kick_mode` (0 ignore, 1 kick,
2 dribble), plus `kick_side` and `scenario`. The ball radius comes from
`robots/ball/ball.yaml`.

Phase 6 scenarios (`PHASE6_SCENARIOS`, not in the default set):
`walk_kick_dir` / `stand_kick_dir` (as `walk_kick` / `stand_kick` with a random
`kick_dir` in +-30 deg per motion) and `walk_dribble` (slow walk with the ball
on the path in front of the robot, `kick_mode` 2). See
[wbt_training/PHASE6.md](../wbt_training/PHASE6.md).

Kick skills (`kick_right`, `kick_left`, `kick_stand_right`, `kick_stand_left`)
take their search ranges from a sidecar next to the database, built from
annotated clips (format in `kick_io.py`):

```bash
python -m motion_matching.build_kick_database --robot lupus --family kick_run path/to/run_kicks/
python -m motion_matching.build_kick_database --robot lupus --family kick_stand path/to/stand_kicks/
python -m motion_matching.run --mode generate --robot lupus --scenario walk_kick
```

Until the real clips exist, `python -m motion_matching.placeholder_clips --robot g1 --build`
writes synthetic stand-in clips and databases so the whole path can be run and
tested on a CPU.

### Training bundle and Phase-2 acceptance

`build_soccer_bundle` generates every Phase-2 scenario in the plan's shares
(walk_kick 40 %, stand_kick 15 %, stand_ball_outside_zone 5 %,
walk_ball_offpath 20 %, walk_no_ball 20 %; default `--total 750` = 300 walk
kicks, ~110 stand kicks) through the same `process_scenario` path as
`run --mode generate`, converts the motions to holosoma's layout (keeping the
kick fields; `kick_side` becomes per-frame, `scenario` moves to
`manifest.json`), validates the directory with
`wbt_training.utils.registry.validate_bundle` and runs the acceptance check:

```bash
python -m motion_matching.build_soccer_bundle --robot lupus --out ~/soccer_bundle --jobs 16
python -m motion_matching.build_soccer_bundle check ~/soccer_bundle      # re-check only
# teacher: --training.registry-name file://$HOME/soccer_bundle
```

Options: `--counts walk_kick=200,stand_kick=100` overrides single scenarios,
`--phase6` adds the Phase-6 scenarios, `--seed-offset N` draws other configs,
`--placeholder-db` uses the synthetic databases (smoke test only). The raw
motion-matching output stays in `<out>.raw/` for the visualizer.

The acceptance ([soccer_acceptance.py](motion_matching/soccer_acceptance.py))
checks every motion, not only samples: ball within 0.25 m (horizontal) of the
kick foot in the contact frame and resting until then; both kick feet in at
least 30 % of each kick scenario (standing leg at entry is reported);
run-up kicks entering in at least 3 of 4 step-phase bins; no contact and
correct ball placement in the negatives; flat terrain (no boxes, ground plane
only). Thresholds live in `AcceptanceConfig`.
