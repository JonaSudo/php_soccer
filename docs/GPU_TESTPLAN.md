# GPU-Testplan LUPUS Walk-Kick (IsaacSim, 2× A6000)

Diese Liste geht die Pipeline in der Reihenfolge des Umsetzungsplans durch. Alles, was ohne GPU geht, ist umgesetzt und auf der CPU getestet. Bei den GPU-Läufen sollte also nur noch Debugging nötig sein. Jeder Schritt sagt, worauf du achten sollst und wo du bei einem Fehler zuerst nachschaust.

Pfade als Beispiel: Bundle unter `~/soccer_bundle`, Logs unter `logs/`. `<run>` steht für den Run-Ordner, den das Training anlegt.

## 0. Umgebung prüfen (5 Minuten)

```bash
git pull && git submodule update --init --recursive
ln -s /pfad/zu/urdf_export robots/lupus/meshes        # CAD-Meshes, nicht in git
python robots/lupus/build_urdf.py && python robots/lupus/build_mjcf.py
python robots/ball/build_ball.py

# CPU-Tests der Trainingsseite (holosoma-Umgebung)
python -m pytest -q wbt_training/tests tests
# CPU-Tests Motion Matching (eigene Umgebung, numpy 2.2.6)
python -m pytest -q motion_matching/tests/test_soccer_bundle.py
```

Erwartet: alles grün bis auf `tests/test_sim2sim_launchers.py::test_sim_launcher_uses_pinned_holosoma_and_forwards_overrides` (schlägt schon auf main fehl). Die Golden-Tests in `motion_matching/tests` hängen von der Plattform ab und schlagen in der Cloud ebenfalls schon auf main fehl.

## 1. Phase 0: LUPUS in IsaacSim (Lauf-Policy als Modellcheck)

```bash
# kurzer Rauchtest: Szene baut, LUPUS steht
python -m wbt_training.train_agent exp:lupus-loco simulator:isaacsim logger:disabled --training.headless True \
    --training.num-envs 64 --algo.config.num-learning-iterations 10
# echtes Training
python -m wbt_training.train_agent exp:lupus-loco simulator:isaacsim --training.headless True
# Sim2Sim in MuJoCo (CPU reicht)
python -m wbt_training.sim2sim.lupus_sim2sim --onnx logs/lupus-loco/<run>/model_<iter>.onnx --vx 0.5 --seconds 10
```

Achten auf: LUPUS steht beim Start und kippt nicht nach vorn (Knöchel-Steifigkeit muss über etwa 50 Nm/rad liegen, Default kp 100). Läuft die Policy in IsaacSim, aber nicht in MuJoCo, dann liegt der Fehler im Modell (Massen, Armature, Kollisionen) und nicht in der Policy.

## 2. Phase 1: Referenzclips (lokal, CPU)

Die Datensätze lassen sich aus der Cloud nicht laden, deshalb lokal:

```bash
python -m reference_data.pipeline human --work work/ ...     # siehe reference_data/README.md
python -m reference_data.pipeline retarget --work work/ --jobs 8
python -m reference_data.pipeline post --work work/ --out reference_out/
python -m reference_data.pipeline summary --out reference_out/
python -m reference_data.verify reference_out/                # Exit 1, wenn ein Clip durchfällt

python -m motion_matching.build_kick_database reference_out/walk_kick --robot lupus --family kick_run
python -m motion_matching.build_kick_database reference_out/stand_kick --robot lupus --family kick_stand
```

Dazu die LUPUS-Locomotion-Datenbank aus den Lauf-Clips bauen (`motion_matching.generate_database --robot lupus`, siehe `motion_matching/README.md`).

## 3. Phase 2: Trainingsbundle bauen und abnehmen (CPU)

```bash
python -m motion_matching.build_soccer_bundle --robot lupus --out ~/soccer_bundle --jobs 16
python -m motion_matching.build_soccer_bundle check ~/soccer_bundle          # nur Abnahme erneut
# Phase 6 zusätzlich:
python -m motion_matching.build_soccer_bundle --robot lupus --out ~/soccer_bundle_p6 --phase6 --jobs 16
# ohne echte Clips nur als Rauchtest:
python -m motion_matching.build_soccer_bundle --robot lupus --out /tmp/smoke_bundle --placeholder-db --total 20
```

Achten auf: Die Abnahme meldet für jede Motion, ob der Ball im Kontakt-Frame am Kickfuß liegt (höchstens 0,25 m), ob beide Kickfüße vorkommen und ob die Anläufe verschiedene Schrittphasen abdecken. `manifest.json` enthält Zählungen und den Bericht. Im Visualizer kannst du Stichproben aus `~/soccer_bundle.raw/` ansehen.

## 4. Phase 3: Teacher

```bash
# Rauchtest: Ball wird gespawnt, Rewards laufen, keine NaNs (2-3 Minuten)
LOGGER=disabled NGPUS=1 REGISTRY=file://$HOME/soccer_bundle \
    bash wbt_training/training_runs/run_lupus_kick_teacher.sh \
    --training.num-envs 256 --algo.config.num-learning-iterations 20

# volles Training: 2 GPUs, 4096 Envs pro GPU, 20k Iterationen
REGISTRY=file://$HOME/soccer_bundle bash wbt_training/training_runs/run_lupus_kick_teacher.sh
# mehr Envs, wenn Speicher frei ist
REGISTRY=file://$HOME/soccer_bundle bash wbt_training/training_runs/run_lupus_kick_teacher.sh --training.num-envs 8192

# Abnahme Teacher mit Kick-Metriken
CHECKPOINT=logs/<run>/model_20000.pt REGISTRY=file://$HOME/soccer_bundle NUM_ENVS=1024 \
    bash wbt_training/training_runs/eval_teacher.sh \
    --kick-metrics.enabled True --kick-metrics.output-path kick_metrics_teacher.json
```

Achten auf (W&B): `ball_kick_direction` steigt, `ball_untouched` und `ball_step_on` gehen gegen 0, das Adaptive Sampling zieht Kick-Motions nach oben. Abnahme laut Plan: mehr als 90 % Kicks innerhalb ±15° (`kick_success_rate`), kein Stand-Kick vor dem Auslöser und kein Ballkontakt in Offpath-Episoden (`miskick_rate`), unter 2 % Stürze (`fall_rate`).

Häufige Fehlerstellen: Ballkörper heißt nicht `ball` (Szene in `config_values/lupus_kick.py`), Kick-Felder fehlen in der Motion-Datei (`motion_convert` behält sie jetzt, `combine_motions` nimmt auch skalare Felder wie `kick_side`), Material-DR des Roboters gibt es nur in IsaacSim.

## 5. Phase 4: Student mit Ball-Detektion

```bash
# DAgger + PPO, 2 GPUs
TEACHER_CHECKPOINT=logs/<teacher_run>/model_20000.pt REGISTRY=file://$HOME/soccer_bundle \
    bash wbt_training/training_runs/run_lupus_kick_distill.sh
# reines DAgger als Vergleich
FINETUNE=0 TEACHER_CHECKPOINT=... REGISTRY=... bash wbt_training/training_runs/run_lupus_kick_distill.sh

# ONNX (schreibt model_<iter>/policy.onnx, auch automatisch bei jedem Save)
python -m wbt_training.training_runs.export_distill_onnx \
    --training.checkpoint logs/<student_run>/model_20000.pt \
    --training.registry-name file://$HOME/soccer_bundle --training.num-envs 2

# Abnahme Student, danach Jitter-Test
python -m wbt_training.training_runs.eval_student --checkpoint logs/<student_run>/model_20000.pt \
    --training.registry-name file://$HOME/soccer_bundle --training.num-envs 1024 --training.max-eval-steps 3000 \
    --kick-metrics.enabled True --kick-metrics.output-path kick_metrics_student.json
P=--observation.groups.actor-obs.terms.command.params
python -m wbt_training.training_runs.eval_student --checkpoint ... --kick-metrics.enabled True \
    $P.commit-replace-prob 0 $P.flicker-prob 0 $P.jitter-enabled True $P.jitter-sigma-v 0.05 $P.jitter-update-hz 20
```

Abnahme: höchstens 5 Prozentpunkte unter dem Teacher, kein abgebrochener Kick bei verrauschtem Command.

## 6. Phase 5: Sim2Sim mit Ball (CPU)

```bash
ONNX=logs/<student_run>/model_20000/policy.onnx
python -m wbt_training.sim2sim.lupus_kick_sim2sim --onnx $ONNX --approach 1.5 --lateral 0.133 --v 0.5
python -m wbt_training.sim2sim.lupus_kick_sim2sim --onnx $ONNX --stand-kick --approach 0.25 --lateral 0.133 --trigger 1.0
python -m wbt_training.sim2sim.lupus_kick_sim2sim --onnx $ONNX --no-kick --approach 1.5 --lateral 0.6 --v 0.5   # Offpath
python -m wbt_training.sim2sim.lupus_kick_sim2sim --onnx $ONNX --grid --episodes 3 --jitter --out sim2sim_report.json
```

Die MuJoCo-Ballphysik dämpft jetzt lineare und Winkelgeschwindigkeit wie PhysX. Vorher rollte der Ball in MuJoCo etwa 1,7-mal so weit wie in IsaacSim. Sim2Sim-Unterschiede bei der Ballgeschwindigkeit sollten also klein sein.

## 7. Phase 6: Kickrichtung und Dribbeln

```bash
REGISTRY=file://$HOME/soccer_bundle_p6 EXP_PRESET=exp:lupus-kickdir-teacher bash wbt_training/training_runs/run_lupus_kick_teacher.sh
REGISTRY=file://$HOME/soccer_bundle_p6 EXP_PRESET=exp:lupus-dribble-teacher bash wbt_training/training_runs/run_lupus_kick_teacher.sh

# Sweep der Dribbel-Gewichte (kurze Läufe), druckt eine Befehlszeile pro Kombination
python -m wbt_training.training_runs.dribble_sweep --registry file://$HOME/soccer_bundle_p6 \
    --near-foot 0.25 0.5 1.0 --along-heading 0.5 1.0 --iterations 5000 > sweep.sh
bash sweep.sh

# Student
EXP_PRESET=exp:lupus-dribble-distill-finetune TEACHER_CHECKPOINT=logs/<dribble_teacher>/model_20000.pt \
    REGISTRY=file://$HOME/soccer_bundle_p6 bash wbt_training/training_runs/run_lupus_kick_distill.sh

python -m wbt_training.sim2sim.lupus_kick_sim2sim --onnx ... --approach 1.5 --lateral 0.133 --v 0.5 --kick-dir-deg 20
python -m wbt_training.sim2sim.lupus_kick_sim2sim --onnx ... --dribble --approach 0.25 --v 0.4
```

Dribbel-Metriken stehen im Kick-Bericht: `dribble_ball_dist_mean`, `dribble_loss_rate` (Ball weiter als 0,6 m vom Fuß), `dribble_speed_err_rms`, `dribble_lateral_speed_mean`. Ballkontakte im Dribbel-Modus zählen nicht mehr als Fehlkick. Die besten Gewichte trägst du als neuen Default in `DribbleRewardCfg` (`config_values/lupus_phase6.py`) ein.

## 8. Auf dem Roboter (später, ohne GPU)

```bash
# Ausrollen des echten Balls -> Dämpfung für ball.yaml und DR-Bereich
python -m wbt_training.deploy.calibrate ball roll1.csv roll2.csv      # Spalten t,x,y (Video-Track)
python -m wbt_training.deploy.calibrate ball rolls.csv                # oder Spalten v0,distance
# Vision gegen Maßband/Mocap -> BallVisionCfg (k, Richtungsfehler, Reichweite, Bodenprojektion)
python -m wbt_training.deploy.calibrate vision detections.csv         # true_x,true_y,det_x,det_y[,gp_x,gp_y,capture_time]
# Latenzen aus PolicyNode.stats -> latency_s_range, ctrl_delay_step_range
python -m wbt_training.deploy.calibrate latency stats.json --actuation-latency-ms 15
```

Für die Parallelkinematik am Sprunggelenk wickelst du die Motor-Firmware in `AnkleLinkageFirmware` (`wbt_training/deploy/ankle_linkage.py`). Die Policy sieht dann Pitch/Roll wie im Training. Vorher am Gestell `signs` und `offsets` der vier Motoren kalibrieren und die Hebelmaße in `robots/lupus/hardware.yaml` nachmessen. Inbetriebnahme in Stufen wie in `wbt_training/deploy/README.md`.
