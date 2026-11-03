# LUPUS: Evaluation, Sim2Sim und Deployment (Phase 5)

Bevor LUPUS auf den Platz geht, läuft die exportierte Policy (`policy.onnx` aus Phase 4) in MuJoCo mit demselben Ball-Rauschmodell. Der Sim2Sim nutzt dabei genau den Code, der später auf dem Jetson läuft.

## 1. Evaluation in IsaacSim (`eval_student.py`, `eval_teacher.py`)

```bash
python -m wbt_training.training_runs.eval_student \
    --checkpoint /pfad/student/model_20000.pt \
    --training.registry-name file:///pfad/lupus_kick_eval_bundle \
    --training.num-envs 1024 --training.max-eval-steps 3000 \
    --kick-metrics.enabled True --kick-metrics.output-path kick_metrics.json
```

`--kick-metrics.*` funktioniert genauso mit `eval_teacher.py` (Abnahme Teacher, Phase 3). Der Bericht enthält Zusammenfassung, Raster und jede Episode.

| Metrik | Definition |
| --- | --- |
| `kick_success_rate` | Anteil der Kick-Episoden mit Kontakt im Kick, Ballgeschwindigkeit ≥ 0,5 m/s und Richtungsfehler ≤ 15° |
| `dir_err_deg_mean(_abs)` | Winkel zwischen Ballrichtung (bei Spitzengeschwindigkeit, 0,5 s nach Kontakt) und Blickrichtung beim Kontakt (Phase 6: + `kick_dir`) |
| `ball_speed_mean` | Spitzengeschwindigkeit des Balls nach dem Kick |
| `miskick_rate` | Episoden mit Ballkontakt, solange kein Kick läuft (Stand vor dem Auslöser, Offpath, ohne Ball) |
| `fall_rate`, `termination_rate` | Basis unter 0,3 m oder > 60° gekippt; Episodenende ohne Timeout |
| `v_err_rms`, `w_err_rms` | Tracking-Fehler von v und ω beim Laufen |

**Raster wie PHP Tabelle I** (`KickEvalGrid`): Anlaufdistanz (0,5/1/2/3 m) × seitlicher Ballversatz zum näheren Fuß (0/0,05/0,10/0,25/0,5 m) × Geschwindigkeit (0 = Stand-Kick, 0,3/0,6/0,9 m/s), je 100 Episoden. Die Episoden werden nach ihren gemessenen Startwerten einsortiert, das Raster funktioniert also mit jedem Motion-Set. Für volle Zellen ein eigenes Eval-Bundle erzeugen; `KickEvalGrid().scenario_specs()` liefert die Spezifikation pro Zelle für den Szenario-Generator aus Phase 2.

**Jitter-Test**: realistisches Command-Rauschen des Behaviors auf die Student-Observation geben, Commit-Ersetzung und Flackern dabei aus:

```bash
P=--observation.groups.actor-obs.terms.command.params
python -m wbt_training.training_runs.eval_student --checkpoint ... --kick-metrics.enabled True \
    $P.commit-replace-prob 0 $P.flicker-prob 0 \
    $P.jitter-enabled True $P.jitter-sigma-v 0.05 $P.jitter-update-hz 20
```

Modell (`utils/command_jitter.py`): OU-Rauschen auf v und ω, Behavior-Rate mit Hold, verspätete Nachrichten, `kick_enable`-Glitches. Mit `"log_file": "behavior_log.npz"` (Arrays `v`, `omega` mit 50 Hz) werden stattdessen die Residuen eines echten Behavior-Logs abgespielt.

## 2. MuJoCo-Sim2Sim mit Ball (CPU)

```bash
# Kick aus dem Lauf, Ball 1,5 m voraus vor dem linken Fuß
python -m wbt_training.sim2sim.lupus_kick_sim2sim --onnx model_20000/policy.onnx --approach 1.5 --lateral 0.133 --v 0.5
# Freistoß: Ball in der Kick-Zone, kick_enable steigt nach 1 s
python -m wbt_training.sim2sim.lupus_kick_sim2sim --onnx ... --stand-kick --approach 0.25 --lateral 0.133 --trigger 1.0
# ganzes Raster, 3 Episoden pro Zelle, mit Command-Jitter
python -m wbt_training.sim2sim.lupus_kick_sim2sim --onnx ... --grid --episodes 3 --jitter --out sim2sim_report.json
```

Aufbau: `lupus.xml` + freier Ball aus `robots/ball/ball.yaml`, PD mit 200 Hz in 50-Hz-Schritten, Ballrollen über lineare Dämpfung. Die Vision ist simuliert wie im Training (Sichtfeld der Kopfkamera, Tiefen-/Richtungsrauschen, 25–40 Hz, Latenz, Dropouts), die Frames kommen erst nach ihrer Latenz beim Policy-Knoten an. Der Kopf scannt links/rechts und verfolgt den Ball, sobald ein Kick ansteht und der Ball näher als 1 m ist. Braucht die CAD-Meshes unter `robots/lupus/meshes` (siehe `robots/lupus/README.md`).

## 3. Jetson Orin: Policy-Prozess

`wbt_training/deploy/` braucht zur Laufzeit nur `numpy` und `onnxruntime` (oder `onnxruntime-gpu`/TensorRT-EP über `providers`). Die Policy läuft in einem eigenen Prozess mit fester Taktung, getrennt von YOLO:

```python
from wbt_training.deploy.policy_io import StudentPolicy
from wbt_training.deploy.runner import PolicyNode, BehaviorCommand

policy = StudentPolicy("policy.onnx")                     # liest Gelenkreihenfolge, Gains, Obs-Layout aus den Metadaten
node = PolicyNode(policy, firmware=MyFirmware(), behavior=MyBehavior())
vision_sub.on_message(lambda m: node.on_vision_frame(m.capture_time, m.found, m.x, m.y))
stats = node.run()                                         # 50 Hz, absolute Deadlines, zählt Überläufe
```

| Schnittstelle | liefert | Hinweis |
| --- | --- | --- |
| `Firmware.read_state()` | Gelenkwinkel/-geschwindigkeiten (Policy-Reihenfolge), IMU-Quaternion (wxyz) und Gyro im Basisframe | Halswinkel gehören dazu, sie sind Observation |
| `Firmware.send_targets(q, kp, kd)` | Positionsziele aller 22 Gelenke mit den Trainings-Gains | Hals: Ziel vom Behavior |
| Vision → `on_vision_frame(capture_time, found, x, y)` | jedes verarbeitete Kamerabild, auch ohne Ball (`found=False`) | `x, y` im Frame des Trainings (Default: Heading-Frame der Basis), Zeitstempel = Aufnahmezeit |
| `Behavior.command()` | `v`, `omega`, `kick_enable` (Phase 6: `kick_dir`, `mode`), `neck` | |

`node.stats.as_dict()` liefert Schleifenperiode, Rechenzeit und Kamera-bis-Policy-Latenz (Mittel, p50, p99, max). Mit diesen Messungen die Randomisierung anpassen: `latency_s_range` und `update_hz_range` in `BallVisionCfg`, Action-Delay (`ctrl_delay_step_range`) im Teacher-Preset.

## 4. Inbetriebnahme in Stufen

1. am Gestell: `PolicyNode` mit Command 0, Gelenkziele und Gains prüfen, Latenzen messen
2. nur Laufen (`kick_enable` = 0, kein Ball)
3. Kick aus dem Stand (Freistoß): Ball in die Kick-Zone legen, `kick_enable` setzen
4. ruhender Ball im Lauf geradeaus
5. Anlauf mit Kurve (ω ≠ 0)
6. Ball neben dem Weg (darf nicht berührt werden)
7. Command und Kopf direkt aus dem Behavior

## Offen

- Transport zur Firmware (ROS 2, seriell) und Vision-Nachrichtenformat festlegen, dann `Firmware`/Vision-Adapter schreiben
- Latenzen auf dem Roboter messen und Randomisierung anpassen
- GPU-Eval mit trainierten Checkpoints
