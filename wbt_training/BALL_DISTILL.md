# LUPUS-Student mit Ball-Detektion (Phase 4)

Der Student bekommt nur, was LUPUS auf dem Feld hat: Propriozeption, den Command des Behaviors und die verrauschte Ballposition aus der Vision. Depth-Kamera und Depth-Backbone entfallen, exportiert wird **ein** ONNX. Der Rest ist das PHP-Rezept (DAgger + PPO, `RelaxTerminationThresholds`, PPO-Gate, Adaptive Sampling aus).

## Start

Voraussetzung: ein trainierter Teacher (`exp:lupus-kick-teacher`, Phase 3) und das Motion-Bundle, mit dem er trainiert wurde.

```bash
# DAgger + PPO (Hauptrezept), 2x A6000
TEACHER_CHECKPOINT=/pfad/teacher/model_20000.pt \
REGISTRY=file:///pfad/lupus_kick_bundle \
NGPUS=2 \
bash wbt_training/training_runs/run_lupus_kick_distill.sh

# reines DAgger (kleineres Student-MLP, 10k Iterationen)
FINETUNE=0 TEACHER_CHECKPOINT=... REGISTRY=... bash wbt_training/training_runs/run_lupus_kick_distill.sh

# Export eines Checkpoints (schreibt model_<iter>/policy.onnx)
python -m wbt_training.training_runs.export_distill_onnx \
    --training.checkpoint /pfad/student/model_20000.pt \
    --training.registry-name file:///pfad/lupus_kick_bundle --training.num-envs 2
```

Beim Training schreibt jeder Checkpoint-Save automatisch `model_<iter>/policy.onnx` daneben.

## Was der Student sieht (`actor_obs`, 8 Frames Historie)

Die Terms werden alphabetisch sortiert verkettet, jeder Term mit seinen 8 Frames hintereinander (ältester zuerst, nach Reset mit Nullen aufgefüllt). Die genaue Spaltenaufteilung steht als `obs_layout` in den ONNX-Metadaten.

| Term | Breite | Inhalt |
| --- | --- | --- |
| `actions` | 22 | letzte Aktion (Hals-Einträge werden ignoriert) |
| `ball_detection` | 4 | `x, y` (Heading-Frame, m), `visible`, Zeit seit letzter Sichtung (s, max. 2) |
| `base_ang_vel` | 3 | Winkelgeschwindigkeit der Basis |
| `command` | 3 | `v`, `ω`, `kick_enable` (Phase 6: + `kick_dir`, Modus als One-Hot) |
| `dof_pos`, `dof_vel` | je 22 | alle Gelenke inkl. Hals: Halswinkel ist reine Information, die Policy steuert ihn nicht |
| `robot_anchor_projected_gravity` | 3 | projizierte Gravitation |

`teacher_obs` (Teacher-Actor-Obs aus `lupus_kick.py` + `which_motion`) und `critic_obs` behalten den wahren Ballzustand. Die Gruppe `depth_camera` ist nur ein Platzhalter mit einer Nullspalte, weil Holosomas `DistillationPPO` sie verlangt; `NoDepthBackbone` macht daraus einen Latent der Breite 0.

## Rauschmodell der Vision (`wbt_training/utils/ball_detection.py`)

Alle Werte stehen als Defaults in `BallVisionCfg`; jedes Feld lässt sich zusätzlich über die `params` des Terms `ball_detection` (`ball_distillation.ball_detection_term`) überschreiben.

| Größe | Startwert | Herkunft | Messen |
| --- | --- | --- | --- |
| Kamera im `head_link` | (0,06; 0; 0,08) m, keine Zusatzneigung | K1-Kopf | am echten Kopf |
| Sichtfeld | 90° × 65°, 0,1–6 m | geschätzt | Kameradatenblatt, YOLO-Reichweite |
| Tiefenfehler | σ = 0,05 · Distanz, Bias ±3 % pro Episode | geschätzt | gegen Maßband/Mocap |
| Richtungsfehler | 1° | geschätzt | gegen Maßband/Mocap |
| Bodenprojektion nah am Ball | aus (`near_ground_projection_range` = 0) | – | prüfen, ob genauer als das Depth-Netz |
| Detektionsrate | 25–40 Hz pro Episode, Policy 50 Hz | Plan | auf dem Roboter |
| Latenz | 40–100 ms pro Episode | geschätzt | Kamera bis Ballposition |
| Dropouts | 5–20 % pro Episode | Plan | Logs der Vision |
| Ausgabe-Frame | `heading` (gravitationsausgerichtet, Basis-Yaw) | offene Frage im Plan | wie publiziert der Vision-Node? |

Zwischen zwei Detektionen bleibt der letzte Wert stehen; `visible` gibt an, ob das neueste Kamerabild den Ball enthielt. Vor der ersten Sichtung einer Episode sind `x = y = 0` und die Zeit steht auf dem Maximum. Kurz vor dem Kontakt liegt der Ball oft unter dem Bild oder der Kopf-Scan schaut weg; die Policy muss den Kick dann aus der Historie timen.

## Command-Robustheit (`config_values/kick_command_obs.py`)

- kleines Rauschen auf `v` (σ 0,03 m/s) und `ω` (σ 0,05 rad/s), nur in der Observation
- Commit-Phase: Ab `kick_phase` 1 bis Kick-Ende wird bei 50 % der Kicks ab einem zufälligen Zeitpunkt (0–25 Schritte) der beobachtete Command samt `kick_enable` zufällig ersetzt, die Referenz kickt weiter. Die Policy lernt so, einen begonnenen Kick nicht abzubrechen.
- Flackern: Steht `kick_enable` in der Referenz auf 0, springt es in der Observation mit 1 % pro Schritt für 1–2 Schritte auf 1 (mindestens 10 Schritte Abstand). Ein einzelner Fehlwert des Behaviors löst damit keinen Freistoß aus.
- Wertebereiche für zufällige Commands: `v` 0–1 m/s, `|ω|` ≤ 1 rad/s (offene Frage: v_min, v_max, ω_max)

## Offene Punkte

- Rauschparameter und Latenz messen und in `BallVisionCfg` eintragen
- Frame der echten Vision-Ausgabe klären (Torso oder Boden/Heading)
- Command-Bereiche festlegen, sobald bekannt ist, was LUPUS schafft
- GPU-Training und Abnahme (Student höchstens 5 Prozentpunkte unter dem Teacher, kein Kickabbruch bei verrauschtem Command): mit `eval_student.py --kick-metrics.enabled True` und dem Jitter-Test, siehe [deploy/README.md](deploy/README.md)
