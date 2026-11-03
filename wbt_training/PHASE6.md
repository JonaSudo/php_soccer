# LUPUS Phase 6: Kickrichtung und Dribbeln

Beide Erweiterungen bringen nur neue Inputs, Rewards und Szenarien, das Rezept von Phase 3/4 bleibt. Code: [`config_values/lupus_phase6.py`](config_values/lupus_phase6.py), Szenarien in [`motion_matching/scenarios/soccer.py`](../motion_matching/motion_matching/scenarios/soccer.py) (`PHASE6_SCENARIOS`).

| Preset | Basis | neu |
| --- | --- | --- |
| `lupus_kickdir_teacher` | `lupus_kick_teacher` | Input `kick_dir`, `ball_kick_direction` misst gegen Blickrichtung beim Kontakt + `kick_dir` |
| `lupus_kickdir_distill_finetune` | `lupus_kick_distill_finetune` | Student-Command `[v, ω, kick_enable, kick_dir]` |
| `lupus_dribble_teacher` | wie `lupus_kickdir_teacher` | zusätzlich Modus als One-Hot (Ignorieren/Kick/Dribbeln) und die Dribbel-Rewards |
| `lupus_dribble_distill_finetune` | wie `lupus_kickdir_distill_finetune` | Student-Command `[v, ω, kick_dir, Modus-One-Hot]` |

Die Teacher-Inputs werden breiter, deshalb starten die Phase-6-Teacher von vorn (kein Laden der Phase-3-Gewichte).

## Kickrichtung

- `kick_dir` in rad relativ zur Laufrichtung, Bereich ±30° (`KICK_DIR_MAX`), Standard 0. Er steht pro Frame in der Motion-npz und ist pro Motion konstant.
- `walk_kick_dir` und `stand_kick_dir` sind `walk_kick`/`stand_kick` mit zufälligem `kick_dir` pro Motion. Die Referenz bleibt der gerade Kick, neue Clips sind nicht nötig; über PPO lernt die Policy, den Fuß anders zu stellen. Seitkicks mit großem Winkel bräuchten eigene Clips.
- Die Phase-2-Szenarien bleiben unverändert (`kick_dir_max=0`, eigener Zufallsstrom für die Richtung).
- Teacher: `ball_kick_direction_dir` ersetzt `ball_kick_direction`, mit `kick_dir` = 0 ist es derselbe Reward.
- Student: Die Commit-Phase ersetzt `kick_dir` mit, ein begonnener Kick wird also auch bei springender Richtung zu Ende geführt. Auf `kick_dir` liegt Rauschen mit σ = 0,02 rad.
- Metriken: `dir_err_deg` misst in Eval (`--kick-metrics.*`) und Sim2Sim gegen Blickrichtung + `kick_dir`.

## Dribbel-Modus

- `kick_mode`: 0 Ignorieren, 1 Kick, 2 Dribbeln. Ohne `kick_mode` in der npz gilt `kick_enable` (0 bzw. 1).
- Szenario `walk_dribble`: Stand, langsames Laufen (v ≤ 0,6 m/s, |ω| ≤ 0,3 rad/s), Stand. Der Ball liegt 0,15–0,30 m vor dem Roboter auf dem Weg, wo das Laufen beginnt. `kick_enable` ist 0 und `kick_mode` 2.
- Rewards (Startgewichte geschätzt, je 0,5):
  - `ball_near_foot`: exp(−(d/0,15 m)²), wobei d der Abstand vom Ball zum näheren Fuß ist
  - `ball_along_heading`: Ballgeschwindigkeit in Blickrichtung nahe am kommandierten v; Querrollen kostet
- `ball_step_on` (Draufsteigen) bleibt aktiv. `ball_untouched` straft im Dribbel-Modus nicht mehr.
- Damit ist „Ball auf dem Weg, Kick aus“ ein trainierter Zustand: Das Behavior schaltet dann auf Dribbeln.

## Daten erzeugen und trainieren

```bash
# zusätzliche Motions (Phase 2 + diese)
python -m motion_matching.run --mode generate --robot lupus --scenario walk_kick_dir
python -m motion_matching.run --mode generate --robot lupus --scenario stand_kick_dir
python -m motion_matching.run --mode generate --robot lupus --scenario walk_dribble

# Teacher (Bundle: Phase-2-Set + obige Motions; für kickdir ohne walk_dribble)
python -m wbt_training.train_agent exp:lupus-dribble-teacher simulator:isaacsim \
    --training.registry-name file:///pfad/lupus_phase6_bundle

# Student
EXP_PRESET=exp:lupus-dribble-distill-finetune TEACHER_CHECKPOINT=/pfad/teacher/model_20000.pt \
REGISTRY=file:///pfad/lupus_phase6_bundle bash wbt_training/training_runs/run_lupus_kick_distill.sh

# Sim2Sim
python -m wbt_training.sim2sim.lupus_kick_sim2sim --onnx .../policy.onnx --approach 1.5 --lateral 0.133 --v 0.5 --kick-dir-deg 20
python -m wbt_training.sim2sim.lupus_kick_sim2sim --onnx .../policy.onnx --dribble --approach 0.25 --v 0.4
```

Auf dem Jetson schickt das Behavior `BehaviorCommand(kick_dir=..., mode=...)`. `command_vector` baut daraus das Layout, das in den ONNX-Metadaten steht (`command_obs_cfg`).

## Offen

- Gewichte der Dribbel-Rewards und `sigma` nach den ersten GPU-Läufen einstellen
- Dribbel-Metriken (Ballabstand, Ballverlust) in `kick_metrics.py` ergänzen, sobald ein trainierter Dribbel-Teacher da ist
- Seitkick (großer `kick_dir`) mit eigenen Clips, z. B. OmniContact `case2_r2l`/`case3_l2r`
