# LUPUS-Robotermodell

Simulationsmodell für LUPUS, erzeugt aus dem CAD-Export (`source/robot_model.urdf`, unverändert) und den Hardwaredaten in `hardware.yaml`.

```bash
# Meshes aus dem CAD-Export verlinken (nicht im Git, ~70 MB STL)
ln -s /pfad/zu/urdf_export robots/lupus/meshes
python robots/lupus/build_urdf.py
python robots/lupus/build_mjcf.py   # braucht mujoco (aus motion_matching)
```

Ausgabe:

- `lupus.urdf`: echte Motorlimits, Halsgelenke, korrigierte Massen, vereinfachte Kollisionen, Basis auf Beckenhöhe; Meshes unter `meshes/`
- `lupus_joints.yaml`: Limits, Drehmoment, Geschwindigkeit und **Armature** pro Gelenk. URDF kennt keine Armature, deshalb kommt sie von hier in die Holosoma-Robot-Config (`dof_armature_list`) und ins MJCF.
- `lupus.xml`: MJCF für MuJoCo-Sim2Sim, aufgebaut wie Holosomas `data/robots/t1/t1_29dof.xml` (Freejoint, Armature, ein Drehmoment-Motor pro Gelenk mit ±Spitzenmoment, Boden). Ein Körper pro beweglichem Gelenk (23 + Welt), CAD-Meshes nur als Visual.

Änderungen immer in `hardware.yaml` machen und neu bauen, nie in `lupus.urdf`.

## Booster K1 als Referenz

Solange LUPUS nicht fertig ist, kommen alle Werte, die nicht aus CAD oder Datenblatt stammen, vom Booster K1 ([booster_assets](https://github.com/BoosterRobotics/booster_assets) `K1_22dof.urdf`). Jeder Wert trägt in `hardware.yaml` ein `status`-Feld (`cad`, `k1_reference`, `estimate`). So siehst du auf einen Blick, was noch ersetzt werden muss:

```bash
grep -n "k1_reference\|estimate" robots/lupus/hardware.yaml
```

| Was | Wo in `hardware.yaml` | Ändern |
| --- | --- | --- |
| Gesamtmasse (aktuell K1, 19,5 kg) | `mass_target.total` | Waagenwert eintragen; `null` = reine CAD-Masse; `method: lump` statt `scale`, wenn der Rest als ein Block (z. B. Elektronik im Torso) sitzen soll |
| Gelenkanschläge | `joints.<name>.lower/upper` | gemessene Werte eintragen, `status: measured` |
| Hals (Lage, Limits, Motor, Massen) | `neck` | Werte des echten Halses |
| Motordaten / Spannung | `motors`, `supply_voltage` | Datenblatt bei anderer Spannung |
| Sprunggelenk-Hebel | `ankle_linkage` | r, d, w aus dem CAD |
| Teilematerialien | `mass_fixes` | Dichte oder Masse pro Mesh |

## Was gegenüber dem Export geändert ist

**Motoren.** In den Meshes stecken 10× MyActuator RMD-X4-36 (Hüfte pitch/roll/yaw, Knie, Schulter pitch) und 10× RMD-X2-7 (Schulter roll/yaw, Ellenbogen, 2 pro Sprunggelenk). Daten aus dem X-Series-V4-Handbuch bei 24 V:

| Motor | Spitzenmoment | Leerlaufdrehzahl | Armature (I_Rotor · i²) | Masse |
| --- | --- | --- | --- | --- |
| RMD-X4-36 | 34 N·m | 11,6 rad/s | 0,0389 kg·m² | 0,36 kg |
| RMD-X2-7 | 7 N·m | 18,6 rad/s | 0,0135 kg·m² | 0,25 kg |

**Sprunggelenk.** Die zwei X2-7 pro Fuß treiben den Fuß über Hebel und Stangen (Parallelkinematik). Den geschlossenen Kreis kann URDF nicht abbilden. Policy und Simulation arbeiten deshalb auf `ankle_pitch`/`ankle_roll` (wie Booster T1/K1 und Unitree G1); `ankle_motor_1/2` sind jetzt `fixed`, ihre Massen hängen am Schienbein. Die Umrechnung Pitch/Roll ↔ Motorwinkel passiert auf dem Roboter. Moment, Geschwindigkeit und Armature von Pitch/Roll sind aus den Hebelmaßen in `ankle_linkage` abgeleitet (Schätzwerte, siehe offene Fragen).

**Hals.** Im Export gibt es keinen Kopf. Neu sind `head_yaw` und `head_pitch` auf der Torso-Oberkante, mit Achsabstand, Limits, Motordaten und Massen (0,3 kg + 0,7 kg) vom K1. Sie gehören nicht zum Action-Space der Policy.

**Massen.** Der Export kam auf 12,14 kg, weil mehrere Teile ohne Material exportiert wurden (Masse 1e-6). Größter Posten: das Gehäuse jedes RMD-X4-36 (10 × 0,31 kg). Fehlende Teile bekommen Masse und Trägheit aus Mesh-Volumen × Dichte; die X2-7 werden auf Datenblattmasse skaliert. Damit kommt das CAD auf 16,5 kg inklusive Kopf. Die Lücke zum K1-Referenzgewicht (19,5 kg: Kabel, Schrauben, Elektronik) schließt `mass_target`, indem alle Massen und Trägheiten mit demselben Faktor (×1,18) skaliert werden.

**Gelenklimits.** Hüfte roll und Sprunggelenk stammen aus dem CAD. Hüfte pitch/yaw, Knie, Schulter pitch/roll und Ellenbogen kommen vom K1 (Schulter roll um die 1,3 rad verschoben, weil der K1 in der Nullpose die Arme anhebt). Schulter yaw hat der K1 nicht, sie bleibt geschätzt (±1,57). Vorzeichen per Vorwärtskinematik geprüft: Hüfte/Schulter pitch negativ = nach vorn, Knie positiv = beugen, Ellenbogen negativ = beugen, Sprunggelenk pitch positiv = Spitze runter.

**Basisrahmen.** Im CAD-Export liegt `base_link` auf dem Boden zwischen den Füßen. Holosoma, Motion Matching und die Retargeter erwarten die Wurzel im Becken (wie G1 `pelvis`, K1 `Trunk`). `base_frame.origin: hip_pitch_midpoint` schiebt `base_link` deshalb auf die Mitte der beiden Hüft-Pitch-Achsen (0,5745 m über der Sohle). Alle `*_base`-Koordinaten in `hardware.yaml` bleiben im CAD-Rahmen.

**Kollisionen.** Statt 187 CAD-Meshes gibt es jetzt eine Form pro starrem Segment (`collision` in `hardware.yaml`): Die Füße sind Boxen in Größe der Sohlenplatte (`footlow`, 0,30 × 0,11 × 0,03 m), weil sie Boden- und Ballkontakt tragen; der Rumpf ist eine Box; alle anderen Segmente sind Zylinder entlang ihrer Hauptachse (im MJCF Kapseln gleicher Länge), deren Radius 90 % der Mesh-Punkte umschließt. Hals und Kopf behalten ihre Boxen. Im MJCF kollidiert der Roboter nicht mit sich selbst, nur mit Boden und Objekten (Ball). `collision.mode: mesh` stellt die CAD-Meshes wieder her.

**MJCF.** `build_mjcf.py` verschmilzt die festen Links selbst (Trägheiten per Steiner), weil MuJocos `fusestatic` die Schwerpunkte bei diesem URDF falsch setzt (MuJoCo 3.8: Rumpf-Schwerpunkt 26 cm vor dem Becken). Der Spawn liegt so, dass die Sohlen den Boden berühren.

**Fußkontaktpunkte.** `left/right_foot_contact_point` sitzen in der Sohlenmitte (fest, `dont_collapse`, 1 g), wie beim T1. Holosoma misst daran Fußhöhe und Schwungphase.

**Gelenkreihenfolge.** Im MJCF und in der Holosoma-Config: linkes Bein, rechtes Bein, linker Arm, rechter Arm, Hals. Die Policy-Gelenke stehen damit vorn, der Hals am Ende.

## Holosoma-Config und Test-Locomotion

Die Robot-Config `lupus_22dof` und das Experiment `exp:lupus-loco` liegen in `wbt_training/config_values/lupus.py`. Limits, Drehmomente, Geschwindigkeiten und Armature liest sie aus `lupus_joints.yaml`, es gibt also keine doppelten Zahlen. `exp:lupus-loco` ist die Validierung aus Phase 0: eine normale Velocity-Tracking-Policy nach dem T1-Locomotion-Rezept, Command (v, ω) ohne Seitschritte.

```bash
# Training auf den A6000 (IsaacSim wandelt lupus.urdf beim Start in USD um)
python -m wbt_training.train_agent exp:lupus-loco simulator:isaacsim --training.headless True
# Schnelltest ohne GPU (1 Env, MuJoCo)
python -m wbt_training.train_agent exp:lupus-loco simulator:mujoco logger:disabled --training.num-envs 1 --algo.config.num-learning-iterations 2
```

- **Hals außerhalb des Action-Space:** Holosoma koppelt den Action-Vektor an alle Gelenke. `PassiveNeckJointPositionActionTerm` ersetzt deshalb die beiden Hals-Ausgaben vor dem PD-Regler durch ein Hals-Ziel (Default-Pose, später der Kopf-Scan-Generator in `passive_targets`). Auf dem Roboter werden die Hals-Ausgaben ignoriert.
- **Default-Pose:** Hüfte pitch −0,2, Knie 0,4, Sprunggelenk −0,2 (Fuß flach), Ellenbogen −0,3; Basis auf 0,573 m.
- **PD-Gains (Startwerte):** Beine kp 100 / kd 3, Arme 20 / 0,5, Hals 5 / 0,5. Action-Scale 0,25 wie im T1-Rezept.
- **Befund:** Nur mit PD (ohne Policy) hält LUPUS die gebeugte Default-Pose ca. 1,5 bis 3 s und kippt dann langsam. Mit kp 60 am Sprunggelenk und der Randomisierung (+3 kg Rumpf, −10 % kp) ging es schneller. Das Gleichgewicht ist Aufgabe der Policy. Auffällig ist aber, dass das Sprunggelenk-Pitch mit den geschätzten Hebeln nur 14 N·m Spitzenmoment hat; die echten Hebelmaße (`ankle_linkage`) sind deshalb wichtig.

## Geprüft

- `lupus.xml` in MuJoCo: 19,5 kg, Schwerpunkt 0,3 cm vor den Sprunggelenken und mittig. Mit PD-Haltung (kp 100 an Beinen) steht LUPUS 4 s in der Nullpose, nur die Füße berühren den Boden; eine Kniebeuge (Knie 1,0 rad) hält mit ca. 13 N·m pro Knie.
- **Hinweis für die PD-Gains:** Die Steifigkeit im Sprunggelenk muss über m·g·h ≈ 50 N·m/rad pro Fuß liegen, sonst kippt LUPUS als umgekehrtes Pendel langsam nach vorn (mit kp 40 am Sprunggelenk passiert das nach etwa 2 s).
- `tests/test_lupus_model.py`: keine Platzhalter-Limits mehr, URDF und Gelenktabelle passen zusammen, Trägheitsberechnung stimmt am Einheitswürfel, nur primitive Kollisionen mit echter Sohlengröße, Basis auf Hüft-Pitch-Mitte, MJCF-Masse/Schwerpunkt und 2 s Stehen (nur mit Meshes).
- `wbt_training/tests/test_lupus_config.py`: `exp:lupus-loco` wählbar, Listen pro Gelenk vollständig, Default-Pose in den Limits, Namen und Reihenfolge = `lupus.xml`, Basishöhe stimmt, Holosoma-Env mit MuJoCo-Backend steht 1 s und ignoriert Hals-Ausgaben der Policy. Ein PPO-Durchlauf (2 Iterationen, 1 Env, CPU) läuft durch.
- Motion Matching: `RobotSpec` `lupus` (`motion_matching/tests/test_robot_spec.py`): Bone-Modell = URDF-Vorwärtskinematik, Spiegelregeln spiegeln die Pose exakt, Limits links/rechts spiegelsymmetrisch.

## Offene Fragen

1. **Waage:** wenn LUPUS fertig ist, Gewicht in `mass_target.total` eintragen.
2. **Versorgungsspannung:** Die Drehzahlen gelten für 24 V. Laufen die Motoren an einer höheren Spannung, steigen die Geschwindigkeitslimits.
3. **Sprunggelenk-Hebel:** Hebellänge r, Abstand der Stangenanlenkung zur Pitch-Achse d und seitlicher Abstand w (aktuell 40/40/27 mm geschätzt).
4. **Gelenkanschläge** der Gelenke mit `status: k1_reference` oder `estimate` am Roboter messen.
5. **Hals:** Motortyp, Lage der Achsen und Kopfmasse (Kamera).
6. **Materialien** der Teile ohne CAD-Material (in `mass_fixes` als Annahme markiert).
