# LUPUS-Robotermodell

Simulationsmodell für LUPUS, erzeugt aus dem CAD-Export (`source/robot_model.urdf`, unverändert) und den Hardwaredaten in `hardware.yaml`.

```bash
# Meshes aus dem CAD-Export verlinken (nicht im Git, ~70 MB STL)
ln -s /pfad/zu/urdf_export robots/lupus/meshes
python robots/lupus/build_urdf.py
```

Ausgabe:

- `lupus.urdf`: echte Motorlimits, Halsgelenke, korrigierte Massen; Meshes unter `meshes/`
- `lupus_joints.yaml`: Limits, Drehmoment, Geschwindigkeit und **Armature** pro Gelenk. URDF kennt keine Armature, deshalb kommt sie von hier in die Holosoma-Robot-Config (`dof_armature_list`) und ins MJCF.

Änderungen immer in `hardware.yaml` machen und neu bauen, nie in `lupus.urdf`.

## Was gegenüber dem Export geändert ist

**Motoren.** In den Meshes stecken 10× MyActuator RMD-X4-36 (Hüfte pitch/roll/yaw, Knie, Schulter pitch) und 10× RMD-X2-7 (Schulter roll/yaw, Ellenbogen, 2 pro Sprunggelenk). Daten aus dem X-Series-V4-Handbuch bei 24 V:

| Motor | Spitzenmoment | Leerlaufdrehzahl | Armature (I_Rotor · i²) | Masse |
| --- | --- | --- | --- | --- |
| RMD-X4-36 | 34 N·m | 11,6 rad/s | 0,0389 kg·m² | 0,36 kg |
| RMD-X2-7 | 7 N·m | 18,6 rad/s | 0,0135 kg·m² | 0,25 kg |

**Sprunggelenk.** Die zwei X2-7 pro Fuß treiben den Fuß über Hebel und Stangen (Parallelkinematik). Den geschlossenen Kreis kann URDF nicht abbilden. Policy und Simulation arbeiten deshalb auf `ankle_pitch`/`ankle_roll` (wie Booster T1/K1 und Unitree G1); `ankle_motor_1/2` sind jetzt `fixed`, ihre Massen hängen am Schienbein. Die Umrechnung Pitch/Roll ↔ Motorwinkel passiert auf dem Roboter. Moment, Geschwindigkeit und Armature von Pitch/Roll sind aus den Hebelmaßen in `ankle_linkage` abgeleitet (Schätzwerte, siehe offene Fragen).

**Hals.** Im Export gibt es keinen Kopf. Neu sind `head_yaw` und `head_pitch` auf der Torso-Oberkante mit Platzhalter-Massen (0,25 kg + 0,40 kg). Sie gehören nicht zum Action-Space der Policy.

**Massen.** Der Export kam auf 12,14 kg, weil mehrere Teile ohne Material exportiert wurden (Masse 1e-6). Größter Posten: das Gehäuse jedes RMD-X4-36 (10 × 0,31 kg). Fehlende Teile bekommen Masse und Trägheit aus Mesh-Volumen × Dichte; die X2-7 werden auf Datenblattmasse skaliert. Ergebnis: **16,1 kg** inklusive 0,65 kg Kopf-Platzhalter.

**Gelenklimits.** Hüfte roll und Sprunggelenk stammen aus dem CAD. Alle anderen waren ±π und sind jetzt vorläufige Werte in K1/T1-Größenordnung (`status: estimate`). Vorzeichen per Vorwärtskinematik geprüft: Hüfte/Schulter pitch negativ = nach vorn, Knie positiv = beugen, Ellenbogen negativ = beugen, Sprunggelenk pitch positiv = Spitze runter.

## Geprüft

- MuJoCo 3.x lädt `lupus.urdf`; mit freier Basis und PD-Haltung steht LUPUS 4 s stabil in der Nullpose und hält eine Kniebeuge (Knie 1,2 rad, ca. 8,6 N·m pro Knie).
- `tests/test_lupus_model.py`: keine Platzhalter-Limits mehr, URDF und Gelenktabelle passen zusammen, Trägheitsberechnung stimmt am Einheitswürfel.

## Offene Fragen

1. **Waage:** Gewicht von LUPUS (mit Akkus, ohne Kopf bzw. mit Kopf) in `measured_mass` eintragen. Das Skript zeigt dann die Differenz. Kabel, Schrauben und Elektronik fehlen im CAD noch.
2. **Versorgungsspannung:** Die Drehzahlen gelten für 24 V. Laufen die Motoren an einer höheren Spannung, steigen die Geschwindigkeitslimits.
3. **Sprunggelenk-Hebel:** Hebellänge r, Abstand der Stangenanlenkung zur Pitch-Achse d und seitlicher Abstand w (aktuell 40/40/27 mm geschätzt).
4. **Gelenkanschläge** der Gelenke mit `status: estimate` am Roboter messen.
5. **Hals:** Motortyp, Lage der Achsen und Kopfmasse (Kamera).
6. **Materialien** der Teile ohne CAD-Material (in `mass_fixes` als Annahme markiert).
