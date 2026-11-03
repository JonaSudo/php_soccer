# Phase 1: Referenzdaten für LUPUS

Dieses Paket macht aus Mocap-Clips (Gehen, Schüsse aus dem Anlauf, Schüsse aus dem Stand)
LUPUS-Referenzclips für Motion Matching (Phase 2) und den Teacher (Phase 3).
Alles läuft auf der CPU.

```
Quelle (BVH / OmniContact / AMASS)
  └─ human_clip      neutrales Menschenformat (Gelenkpositionen z-up, m, optional Ball)
      └─ retarget_lupus   holosoma InteractionMeshRetargeter auf LUPUS (+ Ball für Schüsse)
          └─ postprocess  Boden, Zeitskalierung 0.6–0.8x, Spiegeln, Machbarkeit, Annotation, Export
              └─ verify    Abnahme: keine Bodendurchdringung, Fuß trifft Ball im Kontaktframe
```

## Module

| Datei | Zweck |
|---|---|
| `lupus_retarget.yaml` | Retarget-Konfiguration: Gelenkzuordnung Mensch → LUPUS-Körper, Marker, Ballgrößen, Solver-Parameter |
| `retarget_model.py` | LUPUS-MJCF ohne Meshes, mit Markerkörpern (Sohlenecken, Zehen, Hände) und optional Ball |
| `human_clip.py` | Loader: BVH (LAFAN1, OmniContact-Rohdaten), OmniContact-npz (Ballpose), AMASS (holosoma-Prep), InterMimic |
| `retarget_lupus.py` | Treibt den holosoma-Retargeter direkt an (Submodul bleibt unverändert) |
| `lupus_kin.py` | LUPUS-Kinematik, Gelenkreihenfolgen PHP (URDF) ↔ MuJoCo, Fußboxen |
| `clip_ops.py` | `LupusClip`: PHP-qpos, Zeitskalierung (PCHIP + Slerp), Spiegeln, Bodenausrichtung |
| `feasibility.py` | Gelenkgrenzen, Geschwindigkeiten, Momente per Inverser Dynamik (Kontaktkräfte an den Sohlenecken), Bodendurchdringung |
| `annotate.py` | Schuss-Annotation: Gruppe, Schussfuß, entry_start, skill_start, contact_frame, skill_end, Ballposition |
| `postprocess.py` | Export pro Variante (`<id>[_mirror]_<speed>x.npz/.json`) und `*.kick.json` für Phase 2 |
| `verify.py` | Abnahmetest der exportierten Clips |
| `pipeline.py` + `catalog.yaml` | Batch-Lauf über alle Clips des Katalogs |
| `gmr_lupus.py` | GMR-Konfiguration für LUPUS, aus den G1-Konfigurationen von GMR abgeleitet (Alternative fürs Gehen) |
| `synthetic.py` | Synthetische LUPUS-Clips für die Tests |

## Ausgabeformat

- `<out>/<gruppe>/<name>.npz`: `qpos` (T, 7+22) float32 = `[root_quat_wxyz, root_pos, dofs]`,
  DoFs in der Reihenfolge von `LUPUS_SPEC.joint_names` (URDF-Reihenfolge, nicht MJCF), dazu `fps`,
  bei Schüssen `object_pos_w` (Ballmitte pro Frame), `ball_radius`, `contact_frame`.
- `<name>.json`: Herkunft, Zeitfaktor, Machbarkeitsbericht, Annotation, `accepted`, Gründe, Ergebnis von `verify`.
- `<name>.kick.json` (nur Schüsse, nur ungespiegelt, nur abgenommen): Format von
  `motion_matching/io/kick_io.py`. `kind` ist `run` (Anlauf) oder `stand`, `speed_up` ist 1, weil der Clip
  schon zeitskaliert ist, der Ball steht als `ball_pos` im Weltframe des Clips.
  `build_kick_database` erzeugt die gespiegelte Kopie selbst.
- `manifest.json`: alle Clips mit Status.

Direkt für Phase 2 nutzbar:

```bash
python -m motion_matching.build_kick_database reference_out/walk_kick --robot lupus --family kick_run
python -m motion_matching.build_kick_database reference_out/stand_kick --robot lupus --family kick_stand
```

## Umgebungen

Zwei Umgebungen, weil der holosoma-Retargeter andere Versionen braucht:

1. **Retargeting** (holosoma_retargeting): Python 3.11, `numpy 2.3`, `torch`, `cvxpy` mit `clarabel`,
   `trimesh`, `pyyaml`, `scipy` und **`mujoco==3.8.1`**. Mit neueren MuJoCo-Versionen (getestet 3.17)
   vergleicht holosoma den Gelenktyp falsch (`np.int32 in (mjJNT_HINGE,)` ist False). Dann bleiben alle
   Gelenke still bei 0, ohne Fehlermeldung. Der Test `test_holosoma_retarget_demo` wird übersprungen, wenn
   die Version nicht passt.
2. **Rest** (motion_matching-Umgebung): `pip install -e motion_matching`, dazu `mujoco`, `scipy`, `pyyaml`.

## Daten besorgen (lokal, nicht aus der Cloud erreichbar)

| Datensatz | Wofür | Bezug | Lizenz |
|---|---|---|---|
| LAFAN1 | Gehen, Kurven, Tempo | github.com/ubisoft/ubisoft-laforge-animation-dataset (`lafan1.zip`, Git LFS) | CC BY-NC-ND 4.0: nur intern nutzen, keine abgeleiteten Daten veröffentlichen |
| OmniContact | Schüsse aus dem Anlauf mit Ballpose | huggingface.co/datasets/lightcone02/OmniContact-Dataset, Teilmenge `soccer/case1_kick_forward` (Bedingungen akzeptieren) | CC BY-NC 4.0 |
| AMASS (CMU 10, 74, 69; ACCAD) | Schüsse aus dem Stand, Gehen | amass.is.tue.mpg.de (SMPL-X N), danach `holosoma_retargeting/data_utils/prep_amass_smplx_for_rt.py` | AMASS-Lizenz, nur Forschung |

## Ablauf

```bash
# 1) Quellclips -> Menschenclips
python -m reference_data.pipeline human --work work/ \
    --root lafan=~/data/lafan1 --root omnicontact=~/data/OmniContact-Dataset --root amass=~/data/amass_rt
# 2) Retargeting (Retarget-Umgebung, ca. 0.1–0.3 s pro Frame und Kern)
python -m reference_data.pipeline retarget --work work/ --jobs 8
# 3) Export, Annotation, Abnahme (motion_matching-Umgebung)
python -m reference_data.pipeline post --work work/ --out reference_out/
python -m reference_data.pipeline summary --out reference_out/   # Sekunden je Gruppe
python -m reference_data.verify reference_out/                   # Exit 1, wenn ein Clip durchfällt
```

Jede Stufe überspringt fertige Clips (`--force` macht sie neu), `--only <id>` beschränkt auf einen Clip.
Handkorrekturen (Kontaktframe, Schussfuß, Gruppe, Trim) gehören in `catalog.yaml` unter `annotation`,
in Sekunden des getrimmten Quellclips. Danach `post --force --only <id>`.

Ziele aus dem Plan: 5–10 min Gehen, 10–30 Schüsse aus dem Anlauf (mind. 2 Anlaufschritte, einzelne
Ausfallschritte werden als `reject` aussortiert), etwa 10 Schüsse aus dem Stand.

## Tests

```bash
python -m pytest tests/test_reference_data.py   # beide Umgebungen, ohne GPU und ohne Datensätze
```

Die Tests prüfen synthetische Clips (Stand-Schuss, Gehen), das Exportformat bis in
`build_kick_database` von Phase 2 und, in der Retarget-Umgebung, den holosoma-Retargeter am OMOMO-Demo
des Submoduls.

## Offene Punkte

- **Echte Daten noch nicht gelaufen.** Die Datensätze lassen sich aus der Cloud nicht laden. Geprüft ist
  die Kette mit dem OMOMO-Demo (Markerfehler 5–11 cm) und einem echten Xsens-BVH (Hüfte 1.7 cm, Zehen ca. 4 cm,
  Durchdringung 0.25 cm). Die Bereiche in `catalog.yaml` sind erste Vorschläge (`status: todo`) und müssen gesichtet werden.
- **OmniContact:** Gelenknamen der BVH und die Zeitzuordnung BVH ↔ npz (Ball) sind aus der Doku abgeleitet,
  nicht an echten Dateien geprüft. Beim ersten Lauf den Kontaktabstand in den `.json` ansehen.
- **Ballgröße:** kommt aus `robots/ball/ball.yaml` (zurzeit 0.15 m, Schätzung). Die Menschenbälle sind als
  Größe 5 (0.22 m) angenommen.
- **Knöchelmomente:** Die Machbarkeitsprüfung scheitert meist an den Knöcheln (Pitch 14 N·m, Roll 9.45 N·m,
  Hebel geschätzt). Wenn zu viele Clips durchfallen: `Thresholds(torque_ratio=...)` lockern oder langsamer
  abspielen (`speeds` im Katalog). Wichtig für die Hardware-Entscheidung.
- **GMR** (`gmr_lupus.py`): Konfiguration erzeugt, aber GMR selbst wurde hier nicht ausgeführt.
  Erst lokal prüfen, bevor man es fürs Gehen statt holosoma nutzt.
- **Unbekannte Hardwarewerte** (Masse, Momente) kommen aus `robots/lupus/hardware.yaml` (Booster K1 als
  Referenz). Nach Änderungen dort Modell neu bauen und `post --force` laufen lassen.
