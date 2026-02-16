# RL_plane_navigator
[![DOI](https://zenodo.org/badge/1159129908.svg)](https://doi.org/10.5281/zenodo.18658323)

Reinforcement Learning Umgebung für Air Traffic Control mit verschiedenen RL-Algorithmen und Custom Environments.

## Installation

Das Projekt kann einfach mit pip installiert werden:

```bash
pip install .
```

## Projektstruktur

### Custom Environments (`custom_envs/`)
Hier liegen die selbst erstellten Gym-Environments für verschiedene Crossing-Szenarien:
- `crossing_planes.py` - Basis Crossing Environment mit Absolute Aktionen
- `crossing_planes_abs.py` - älteren variante mit Absolute Aktionen
- `crossing_planes_conti.py` - Kontinuierliche Aktionen
- `crossing_planes_multiHead.py` - Multi-Head Architektur
- `lunarlander_multioutput.py` - LunarLander Variante mit Multi-Output

### RL-Algorithmen (`rl_zoo3/`)
In diesem Ordner liegen die neu implementierten RL-Algorithmen und das RLZOO-Framwork

### Simulator (`simulator/`)
Die Anbindung an BlueSky (BS) liegt hier:
- `blue_sky_adapter.py` - Adapter für die BlueSky Flugsimulation
- `simulator_interface.py` - Generisches Simulator Interface
- `test_blue_sky_adapter.py` - Tests für den BS Adapter

### Hyperparameter (`hyperparams/`)
Alle Hyperparameter-Konfigurationen für die verschiedenen Algorithmen liegen hier als YAML-Dateien:
- `ppo.yml` - Standard PPO
- `masked_ppo.yml` - Masked PPO Varianten
- `masked_hybrid_ppo.yml` - Hybrid PPO Varianten
- `multioutputppo.yml` - Multi-Output PPO
- Weitere Algorithmus-spezifische Konfigurationen

### Scenario Parser (`parser/`)
Hier liegt die logick die aud den DFS Scenario files das scenario parst.
Hier ist noch Arbeit. Das Scenario sollte langfristig aus anderen scenario files geparsed werden.
Die hier verwendete Daten sid warscheinlich vorverarbeitetete Radar tracks.

## Verwendung

### Training starten
```bash
./startTraining.sh
```

### Dashboards

Das TensorBoard Dashboard kann mit folgendem Skript gestartet werden:
```bash
./startDashbords.sh
```

Oder manuell:
```bash
tensorboard --logdir logs/tensorboard/
```

### Optuna Hyperparameter-Optimierung
```bash
./startOptuna.sh
```

### Visualisierung
```bash
./startVisualisation.sh
```

## Tests

Im `scripts/` Ordner befinden sich drei Test-Skripte für verschiedene Experimente:
Die Ergebnisse dieser Experimente wurden anschießend in der Bachlorthesis genauer beleuchtet.

### 1. Lunar Lander Tests
```bash
./scripts/run_lunar_tests_full.sh
```


### 2. Crossing Tests
```bash
./scripts/run_crossing_tests_full.sh
```

### 3. Finale Evaluierung
```bash
run_final_test.sh
```

## Logs und Ergebnisse

- `logs/` - Training Logs und TensorBoard Daten
- `logs_final_final_Lunar/` - Ergebnisse der finalen Lunar Lander Tests
- `logs_secondTest/` - Ergebnisse der zweiten Testreihe
- `results_*/` - CSV-Dateien und Plots der Evaluierungen

## Weitere Skripte

- `train.py` / `train_sbx.py` - Training mit verschiedenen Backends
- `enjoy.py` / `enjoy_sbx.py` - Trainierte Modelle ausführen

sbx ist ein Backaend das in Umgebungen mit grafikartenzugriff um ein vielfaches schneller sein können.
