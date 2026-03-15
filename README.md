# SHDB-AF ECG — Pipeline données + baseline ResNet1D (Pixi + MLflow)

Ce dépôt propose un **pipeline reproductible** pour travailler avec la base **PhysioNet SHDB-AF** (WFDB) et entraîner une baseline **ResNet1D** sur des fenêtres ECG **multi-classes** (MVP : `N / AFIB / AFL / AT`), avec suivi des runs dans **MLflow**.

> Objectif principal : disposer d’une chaîne complète “de la donnée brute → dataset fenêtré → entraînement traçable”, et itérer rapidement sur les choix de fenêtrage / entraînement.

---

## Contenu

- Téléchargement SHDB-AF via WFDB (PhysioNet)
- Inventaire local (`index.parquet`)
- Split **par sujet** anti-fuite (`splits.parquet`)
- Extraction des **segments de rythme** depuis `.atr` (`segments.parquet`)
- Génération d’un dataset fenêtré **memmap** (`.npy`) + `manifest.parquet`
- Entraînement PyTorch + métriques + artefacts + **MLflow** (SQLite) + **Model Registry** (optionnel)

---

## Prérequis

- macOS / Linux (testé sur macOS Apple Silicon)
- [Pixi](https://pixi.sh/)
- Python géré par Pixi (aucune installation Python manuelle nécessaire)

---

## Installation

```bash
pixi install

Test rapide :

pixi run python -c "import wfdb, polars, torch; print('ok')"
```

⸻

Configuration

Les paramètres sont centralisés dans :
- config/settings.toml

Points clés :
- seed global
- chemins de données ([paths])
- ratios de split sujet ([split])
- fenêtrage ([dataset] : win_sec, stride_sec, majority_thr, classes, format="npy")
- entraînement ([train])
- modèle ([model])

⸻

Pipeline données (commande par commande)
```
Ordre recommandé :

pixi run download
pixi run index
pixi run split
pixi run segments
pixi run window-counts
pixi run make-windows
```

Artefacts générés
- data/raw/shdb-af/ : données WFDB (brut)
- data/interim/index.parquet : inventaire records
- data/interim/splits.parquet : split par sujet (train/val/test)
- data/interim/segments.parquet : segments de rythmes
- data/processed/<dataset_tag>/manifest.parquet + fichiers .npy : dataset fenêtré

dataset_tag est dérivé de la config (fenêtre/stride/canaux/classes/format).

⸻

Entraînement + MLflow

Lancer un run :
```
pixi run train
```
Lancer l’UI MLflow (SQLite) :
```
pixi run mlflow-ui
```
Ouvrir ensuite l’URL affichée (souvent http://127.0.0.1:5000).

Ce qui est loggué
- paramètres du run (dataset_tag, hyperparams, modèle…)
- métriques : val_f1_macro, val_acc, F1 par classe, etc.
- perf système : temps data/compute, throughput samples_per_sec_*, mémoire
- artefacts : report_val.json, cm_val.png, report_test.json, cm_test.png, best.pt
- (optionnel) enregistrement du modèle dans le Model Registry MLflow

⸻

Structure du dépôt
```
src/shdbaf/
  settings.py
  artifacts.py
  data/
    download.py
    index.py
    splits.py
    segments.py
    window_counts.py
    windowing.py
    torch_dataset.py          # memmap .npy (dataloader officiel)
  models/
    resnet1d.py
  training/
    train.py                  # entrypoint
    loop.py                   # boucle d’entraînement 1 epoch
    metrics.py                # eval + confusion matrix
    profiling.py              # timings + throughput
    mlflow_utils.py           # init/log/register MLflow
    system.py                 # device + mémoire
config/
  settings.toml
data.md                       # documentation détaillée du pipeline données
```

⸻

Résultats initiaux (résumé)

Premiers essais sur le fenêtrage :
- 10s / stride 5s : très bonnes perfs sur N et AFIB, mais AFL et surtout AT restent difficiles (classes rares, confusion avec AFIB).
- 30s / stride 10s : N/AFIB restent solides, mais AFL/AT ne s’améliorent pas dans les premiers runs.

➡️ Interprétation : le pipeline est sain, mais la séparation AFL/AT nécessite souvent des ajustements d’entraînement (imbalance) et/ou des features orientées “rythme”.

⸻

Données / licence / citation
- Dataset : [SHDB-AF sur PhysioNet](https://physionet.org/content/shdb-af/1.0.1/)

Les données ne sont pas versionnées dans ce dépôt. Utilisez pixi run download.

Merci de respecter les conditions d’utilisation/citation indiquées sur PhysioNet pour SHDB-AF.

⸻

Aide / documentation
- Documentation pipeline données : data.md
- WFDB Python : https://wfdb.readthedocs.io/
- MLflow : https://mlflow.org/docs/latest/