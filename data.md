# SHDB-AF — Pipeline Données (data.md)

Ce document décrit **toute la chaîne de traitement des données** du projet *shdb-af-ecg* : depuis le téléchargement PhysioNet jusqu’à la génération d’un dataset fenêtré prêt pour PyTorch (format memmap `.npy`) et son utilisation en entraînement / évaluation avec MLflow.

> Hypothèse : le projet est structuré avec un package `shdbaf` (dans `src/shdbaf`) et un environnement **Pixi**.
> Les paramètres “source de vérité” sont centralisés dans `config/settings.toml` et chargés par `src/shdbaf/settings.py`.

---

## 1) Vue d’ensemble : artefacts, dossiers et dépendances

### 1.1 Dossiers standard

- `data/raw/shdb-af/`  
  Données brutes téléchargées depuis PhysioNet (WFDB) : `.hea`, `.dat`, annotations `.atr`, métadonnées `AdditionalData.csv`, etc.

- `data/interim/`  
  Artefacts “intermédiaires” **déterministes** et légers :
  - `index.parquet` : inventaire des enregistrements locaux + méta header (fs, sig_len, etc.)
  - `splits.parquet` : split **par sujet** (train/val/test) sans fuite
  - `segments.parquet` : segments de rythmes (déduits des annotations `.atr`)

- `data/processed/<dataset_tag>/`  
  Dataset fenêtré prêt pour l’entraînement (manifest + fichiers de données).
  - `manifest.parquet` : “table des fenêtres” (split, record, start/end, label, chunk, offset…)
  - fichiers `.npy` : les fenêtres en chunks (memmap-friendly, accès rapide par memmap)

- `.artifacts/`  
  Sorties du training + exports (best checkpoint, confusion matrix, reports, etc.).

- `mlflow.db`  
  Backend SQLite MLflow (runs, metrics, model registry…).

---

## 2) Paramètres : `config/settings.toml` (source de vérité)

Exemple (résumé) :

```toml
seed = 42

[paths]
raw_dir = "data/raw/shdb-af"
interim_dir = "data/interim"
processed_dir = "data/processed"
artifacts_dir = ".artifacts"
mlflow_db = "mlflow.db"

[split]
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15
only_annotated = true

[dataset]
win_sec = 10
stride_sec = 10
channels = "both"
majority_thr = 0.60
classes = ["N", "AFIB", "AFL", "AT"]
chunk_size = 4096
format = "npy"

[train]
batch_size = 256
lr = 1e-3
epochs = 3
steps_per_epoch = 2000
num_workers = 0
cache_size = 8
shuffle = true
log_every = 25
profile_batches = 150
warmup_batches = 10
# eval_every = 500
# eval_batches = 40

[model]
name = "resnet1d"
base = 32
k = 7
blocks = [2, 2, 2]
```

### Pourquoi centraliser ?
- Reproductibilité : un run = un `settings.toml`.
- Traçabilité : MLflow loggue les params de run (dataset_tag, win/stride, etc.).
- Moins de duplication : chemins, seed, ratios, hyperparams, tout est unifié.

---

## 3) Étape 0 — Installation / environnement

### Commandes
```bash
pixi install
```

### Vérifications rapides
```bash
pixi run python -c "import wfdb, polars, torch; print('ok')"
```

---

## 4) Étape 1 — Télécharger SHDB-AF depuis PhysioNet

### Objectif
Avoir localement les fichiers WFDB nécessaires :
- `.hea` (headers)
- `.dat` (signal)
- `.atr` (annotations de rythme, si disponibles)
- `AdditionalData.csv` (métadonnées sujet, annotation flag)
- `RECORDS.txt` (liste des enregistrements)

### Pourquoi passer par `wfdb` ?
WFDB sait télécharger proprement les ressources PhysioNet et gère les détails de structure.

### Commande
```bash
pixi run download
```

### Résultat attendu
Dans `data/raw/shdb-af/` :
- `RECORDS.txt`
- `AdditionalData.csv`
- fichiers `001.hea`, `001.dat`, `001.atr`, … selon disponibilités.

---

## 5) Étape 2 — Construire l’index “source de vérité” (`index.parquet`)

### Objectif
Scanner localement les `.hea`, et construire un inventaire complet des enregistrements :
- fréquence d’échantillonnage `fs`
- nombre de canaux `n_sig`
- longueur `sig_len`
- durée `duration_sec`
- présence de `.dat`, `.atr`, `.qrs`…

### Pourquoi ?
- Détecter rapidement les incohérences (headers sans `.dat`, etc.).
- Base stable pour toutes les étapes suivantes (split, segments, windowing).

### Commande
```bash
pixi run index
```

### Artefact
- `data/interim/index.parquet`

### Contrôles utiles
```bash
pixi run python -c "import polars as pl; df=pl.read_parquet('data/interim/index.parquet'); print(df.shape); print(df.select(['fs','n_sig']).unique())"
```

---

## 6) Étape 3 — Split par sujet (anti-fuite) (`splits.parquet`)

### Objectif
Créer un split **par sujet** (Subject_ID) pour éviter toute fuite :
- Un même sujet ne doit **jamais** être à cheval entre train/val/test.

### Sources
- `index.parquet` (records existants)
- `AdditionalData.csv` (mapping `Data_ID -> Subject_ID`, drapeau Annotated)

### Choix de conception
- `only_annotated = true` : MVP sur les records disposant d’annotations exploitables.
- Split déterministe via `seed`.

### Commande
```bash
pixi run split
```

### Artefact
- `data/interim/splits.parquet`

### Vérifications
```bash
pixi run python -c "import polars as pl; df=pl.read_parquet('data/interim/splits.parquet'); print(df.group_by('split').len()); print(df.group_by('split').agg(pl.col('subject_id').n_unique()))"
```

---

## 7) Étape 4 — Extraction des segments de rythme (`segments.parquet`)

### Objectif
Transformer les annotations `.atr` en segments temporels :
- `start_sec`, `end_sec`, `label`
- `record`, `subject_id`, `split`

### Pourquoi ?
Les annotations WFDB utilisent des “marqueurs de changement” (ex `(AFIB`, `(N`, etc.).  
On reconstruit alors des segments continus : *de ce marqueur jusqu’au prochain*.

### Commande
```bash
pixi run segments
```

### Artefact
- `data/interim/segments.parquet`

### Contrôle
```bash
pixi run python -c "import polars as pl; df=pl.read_parquet('data/interim/segments.parquet'); print(df.group_by('label').len().sort('len', descending=True))"
```

---

## 8) Étape 5 — Estimer la distribution de fenêtres (sanity check)

### Objectif
Avant de générer un gros dataset, estimer :
- nombre de fenêtres conservées
- distribution par classe
- pertes dues aux fenêtres “mixtes” (majority)

### Commande
```bash
pixi run window-counts
```

### Interprétation
- Si les classes rares deviennent quasi nulles : ajuster `stride_sec` (overlap), `win_sec`, `majority_thr`.
- Si trop de fenêtres “skipped/mixed” : baisser `majority_thr` (ou revoir le schéma de labels).

---

## 9) Étape 6 — Windowing (fenêtrage) vers dataset “processed”

### Objectif
Construire un dataset supervisé par fenêtre :
- fenêtres fixes `(win_sec)` glissant avec `(stride_sec)`
- affectation d’un label par **majorité temporelle** (≥ `majority_thr`)
- sélection des canaux (`ecg1`, `ecg2`, `both`)
- écriture en **chunks** de taille `chunk_size`
- génération d’un `manifest.parquet`

### Pourquoi le manifest ?
- C’est l’index de référence du dataset processed.
- Permet de reconstruire un mapping stable split/label/record/start, etc.
- Le dataloader PyTorch peut accéder aux fichiers chunkés via `chunk + offset`.

### Formats
- `format = "npy"` : **format unique** du projet (memmap-friendly, rapide, robuste I/O).

### Commande
```bash
pixi run make-windows
```

Le dossier de sortie est :
- `data/processed/<dataset_tag>/`
où `<dataset_tag>` est dérivé de `settings.dataset.tag()`.

### Artefacts
- `data/processed/<tag>/manifest.parquet`
- fichiers chunkés `.npy`

### Contrôles
```bash
pixi run python -c "import polars as pl; from shdbaf.settings import load_settings; S=load_settings(); tag=S.dataset.tag(); m=pl.read_parquet(f'data/processed/{tag}/manifest.parquet'); print(m.group_by(['split','label']).len())"
```

---

## 10) Accès au dataset côté PyTorch (memmap)

> Note : le projet n’utilise plus le format `.npz`. Le dataloader officiel est `src/shdbaf/data/torch_dataset.py` (memmap `.npy`).

### Objectif
Charger efficacement des fenêtres depuis un dataset très volumineux.
Le dataloader :
- lit `manifest.parquet`
- ouvre les chunks `.npy` en memmap
- utilise un petit cache LRU (`cache_size`) pour éviter de rouvrir en boucle

### Test rapide
```bash
pixi run python -c "from pathlib import Path; from shdbaf.data.torch_dataset import MemmapWindowDataset, MemmapDatasetConfig; from shdbaf.settings import load_settings; S=load_settings(); tag=S.dataset.tag(); root=Path('data/processed')/tag; ds=MemmapWindowDataset(MemmapDatasetConfig(root=root, manifest_path=root/'manifest.parquet', split='train', cache_size=2)); x,y=ds[0]; print(len(ds), x.shape, y)"
```

---

## 11) Entraînement + suivi MLflow

### Objectif
- suivre la perf (loss, f1, confusion matrix)
- suivre la perf système (timings batch, throughput, mémoire)
- historiser et comparer les runs
- (optionnel) enregistrer le modèle dans le Model Registry

### Commande
```bash
pixi run train
```

### Lancer l’UI MLflow
```bash
pixi run mlflow-ui
```

Ouvre ensuite l’URL affichée (souvent `http://127.0.0.1:5000`).

### Métriques utiles (typique)
- `val_f1_macro`, `val_acc`, f1 par classe `val_f1_<LABEL>`
- `samples_per_sec_step`, `samples_per_sec_epoch`
- `time_compute_ms_step`, `time_data_wait_ms_step`
- `mem_rss_mb_step`, `mem_avail_mb_step`

### Artefacts (eval)
- `cm_val.png`, `report_val.json`
- `cm_test.png`, `report_test.json`
- `best.pt`

---

## 12) Dépannage (FAQ)

### 12.1 “Aucun fichier n’apparaît dans /data/raw”
- Vérifier `paths.raw_dir` dans `settings.toml`.
- Vérifier que `pixi run download` écrit bien dans `data/raw/shdb-af`.

### 12.2 `ModuleNotFoundError: No module named 'shdbaf'`
- Le package doit être importable : structure `src/shdbaf/...`
- Lancer depuis la racine projet, et vérifier que Pixi inclut `src` dans `PYTHONPATH` (ou packaging correct).

### 12.3 `Descriptors cannot be created directly` (protobuf / mlflow)
- Conflit versions protobuf / torch / mlflow.
- Solution : rester cohérent avec conda-forge et une version protobuf compatible.

### 12.4 Performance faible / très lente
- Mesurer `data_wait` vs `compute` (MLflow).
- Sur Mac MPS, la chauffe peut faire baisser le throughput.
- Réduire `batch_size`, fermer les applis lourdes, éviter `num_workers` trop élevés.

### 12.5 AFL/AT non apprises (F1≈0)
- Problème classique de déséquilibre + représentations.
- Pistes :
  - `stride_sec` plus petit (overlap)
  - `majority_thr` plus élevé (fenêtres plus “pures”)
  - focal loss / oversampling
  - ajouter des features RR (rythme) si possible

---

## 13) Pipeline “commande unique” (rappel)

Ordre standard :

```bash
pixi run download
pixi run index
pixi run split
pixi run segments
pixi run window-counts
pixi run make-windows
pixi run train
pixi run mlflow-ui
```

---

## 14) Références
- PhysioNet SHDB-AF : https://physionet.org/content/shdb-af/1.0.1/
- WFDB Python : https://wfdb.readthedocs.io/
- MLflow : https://mlflow.org/docs/latest/
