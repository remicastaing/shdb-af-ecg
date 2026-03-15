# SHDB-AF ECG — Pipeline données + baseline ResNet1D

Pipeline reproductible pour la classification de rythmes cardiaques à partir de données ECG PhysioNet SHDB-AF.

> **Objectif** : Chaîne complète "données brutes → dataset fenêtré → entraînement traçable" avec ResNet1D et MLflow.

---

## 📋 Vue d'ensemble

Ce projet propose un pipeline complet pour :
- Télécharger et préparer les données **PhysioNet SHDB-AF** (format WFDB)
- Générer un dataset fenêtré **multi-classes** : `N` (Normal), `AFIB` (Atrial Fibrillation), `AFL` (Atrial Flutter), `AT` (Atrial Tachycardia)
- Entraîner un modèle **ResNet1D** pour la classification de rythmes ECG
- Suivre les expériences avec **MLflow** (métriques, artefacts, Model Registry)

### Caractéristiques principales

✅ **Reproductible** : Configuration centralisée, seeds fixés, versioning des artefacts  
✅ **Traçable** : Tous les runs enregistrés dans MLflow avec paramètres et métriques  
✅ **Performance** : Datasets memmap pour gérer de gros volumes, profiling système intégré  
✅ **Anti-fuite** : Split par sujet (Subject_ID) pour éviter toute contamination entre ensembles

---

## 🚀 Démarrage rapide

### Prérequis

- **macOS / Linux** (testé sur macOS Apple Silicon)
- **[Pixi](https://pixi.sh/)** (gestionnaire d'environnement)
- Python géré automatiquement par Pixi

### Installation

```bash
# Installer les dépendances
pixi install

# Test rapide
pixi run python -c "import wfdb, polars, torch; print('ok')"
```

### Pipeline complet

```bash
# 1. Télécharger les données PhysioNet
pixi run download

# 2. Préparer les données (index, split, segments, fenêtrage)
pixi run index
pixi run split
pixi run segments
pixi run window-counts    # Vérifier la distribution
pixi run make-windows

# 3. Entraîner le modèle
pixi run train

# 4. Visualiser les résultats dans MLflow
pixi run mlflow-ui        # Ouvrir http://127.0.0.1:5000
```

---

## ⚙️ Configuration

Les paramètres sont centralisés dans **[`config/settings.toml`](config/settings.toml)**:

| Section | Paramètres clés |
|---------|----------------|
| **[paths]** | Chemins des données (raw, interim, processed) |
| **[split]** | Ratios train/val/test (0.70/0.15/0.15), anti-fuite par sujet |
| **[dataset]** | Fenêtrage : `win_sec`, `stride_sec`, `channels`, `classes`, `majority_thr` |
| **[train]** | Hyperparamètres : `batch_size`, `lr`, `epochs`, `steps_per_epoch` |
| **[model]** | Architecture ResNet1D : `base`, `k`, `blocks` |

**Exemple de configuration** :
```toml
[dataset]
win_sec = 30              # Fenêtre de 30 secondes
stride_sec = 5            # Stride de 5s (overlap de 25s)
channels = "both"         # ECG1 + ECG2
classes = ["N", "AFIB", "AFL", "AT"]

[train]
batch_size = 256
lr = 1e-3
epochs = 3
```

Le **dataset_tag** est généré automatiquement : `win30s_stride5s_both_4c_npy`

---

## 📊 Pipeline de données

### Artefacts générés

```
data/
├── raw/shdb-af/                      # Données WFDB brutes (téléchargées)
├── interim/
│   ├── index.parquet                 # Inventaire des enregistrements
│   ├── splits.parquet                # Split train/val/test par sujet
│   └── segments.parquet              # Segments de rythmes annotés
└── processed/<dataset_tag>/
    ├── manifest.parquet              # Manifest des fenêtres
    └── chunk_*.npy                   # Données fenêtrées (memmap)
```

### Étapes du pipeline

| Commande | Description | Sortie |
|----------|-------------|--------|
| `pixi run download` | Télécharge SHDB-AF depuis PhysioNet | `data/raw/shdb-af/` |
| `pixi run index` | Scanne les headers WFDB | `index.parquet` |
| `pixi run split` | Split par sujet (anti-fuite) | `splits.parquet` |
| `pixi run segments` | Extrait les segments de rythme | `segments.parquet` |
| `pixi run window-counts` | Estime la distribution de fenêtres | Console |
| `pixi run make-windows` | Génère le dataset fenêtré | `manifest.parquet` + `.npy` |

---

## 🧠 Modèle et entraînement

### Architecture ResNet1D

- **Type** : Réseau convolutif 1D avec connexions résiduelles
- **Input** : (batch, 2, T) — 2 canaux ECG, T samples
- **Output** : (batch, 4) — logits pour N, AFIB, AFL, AT
- **Paramètres** : ~200k (base=32, k=7, blocks=[2,2,2])

Voir **[modeles.md](modeles.md)** pour les détails d'architecture.

### Entraînement

```bash
pixi run train
```

**Fonctionnalités** :
- **Class weights** : gère le déséquilibre entre classes
- **Profiling système** : mesure temps data/compute, throughput, mémoire
- **Évaluation continue** : validation tous les `eval_every` steps
- **Best checkpoint** : sauvegarde du meilleur modèle (F1 validation)

**Métriques loggées** (MLflow) :
- `val_f1_macro`, `val_acc` : métriques globales
- `val_f1_<label>` : F1 par classe (N, AFIB, AFL, AT)
- `samples_per_sec_*` : throughput
- `time_data_wait_ms_*`, `time_compute_ms_*` : profiling
- `mem_rss_mb_*` : mémoire

### Visualisation MLflow

```bash
pixi run mlflow-ui
```

Ouvrir **http://127.0.0.1:5000** pour :
- Comparer les runs (métriques, paramètres)
- Visualiser les courbes (loss, F1, throughput)
- Télécharger les artefacts (checkpoints, confusion matrices)
- Gérer le Model Registry

---

## 📁 Structure du projet

```
shdb-af-ecg/
├── config/
│   └── settings.toml                 # Configuration centralisée
├── data/                             # Données (non versionnées)
├── src/shdbaf/                       # Package Python principal
│   ├── data/                         # Modules de traitement des données
│   │   ├── download.py
│   │   ├── index.py
│   │   ├── splits.py
│   │   ├── segments.py
│   │   ├── window_counts.py
│   │   ├── windowing.py
│   │   └── torch_dataset.py         # DataLoader memmap
│   ├── models/
│   │   └── resnet1d.py              # Architecture ResNet1D
│   └── training/
│       ├── train.py                  # Script d'entraînement
│       ├── loop.py
│       ├── metrics.py
│       ├── profiling.py
│       ├── mlflow_utils.py
│       └── system.py
├── mlruns/                           # Artefacts MLflow (local)
├── mlflow.db                         # Backend SQLite MLflow
├── pixi.toml                         # Configuration Pixi
├── README.md                         # Ce fichier
├── architecture.md                   # Documentation technique complète
├── data.md                           # Documentation du pipeline données
└── modeles.md                        # Documentation des modèles
```

---

## 📈 Résultats initiaux

### Performances observées

| Configuration | N | AFIB | AFL | AT | Notes |
|--------------|---|------|-----|----|----|
| **win=10s, stride=5s** | ✅ Excellent | ✅ Bon | ⚠️ Moyen | ❌ Faible | Classes rares difficiles |
| **win=30s, stride=10s** | ✅ Excellent | ✅ Bon | ⚠️ Moyen | ❌ Faible | AFL/AT confusion avec AFIB |

### Interprétation

- **N et AFIB** : bien appris (classes majoritaires, patterns clairs)
- **AFL et AT** : difficiles (classes rares < 1%, confusion avec AFIB)

**Pistes d'amélioration** :
- Augmenter l'overlap (stride plus petit)
- Focal loss ou oversampling
- Features RR interval (rythme cardiaque)
- Augmentation de données

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[README.md](README.md)** | Guide de démarrage rapide (ce fichier) |
| **[architecture.md](architecture.md)** | Architecture technique complète (stack, choix techniques) |
| **[data.md](data.md)** | Documentation détaillée du pipeline de données |
| **[modeles.md](modeles.md)** | Documentation des modèles (ResNet1D, alternatives) |

---

## 📖 Références

### Dataset
- **SHDB-AF** : [PhysioNet SHDB-AF Database](https://physionet.org/content/shdb-af/1.0.1/)

⚠️ **Les données ne sont pas versionnées dans ce dépôt.** Utilisez `pixi run download`.  
Merci de respecter les conditions d'utilisation et de citation de PhysioNet.

### Bibliothèques
- **Pixi** : [pixi.sh](https://pixi.sh/)
- **WFDB Python** : [wfdb.readthedocs.io](https://wfdb.readthedocs.io/)
- **PyTorch** : [pytorch.org](https://pytorch.org/)
- **Polars** : [pola.rs](https://pola.rs/)
- **MLflow** : [mlflow.org/docs/latest/](https://mlflow.org/docs/latest/)

---

## 🤝 Contribution

Pour toute question ou contribution : **Rémi Castaing** — remi.castaing@gmail.com

---

## 📝 Licence

Voir les conditions d'utilisation de [PhysioNet SHDB-AF](https://physionet.org/content/shdb-af/1.0.1/).
