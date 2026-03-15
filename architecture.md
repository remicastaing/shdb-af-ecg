# Architecture du projet SHDB-AF ECG

Ce document décrit en détail l'architecture technique du projet **SHDB-AF ECG**, un pipeline complet de classification de rythmes cardiaques à partir de données ECG de la base PhysioNet SHDB-AF.

---

## 1. Vue d'ensemble

### 1.1 Objectif du projet

Le projet vise à créer un **pipeline reproductible** pour :
- Télécharger et préparer les données ECG de la base SHDB-AF (PhysioNet)
- Extraire des segments de rythmes annotés et générer des fenêtres d'entraînement
- Entraîner un modèle de classification multi-classes (ResNet1D) pour détecter 4 types de rythmes : `N` (Normal), `AFIB` (Atrial Fibrillation), `AFL` (Atrial Flutter), `AT` (Atrial Tachycardia)
- Suivre les expériences avec MLflow (métriques, artefacts, Model Registry)

### 1.2 Principes directeurs

- **Reproductibilité** : Configuration centralisée (`settings.toml`), seed fixe, versioning des artefacts
- **Traçabilité** : Tous les runs sont enregistrés dans MLflow avec leurs paramètres et métriques
- **Modularité** : Pipeline décomposé en étapes indépendantes et réutilisables
- **Performance** : Utilisation de memmap pour les datasets volumineux, profiling système intégré
- **Anti-fuite** : Split par sujet (Subject_ID) pour éviter toute fuite de données entre train/val/test

---

## 2. Stack technique

### 2.1 Gestion d'environnement : Pixi

Le projet utilise **[Pixi](https://pixi.sh/)** comme gestionnaire d'environnement, qui :
- Gère Python et toutes les dépendances via conda-forge
- Simplifie l'installation : `pixi install` suffit
- Définit des tâches (`pixi run <task>`) pour chaque étape du pipeline
- Garantit la reproductibilité multi-plateforme (testé sur macOS Apple Silicon)

**Fichier de configuration** : [`pixi.toml`](pixi.toml)

### 2.2 Dépendances principales

| Bibliothèque | Version | Usage |
|-------------|---------|-------|
| **Python** | ≥3.11, <3.13 | Langage principal |
| **WFDB** | * | Téléchargement et lecture des données PhysioNet (format WFDB) |
| **Polars** | * | Manipulation de données tabulaires (plus rapide que pandas) |
| **PyArrow** | * | Backend pour Parquet (format de stockage intermédiaire) |
| **NumPy** | ≥2.4.2, <3 | Calculs numériques, memmap pour datasets volumineux |
| **PyTorch** | * | Framework de deep learning (modèle ResNet1D, entraînement) |
| **Scikit-learn** | * | Métriques d'évaluation (F1-score, confusion matrix) |
| **MLflow** | ≥3 | Suivi des expériences, Model Registry |
| **Pydantic** | ≥2 | Validation et typage de la configuration |
| **Matplotlib** | ≥3.10.8, <4 | Génération des confusion matrices (PNG) |
| **Typer** | * | CLI pour les scripts |
| **Rich** | * | Affichage console enrichi |
| **Tqdm** | ≥4.67.3, <5 | Barres de progression |
| **Psutil** | * | Monitoring mémoire et CPU |

### 2.3 Structure de fichiers

```
shdb-af-ecg/
├── config/
│   └── settings.toml              # Configuration centralisée (source de vérité)
├── data/
│   ├── raw/
│   │   └── shdb-af/               # Données WFDB brutes (téléchargées)
│   ├── interim/                   # Artefacts intermédiaires
│   │   ├── index.parquet          # Inventaire des enregistrements
│   │   ├── splits.parquet         # Split train/val/test par sujet
│   │   └── segments.parquet       # Segments de rythmes annotés
│   └── processed/
│       └── <dataset_tag>/         # Datasets fenêtrés (par configuration)
│           ├── manifest.parquet   # Manifest des fenêtres
│           ├── chunk_00001_x.npy  # Données X (memmap)
│           ├── chunk_00001_y.npy  # Labels Y
│           └── ...
├── src/
│   └── shdbaf/                    # Package Python principal
│       ├── settings.py            # Chargement de la configuration
│       ├── artifacts.py           # Helpers pour les chemins d'artefacts
│       ├── data/                  # Modules de traitement des données
│       │   ├── download.py        # Téléchargement SHDB-AF
│       │   ├── index.py           # Inventaire des enregistrements
│       │   ├── splits.py          # Split par sujet
│       │   ├── segments.py        # Extraction des segments de rythme
│       │   ├── window_counts.py   # Estimation de la distribution
│       │   ├── windowing.py       # Génération du dataset fenêtré
│       │   └── torch_dataset.py   # DataLoader PyTorch (memmap)
│       ├── models/
│       │   └── resnet1d.py        # Architecture ResNet1D
│       └── training/
│           ├── train.py           # Script d'entraînement principal
│           ├── loop.py            # Boucle d'entraînement par epoch
│           ├── metrics.py         # Évaluation et métriques
│           ├── profiling.py       # Profiling système (temps, throughput)
│           ├── mlflow_utils.py    # Utilitaires MLflow
│           └── system.py          # Détection device (CPU/MPS/CUDA)
├── mlruns/                        # Artefacts MLflow (runs locaux)
├── mlflow.db                      # Backend SQLite MLflow
├── .artifacts/                    # Artefacts du dernier run
├── pixi.toml                      # Configuration Pixi
├── pyproject.toml                 # Métadonnées du package Python
├── README.md                      # Documentation utilisateur
├── data.md                        # Documentation détaillée du pipeline
└── architecture.md                # Ce document
```

---

## 3. Configuration centralisée

### 3.1 Fichier `config/settings.toml`

Toute la configuration du projet est centralisée dans ce fichier TOML, organisé en sections :

#### **[Seed global]**
```toml
seed = 42
```
Assure la reproductibilité des splits et de l'entraînement.

#### **[paths]** — Chemins des données
```toml
raw_dir = "data/raw/shdb-af"
interim_dir = "data/interim"
processed_dir = "data/processed"
artifacts_dir = ".artifacts"
mlflow_db = "mlflow.db"
```

#### **[split]** — Split train/val/test par sujet
```toml
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15
only_annotated = true
```

#### **[dataset]** — Paramètres de fenêtrage
```toml
win_sec = 30              # Durée d'une fenêtre (secondes)
stride_sec = 5            # Stride (overlap = win_sec - stride_sec)
channels = "both"         # Canaux ECG : ecg1 | ecg2 | both
majority_thr = 0.60       # Seuil de majorité pour l'étiquetage
classes = ["N", "AFIB", "AFL", "AT"]
chunk_size = 4096         # Nombre de fenêtres par chunk
format = "npy"            # Format : npy (memmap-friendly)
```

Le **dataset_tag** est généré automatiquement à partir de ces paramètres :
```python
# Exemple: win30s_stride5s_both_4c_npy
tag = f"win{win_sec}s_stride{stride_sec}s_{channels}_{len(classes)}c_{format}"
```

#### **[train]** — Hyperparamètres d'entraînement
```toml
batch_size = 256
cache_size = 32           # Cache memmap (nombre de chunks)
num_workers = 0           # DataLoader workers (0 recommandé sur macOS)
lr = 1e-3                 # Learning rate
epochs = 3
steps_per_epoch = 2000    # Nombre de steps par epoch
shuffle = true
log_every = 25            # Fréquence de logging (steps)
profile_batches = 150     # Nombre de batches pour profiling
warmup_batches = 10       # Batches de warmup avant profiling
eval_every = 500          # Évaluation partielle sur val tous les N steps
eval_batches = 40         # Nombre de batches pour l'évaluation partielle
```

#### **[model]** — Architecture ResNet1D
```toml
name = "resnet1d"
base = 32                 # Nombre de filtres de base
k = 7                     # Taille du kernel (conv1d)
blocks = [2, 2, 2]        # Nombre de blocs par couche
```

### 3.2 Chargement de la configuration

Le module [`src/shdbaf/settings.py`](src/shdbaf/settings.py:1) définit des modèles Pydantic pour valider et typer la configuration :

```python
from shdbaf.settings import load_settings

S = load_settings()  # Charge config/settings.toml
tag = S.dataset.tag()  # Génère le dataset_tag
```

**Avantages** :
- Validation automatique des types et valeurs
- Autocomplétion IDE
- Documentation inline
- Centralisation de la configuration

---

## 4. Pipeline de données

### 4.1 Étape 1 — Téléchargement WFDB

**Script** : [`src/shdbaf/data/download.py`](src/shdbaf/data/download.py:1)  
**Commande** : `pixi run download`

Télécharge la base SHDB-AF depuis PhysioNet via la bibliothèque `wfdb` :
- Fichiers `.hea` (headers), `.dat` (signal brut), `.atr` (annotations)
- `AdditionalData.csv` (métadonnées : Subject_ID, Annotated flag)
- `RECORDS.txt` (liste des enregistrements)

**Sortie** : `data/raw/shdb-af/`

### 4.2 Étape 2 — Index des enregistrements

**Script** : [`src/shdbaf/data/index.py`](src/shdbaf/data/index.py:1)  
**Commande** : `pixi run index`

Scanne les headers `.hea` pour construire un inventaire complet :
- Fréquence d'échantillonnage (`fs`), nombre de canaux (`n_sig`), longueur (`sig_len`)
- Durée en secondes (`duration_sec`)
- Présence de fichiers associés (`.dat`, `.atr`, `.qrs`)

**Sortie** : `data/interim/index.parquet`

### 4.3 Étape 3 — Split par sujet (anti-fuite)

**Script** : [`src/shdbaf/data/splits.py`](src/shdbaf/data/splits.py:1)  
**Commande** : `pixi run split`

Crée un split **train/val/test** en assurant qu'un même sujet n'est jamais présent dans plusieurs ensembles :
- Lit `index.parquet` et `AdditionalData.csv`
- Filtre les enregistrements annotés si `only_annotated = true`
- Attribue chaque sujet à un split selon les ratios définis
- Utilise le `seed` pour la reproductibilité

**Sortie** : `data/interim/splits.parquet`

### 4.4 Étape 4 — Extraction des segments de rythme

**Script** : [`src/shdbaf/data/segments.py`](src/shdbaf/data/segments.py:1)  
**Commande** : `pixi run segments`

Transforme les annotations `.atr` en segments temporels :
- Les annotations WFDB utilisent des marqueurs de changement (ex : `(AFIB`, `(N`)
- Reconstruction de segments continus : [start_sec, end_sec, label]
- Enrichissement avec subject_id et split

**Sortie** : `data/interim/segments.parquet`

### 4.5 Étape 5 — Estimation de la distribution (sanity check)

**Script** : [`src/shdbaf/data/window_counts.py`](src/shdbaf/data/window_counts.py:1)  
**Commande** : `pixi run window-counts`

Estime le nombre de fenêtres par classe avant la génération complète :
- Simule le fenêtrage avec les paramètres actuels
- Affiche la distribution par split et classe
- Aide à détecter les déséquilibres extrêmes

**Sortie** : Console (pas de fichier)

### 4.6 Étape 6 — Génération du dataset fenêtré

**Script** : [`src/shdbaf/data/windowing.py`](src/shdbaf/data/windowing.py:1)  
**Commande** : `pixi run make-windows`

Génère un dataset supervisé par fenêtre :
- Fenêtrage glissant : fenêtres de `win_sec` secondes avec stride `stride_sec`
- Étiquetage par majorité temporelle : une fenêtre est gardée si ≥`majority_thr` appartient à une classe
- Sélection des canaux : `ecg1`, `ecg2` ou `both` (2 canaux)
- Écriture en chunks memmap `.npy` de taille `chunk_size`
- Génération d'un manifest Parquet avec métadonnées

**Sortie** : 
- `data/processed/<dataset_tag>/manifest.parquet`
- `data/processed/<dataset_tag>/chunk_*_x.npy` (shape: [n_windows, n_samples, n_channels])
- `data/processed/<dataset_tag>/chunk_*_y.npy` (labels)

**Format du manifest** :
| Colonne | Type | Description |
|---------|------|-------------|
| `split` | str | train/val/test |
| `record` | str | Nom de l'enregistrement |
| `subject_id` | str | Identifiant sujet |
| `start_sample` | int | Index de début dans le signal brut |
| `label` | str | Classe (N, AFIB, AFL, AT) |
| `label_id` | int | ID numérique de la classe |
| `chunk` | str | Nom du chunk (ex: chunk_00001) |
| `offset` | int | Offset dans le chunk |

---



## 5. DataLoader PyTorch avec memmap

### 5.1 Motivation

Les datasets ECG fenêtrés sont volumineux (plusieurs Go). Charger tout en RAM est :
- Impossible sur certaines machines
- Inefficace (beaucoup de fenêtres ne sont jamais vues dans un epoch limité)

**Solution** : utiliser `numpy.memmap` pour accéder aux données sur disque sans tout charger en RAM.

### 5.2 Implémentation : `MemmapWindowDataset`

**Fichier** : [`src/shdbaf/data/torch_dataset.py`](src/shdbaf/data/torch_dataset.py:1)

#### **Fonctionnement**
1. Lit le manifest Parquet pour obtenir la liste des fenêtres (chunk + offset)
2. Pour chaque `__getitem__`, ouvre le chunk correspondant en memmap si pas déjà en cache
3. Extrait la fenêtre à l'offset spécifié
4. Maintient un **cache LRU** de `cache_size` chunks ouverts pour éviter les réouvertures

#### **Configuration**
```python
MemmapDatasetConfig(
    root=Path("data/processed/win30s_stride5s_both_4c_npy"),
    manifest_path=Path("...manifest.parquet"),
    split="train",
    cache_size=32,          # Garde 32 chunks ouverts
    channels_first=True,    # Transpose (T, C) → (C, T) pour PyTorch
    dtype=torch.float32
)
```

#### **Avantages**
- **Faible empreinte RAM** : seuls les chunks actifs sont en mémoire
- **Accès rapide** : memmap est optimisé pour l'accès aléatoire
- **Flexibilité** : peut gérer des datasets > RAM disponible

---

## 6. Pipeline d'entraînement

### 6.1 Script principal : `train.py`

**Fichier** : [`src/shdbaf/training/train.py`](src/shdbaf/training/train.py:1)  
**Commande** : `pixi run train`

#### **Workflow**
1. **Initialisation**
   - Charge `settings.toml`
   - Fixe les seeds (reproductibilité)
   - Détecte le device (CPU/MPS/CUDA)
   - Configure MLflow (SQLite backend)

2. **Préparation des données**
   - Vérifie l'existence du dataset fenêtré
   - Calcule les class counts et class weights
   - Crée les datasets (train/val/test) et DataLoaders

3. **Initialisation du modèle**
   - Instancie ResNet1D avec les hyperparamètres de `settings.toml`
   - Configure l'optimiseur AdamW
   - Crée la fonction de perte avec class weights

4. **Entraînement**
   - Boucle sur les epochs
   - Pour chaque epoch : `train_one_epoch()` (voir 6.2)
   - Évaluation complète sur validation après chaque epoch
   - Sauvegarde du meilleur modèle (best_val_f1)

5. **Évaluation finale**
   - Charge le meilleur checkpoint
   - Évalue sur le test set
   - Enregistre les métriques et artefacts dans MLflow

5. **Model Registry**
   - Enregistre le meilleur modèle dans le Model Registry MLflow
   - Tags : dataset_tag, seed, win_sec, stride_sec, channels, classes

### 6.2 Boucle d'entraînement : `loop.py`

**Fichier** : [`src/shdbaf/training/loop.py`](src/shdbaf/training/loop.py:1)

#### **Fonction `train_one_epoch()`**

Entraîne le modèle sur `steps_per_epoch` batches (pas sur tout le dataset) :

1. **Warmup batches**
   - Ignore les premiers batches pour stabiliser les mesures (OS cache, GPU warmup)

2. **Training loop**
   - Pour chaque batch :
     - Forward pass → calcul de la loss
     - Backward pass + optimisation
     - **Profiling** : mesure du temps data wait vs compute (voir 6.3)
     - Logging MLflow tous les `log_every` steps

3. **Évaluation partielle** (optionnel)
   - Tous les `eval_every` steps : évalue sur `eval_batches` du val set
   - Calcule F1-score macro et log dans MLflow
   - Permet de suivre la convergence sans attendre la fin de l'epoch

4. **Métriques epoch**
   - Loss moyenne
   - Throughput : samples/sec (epoch)
   - Durée totale

### 6.3 Profiling système : `profiling.py`

**Fichier** : [`src/shdbaf/training/profiling.py`](src/shdbaf/training/profiling.py:1)

Mesure les performances système en temps réel :

#### **Métriques collectées**
- **`time_data_wait_ms_step`** : temps d'attente pour charger un batch
- **`time_compute_ms_step`** : temps de forward + backward + step optimizer
- **`samples_per_sec_step`** : throughput instantané
- **`mem_rss_mb_step`** : RAM utilisée par le processus
- **`mem_avail_mb_step`** : RAM disponible sur le système

#### **Utilisation**
```python
profiler = StepProfiler(batch_size=256, device="mps")

# Avant le chargement du batch
profiler.mark_data_start()

# Batch chargé
X, y = next(train_loader)
profiler.mark_data_end()

# Forward + backward
profiler.mark_compute_start()
loss = loss_fn(model(X), y)
loss.backward()
opt.step()
profiler.mark_compute_end()

# Enregistrer les métriques
metrics = profiler.flush_step_metrics()
mlflow.log_metrics(metrics, step=global_step)
```

#### **Détection des bottlenecks**
- **`time_data_wait` > `time_compute`** : I/O bottleneck → augmenter `cache_size`, utiliser SSD
- **`time_compute` >> `time_data_wait`** : GPU/CPU bottleneck → batch_size plus grand, optimiser le modèle

### 6.4 Métriques d'évaluation : `metrics.py`

**Fichier** : [`src/shdbaf/training/metrics.py`](src/shdbaf/training/metrics.py:1)

#### **Fonction `evaluate_full()`**

Évalue le modèle sur un DataLoader complet :

```python
acc, f1_macro, cm, report = evaluate_full(model, val_loader, device, labels)
```

**Métriques retournées** :
- **Accuracy** : (TP + TN) / Total
- **F1-score macro** : moyenne non pondérée des F1 par classe
- **Confusion matrix** : matrice de confusion (numpy array)
- **Classification report** : dict avec précision/rappel/F1 par classe

#### **Génération des artefacts**
```python
# Confusion matrix PNG
save_confusion_matrix_png(cm, labels, "cm_val.png", title="Validation")

# JSON report
import json
with open("report_val.json", "w") as f:
    json.dump(report, f, indent=2)
```

---

## 7. Suivi des expériences avec MLflow

### 7.1 Configuration MLflow

**Backend** : SQLite local (`mlflow.db`)  
**Fichier** : [`src/shdbaf/training/mlflow_utils.py`](src/shdbaf/training/mlflow_utils.py:1)

```python
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("shdb-af-ecg")
```

### 7.2 Paramètres loggués

| Paramètre | Description |
|-----------|-------------|
| `seed` | Seed global |
| `device` | CPU/MPS/CUDA |
| `dataset_tag` | Tag du dataset (ex: win30s_stride5s_both_4c_npy) |
| `labels` | Classes (N,AFIB,AFL,AT) |
| `batch_size`, `lr`, `epochs`, `steps_per_epoch` | Hyperparamètres |
| `num_workers`, `cache_size`, `shuffle` | Config DataLoader |
| `model`, `base`, `k`, `blocks` | Architecture ResNet1D |

### 7.3 Métriques loggées

#### **Par step (pendant l'entraînement)**
- `train_loss_step` : loss instantanée
- `time_data_wait_ms_step`, `time_compute_ms_step` : profiling
- `samples_per_sec_step` : throughput
- `mem_rss_mb_step`, `mem_avail_mb_step` : mémoire

#### **Par epoch**
- `train_loss_epoch` : loss moyenne
- `val_f1_macro`, `val_acc` : métriques validation
- `val_f1_N`, `val_f1_AFIB`, `val_f1_AFL`, `val_f1_AT` : F1 par classe
- `epoch_duration_s` : durée de l'epoch
- `samples_per_sec_epoch` : throughput moyen

#### **Finales (test set)**
- `best_val_f1_macro` : meilleur F1 sur validation
- `test_f1_macro`, `test_acc` : métriques test
- `test_f1_<label>` : F1 par classe sur test

### 7.4 Artefacts loggués

**Structure dans MLflow** :
```
mlruns/
└── <experiment_id>/
    └── <run_id>/
        ├── artifacts/
        │   ├── checkpoints/
        │   │   └── best.pt                 # Meilleur checkpoint
        │   ├── data/
        │   │   ├── class_counts.json       # Comptage par classe
        │   │   └── class_weights.json      # Poids de classe
        │   ├── eval/
        │   │   ├── cm_val.txt              # Confusion matrix (texte)
        │   │   ├── cm_val.png              # Confusion matrix (image)
        │   │   ├── report_val.json         # Rapport validation
        │   │   ├── cm_test.txt
        │   │   ├── cm_test.png
        │   │   └── report_test.json
        │   └── model/
        │       └── <torch_model>           # Modèle enregistré
        └── params/
            └── ...
```

### 7.5 Model Registry

Le meilleur modèle est enregistré dans le **Model Registry MLflow** avec :
- **Nom** : `shdbaf_resnet1d_4c` (stable, indépendant du run)
- **Tags** : dataset_tag, seed, win_sec, stride_sec, channels, classes
- **Version** : incrémentée automatiquement à chaque enregistrement

**Avantages** :
- Versioning automatique des modèles
- Traçabilité : lien entre modèle et run MLflow
- Déploiement facilité : `mlflow models serve -m "models:/shdbaf_resnet1d_4c/1"`

### 7.6 Interface MLflow

**Lancer l'UI** :
```bash
pixi run mlflow-ui
```

Ouvre `http://127.0.0.1:5000` pour :
- Comparer les runs (métriques, paramètres)
- Visualiser les courbes (loss, F1, throughput)
- Télécharger les artefacts
- Gérer le Model Registry

---

## 8. Choix techniques justifiés

### 8.1 Pixi vs Conda/Poetry

**Choix** : Pixi

**Justification** :
- Intègre Python + dépendances + tâches dans un seul outil
- Résolution de dépendances rapide (conda-forge)
- Multi-plateforme testé (macOS Apple Silicon)
- Pas besoin d'installer Python manuellement

### 8.2 Polars vs Pandas

**Choix** : Polars

**Justification** :
- 5-10× plus rapide sur les opérations tabulaires
- Meilleure gestion de la mémoire
- API moderne et expressive
- Support natif Parquet

### 8.3 Parquet vs CSV

**Choix** : Parquet

**Justification** :
- Format binaire compressé (10× plus petit que CSV)
- Lecture/écriture rapide
- Schéma typé intégré
- Compatible avec Polars, Pandas, PyArrow

### 8.4 Memmap `.npy` vs PyTorch `.pt`

**Choix** : NumPy memmap (format `.npy`)

**Justification** :
- Accès aléatoire ultra-rapide (fichier mappé en mémoire virtuelle)
- Pas de désérialisation (données brutes)
- Compatible tous les outils NumPy/PyTorch
- Simple à implémenter (pas de protocole de sérialisation)

**Comparaison** :
| Format | Taille | Vitesse lecture | Compatibilité |
|--------|--------|-----------------|---------------|
| `.npz` | Compressé | Lent (décompression) | NumPy |
| `.pt` | Non compressé | Moyen | PyTorch |
| `.npy` (memmap) | Non compressé | **Rapide** | NumPy + PyTorch |

### 8.5 Pydantic vs ConfigParser

**Choix** : Pydantic + TOML

**Justification** :
- Validation de types automatique
- Autocomplétion IDE (type hints)
- Documentation inline
- Format TOML plus lisible que INI/JSON

### 8.6 MLflow SQLite vs Server

**Choix** : SQLite local

**Justification** :
- Pas de serveur à maintenir
- Portabilité : un seul fichier `mlflow.db`
- Suffisant pour usage local/recherche
- Migration vers serveur facile si besoin

---

## 9. Workflow complet

### 9.1 Première utilisation

```bash
# 1. Installation
pixi install

# 2. Télécharger les données
pixi run download

# 3. Pipeline de données
pixi run index
pixi run split
pixi run segments
pixi run window-counts  # Vérifier la distribution
pixi run make-windows

# 4. Entraînement
pixi run train

# 5. Visualiser les résultats
pixi run mlflow-ui
```

### 9.2 Itération sur les hyperparamètres

Pour tester une nouvelle configuration :

1. **Modifier `config/settings.toml`** (ex: `win_sec = 30`, `stride_sec = 5`)
2. **Regénérer le dataset** : `pixi run make-windows` (nouveau `dataset_tag`)
3. **Entraîner** : `pixi run train`
4. **Comparer dans MLflow** : les deux runs sont tracés

### 9.3 Ajout d'une nouvelle classe

1. Modifier `config/settings.toml` : `classes = ["N", "AFIB", "AFL", "AT", "NOD"]`
2. Regénérer le dataset : `pixi run make-windows`
3. Le modèle s'adapte automatiquement (`num_classes = len(classes)`)

---

## 10. Points d'attention et limitations

### 10.1 Performance sur macOS MPS

**Observation** : Sur Apple Silicon (M1/M2), le backend MPS peut chauffer et throttle.

**Solutions** :
- Réduire `batch_size`
- Limiter `steps_per_epoch`
- Utiliser `num_workers=0` (plus stable sur macOS)
- Fermer les applications gourmandes

### 10.2 Déséquilibre des classes

**Problème** : AFL et AT sont très rares (≪ 1% des fenêtres).

**Solutions implémentées** :
- Class weights dans la loss
- Stride réduit (overlap) pour plus de fenêtres

**Pistes futures** :
- Oversampling (ex: RandomOverSampler)
- Focal Loss
- Stratified sampling par epoch

### 10.3 Confusion AFL vs AFIB

**Observation** : Le modèle confond souvent AFL et AFIB (rythmes similaires).

**Pistes** :
- Features RR interval (rythme cardiaque)
- Augmentation de données (transformations temporelles)
- Architectures multi-modales (ECG + metadata)

### 10.4 Évolutivité

**Actuel** : Pipeline local, données < 100 Go

**Pour scaler** :
- Remplacer SQLite MLflow par un serveur distant
- Utiliser un cluster de calcul (ex: Ray, Dask)
- Stockage distribué (S3, MinIO)

---

## 11. Extensions futures

### 11.1 Architectures alternatives

- **LSTM/GRU** : modèles récurrents pour les dépendances temporelles longues
- **Transformers** : attention sur les séries temporelles
- **Hybrid CNN-RNN** : combine features locales + dépendances temporelles

### 11.2 Features additionnelles

- **RR intervals** : variabilité du rythme cardiaque
- **Spectrogrammes** : analyse fréquentielle
- **Wavelet transform** : décomposition multi-échelle

### 11.3 Augmentation de données

- Scaling (amplitude)
- Time warping
- Jittering (bruit)
- Cutout (masquage temporel)

### 11.4 Déploiement

- **API REST** : `mlflow models serve` ou FastAPI
- **Mobile** : conversion en TorchScript/ONNX pour iOS/Android
- **Edge** : déploiement sur devices médicaux

---

## 12. Références techniques

### 12.1 Standards et formats

- **WFDB** : Format de données PhysioNet — [wfdb.io](https://wfdb.io/)
- **Parquet** : Format de stockage columnar — [parquet.apache.org](https://parquet.apache.org/)
- **MLflow** : Plateforme de MLOps — [mlflow.org](https://mlflow.org/)

### 12.2 Bibliothèques

- **PyTorch** : [pytorch.org](https://pytorch.org/)
- **Polars** : [pola.rs](https://pola.rs/)
- **Pixi** : [pixi.sh](https://pixi.sh/)
- **Pydantic** : [docs.pydantic.dev](https://docs.pydantic.dev/)

### 12.3 Littérature

- **ResNet** : He et al., "Deep Residual Learning for Image Recognition" (2015)
- **ECG Classification** : Hannun et al., "Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network" (2019)
- **PhysioNet** : Goldberger et al., "PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals" (2000)

---

## 13. Maintenance et contribution

### 13.1 Tests

**TODO** : Ajouter des tests unitaires avec `pytest` :
```bash
pixi run pytest tests/
```

### 13.2 Documentation

- **README.md** : guide utilisateur rapide
- **data.md** : documentation détaillée du pipeline de données
- **architecture.md** : ce document (vision technique complète)

### 13.3 Versioning

- **Git** : versionner le code, les configs, mais **pas** les données brutes ni les artefacts MLflow
- `.gitignore` : exclure `data/`, `mlruns/`, `.artifacts/`, `mlflow.db`

### 13.4 Contact

Pour toute question ou contribution : Rémi Castaing — remi.castaing@gmail.com

---

**Fin du document d'architecture**
