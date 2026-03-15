# Architecture des modèles

## 1. ResNet1D

**Fichier** : [`src/shdbaf/models/resnet1d.py`](src/shdbaf/models/resnet1d.py:1)

Le modèle est un **ResNet1D** adapté aux séries temporelles (ECG), inspiré de l'architecture ResNet classique :

### 1.1 Structure
```
Input: (batch, 2, T)  # 2 canaux ECG, T samples

Stem:
  Conv1D(2 → base, k=7, stride=2) + BN + ReLU
  MaxPool1D(k=3, stride=2)

Layer 1: n × BasicBlock1D(base → base)
Layer 2: n × BasicBlock1D(base → base*2, stride=2)
Layer 3: n × BasicBlock1D(base*2 → base*4, stride=2)

Head:
  AdaptiveAvgPool1D(1)
  Flatten
  Linear(base*4 → num_classes)

Output: (batch, num_classes)  # logits
```

### 1.2 BasicBlock1D
```python
Conv1D(in_ch → out_ch, k=7, stride, padding=3) + BN + ReLU
Conv1D(out_ch → out_ch, k=7, stride=1, padding=3) + BN
+ residual connection (downsampling si nécessaire)
ReLU
```

### 1.3 Hyperparamètres
- `base=32` : nombre de filtres de base (32 → 64 → 128)
- `k=7` : taille du kernel convolutif
- `blocks=[2, 2, 2]` : 2 blocs par couche (6 blocs au total)

### 1.4 Justification
- **1D Convolutions** : adaptées aux séries temporelles (patterns locaux)
- **Residual connections** : facilitent l'entraînement de réseaux profonds
- **Kernel large (k=7)** : capture des features temporelles plus larges qu'un kernel 3x3
- **Downsampling progressif** : réduit la dimension temporelle tout en augmentant le nombre de features

### 1.5 Fonction de perte : CrossEntropyLoss avec class weights

Pour gérer le déséquilibre entre classes (N >> AFIB >> AFL > AT), on utilise des **class weights** :

```python
# Calcul des poids : inverse de la fréquence, normalisé
cls_w = compute_class_weights(manifest, "train", num_classes)
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(cls_w, device=device))
```

Cela pénalise davantage les erreurs sur les classes rares (AFL, AT).

### 1.6 Optimiseur

**AdamW** avec learning rate fixe :
```python
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
```

---