from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import OrderedDict
from typing import Dict, Tuple

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class MemmapDatasetConfig:
    root: Path
    manifest_path: Path
    split: str = "train"
    cache_size: int = 8
    channels_first: bool = True
    dtype: torch.dtype = torch.float32


class MemmapWindowDataset(Dataset):
    """
    Lit des chunks NPY memmap:
      chunk_00001_x.npy, chunk_00001_y.npy
    + manifest.parquet (chunk + offset + label_id).
    """
    def __init__(self, cfg: MemmapDatasetConfig):
        self.cfg = cfg
        m = pl.read_parquet(cfg.manifest_path).filter(pl.col("split") == cfg.split)

        self.chunks = m["chunk"].to_list()      # "chunk_00001"
        self.offsets = m["offset"].to_list()    # int
        self.labels = m["label_id"].to_list()   # int

        self._cache: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()

    def __len__(self) -> int:
        return len(self.labels)

    def _load_chunk(self, stem: str) -> Dict[str, np.ndarray]:
        if stem in self._cache:
            self._cache.move_to_end(stem)
            return self._cache[stem]

        x = np.load(self.cfg.root / f"{stem}_x.npy", mmap_mode="r")
        y = np.load(self.cfg.root / f"{stem}_y.npy", mmap_mode="r")
        obj = {"x": x, "y": y}

        self._cache[stem] = obj
        self._cache.move_to_end(stem)
        while len(self._cache) > self.cfg.cache_size:
            self._cache.popitem(last=False)

        return obj

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        stem = self.chunks[idx]
        off = int(self.offsets[idx])
        y = int(self.labels[idx])

        obj = self._load_chunk(stem)
        x = obj["x"][off]  # (T, C)

        xt = torch.tensor(x, dtype=self.cfg.dtype)
        if self.cfg.channels_first:
            xt = xt.transpose(0, 1)  # (C, T)
        yt = torch.tensor(y, dtype=torch.long)
        return xt, yt