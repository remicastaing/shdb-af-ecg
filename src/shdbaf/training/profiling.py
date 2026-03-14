from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


def samples_per_sec(batch_size: int, batch_ms: float) -> float:
    """Débit (samples/s) à partir du temps moyen d'un batch (ms)."""
    return float(batch_size) / (batch_ms / 1000.0) if batch_ms > 0 else 0.0


@dataclass
class StepWindow:
    """Fenêtre glissante de timings entre deux logs."""
    data_wait_s: List[float] = field(default_factory=list)
    to_dev_s: List[float] = field(default_factory=list)
    compute_s: List[float] = field(default_factory=list)
    batch_s: List[float] = field(default_factory=list)

    def append(self, data_wait: float, to_dev: float, compute: float, batch: float) -> None:
        self.data_wait_s.append(data_wait)
        self.to_dev_s.append(to_dev)
        self.compute_s.append(compute)
        self.batch_s.append(batch)

    def reset(self) -> None:
        self.data_wait_s.clear()
        self.to_dev_s.clear()
        self.compute_s.clear()
        self.batch_s.clear()

    def means_ms(self) -> tuple[float, float, float, float]:
        """Retourne (data_ms, to_dev_ms, compute_ms, batch_ms)."""
        data_ms = float(np.mean(self.data_wait_s) * 1000) if self.data_wait_s else 0.0
        to_dev_ms = float(np.mean(self.to_dev_s) * 1000) if self.to_dev_s else 0.0
        compute_ms = float(np.mean(self.compute_s) * 1000) if self.compute_s else 0.0
        batch_ms = float(np.mean(self.batch_s) * 1000) if self.batch_s else 0.0
        return data_ms, to_dev_ms, compute_ms, batch_ms


@dataclass
class WarmProfile:
    """
    Profiling "début d'epoch" (moyennes des premiers batches hors warmup).
    Sert à avoir une mesure comparable entre runs.
    """
    enabled: bool
    warmup_batches: int
    profile_batches: int
    data_wait_s: List[float] = field(default_factory=list)
    to_dev_s: List[float] = field(default_factory=list)
    compute_s: List[float] = field(default_factory=list)
    batch_s: List[float] = field(default_factory=list)

    def consider(self, step: int, data_wait: float, to_dev: float, compute: float, batch: float) -> None:
        if not self.enabled:
            return
        if step < self.warmup_batches:
            return
        if step >= self.profile_batches:
            return
        self.data_wait_s.append(data_wait)
        self.to_dev_s.append(to_dev)
        self.compute_s.append(compute)
        self.batch_s.append(batch)

    def summary_ms(self) -> dict:
        if not self.enabled or not self.data_wait_s:
            return {}
        return {
            "profile_data_ms": float(np.mean(self.data_wait_s) * 1000),
            "profile_to_device_ms": float(np.mean(self.to_dev_s) * 1000),
            "profile_compute_ms": float(np.mean(self.compute_s) * 1000),
            "profile_batch_ms": float(np.mean(self.batch_s) * 1000),
        }