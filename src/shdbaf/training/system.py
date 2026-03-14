from __future__ import annotations

import os

import psutil
import torch


def get_device() -> str:
    """Choisit automatiquement le backend (MPS sur Mac, sinon CUDA, sinon CPU)."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def sync_if_needed(device: str) -> None:
    """Synchronise device pour des timings fiables (MPS/CUDA asynchrones)."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def mem_rss_mb() -> float:
    """Mémoire RSS du process Python (Mo)."""
    p = psutil.Process(os.getpid())
    return p.memory_info().rss / (1024 * 1024)


def sys_mem_available_mb() -> float:
    """Mémoire système disponible (Mo)."""
    return psutil.virtual_memory().available / (1024 * 1024)