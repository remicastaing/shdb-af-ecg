from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


def project_root() -> Path:
    # src/shdbaf/settings.py -> src/shdbaf -> src -> PROJECT_ROOT
    return Path(__file__).resolve().parents[2]


class PathsCfg(BaseModel):
    raw_dir: str = "data/raw/shdb-af"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    artifacts_dir: str = ".artifacts"
    mlflow_db: str = "mlflow.db"

    def resolve(self, p: str) -> Path:
        return project_root() / p

    @property
    def raw_path(self) -> Path:
        return self.resolve(self.raw_dir)

    @property
    def interim_path(self) -> Path:
        return self.resolve(self.interim_dir)

    @property
    def processed_path(self) -> Path:
        return self.resolve(self.processed_dir)

    @property
    def artifacts_path(self) -> Path:
        return self.resolve(self.artifacts_dir)

    @property
    def mlflow_db_path(self) -> Path:
        return self.resolve(self.mlflow_db)

class SplitCfg(BaseModel):
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    only_annotated: bool = True

class LabelsCfg(BaseModel):
    known: tuple[str, ...] = ("N", "AFIB", "AFL", "AT", "NOD", "PAT")

class DatasetCfg(BaseModel):
    win_sec: float = 10.0
    stride_sec: float = 10.0
    channels: Literal["ecg1", "ecg2", "both"] = "both"
    majority_thr: float = 0.60
    classes: tuple[str, ...] = ("N", "AFIB", "AFL", "AT")
    chunk_size: int = 4096
    format: Literal["npy", "npz"] = "npy"

    def tag(self) -> str:
        w = int(self.win_sec)
        st = int(self.stride_sec)
        cls = f"{len(self.classes)}c"
        return f"win{w}s_stride{st}s_{self.channels}_{cls}_{self.format}"


class TrainCfg(BaseModel):
    batch_size: int = 256
    lr: float = 1e-3
    epochs: int = 3
    steps_per_epoch: int = 2000
    num_workers: int = 0
    cache_size: int = 8
    shuffle: bool = True
    log_every: int = 25
    profile_batches: int = 150
    warmup_batches: int = 10


class ModelCfg(BaseModel):
    name: str = "resnet1d"
    base: int = 32
    k: int = 7
    blocks: tuple[int, ...] = (2, 2, 2)


class Settings(BaseModel):
    seed: int = 42
    split: SplitCfg = Field(default_factory=SplitCfg)
    paths: PathsCfg = Field(default_factory=PathsCfg)
    dataset: DatasetCfg = Field(default_factory=DatasetCfg)
    train: TrainCfg = Field(default_factory=TrainCfg)
    model: ModelCfg = Field(default_factory=ModelCfg)


def load_settings(path: str | Path = "config/settings.toml") -> Settings:
    p = project_root() / path if not isinstance(path, Path) else path
    with p.open("rb") as f:
        data = tomllib.load(f)
    return Settings.model_validate(data)
