from __future__ import annotations

from pathlib import Path

from shdbaf.settings import load_settings


# ---------- Helpers ----------
def _S():
    return load_settings()


# ---------- Raw files ----------
def records_txt() -> Path:
    """Fichier listant les enregistrements WFDB à télécharger."""
    S = _S()
    return S.paths.raw_path / "RECORDS.txt"


def additional_csv() -> Path:
    """Metadata PhysioNet (Data_ID, Subject_ID, Annotated...)."""
    S = _S()
    return S.paths.raw_path / "AdditionalData.csv"


# ---------- Interim artifacts ----------
def index_parquet() -> Path:
    S = _S()
    return S.paths.interim_path / "index.parquet"


def splits_parquet() -> Path:
    S = _S()
    return S.paths.interim_path / "splits.parquet"


def segments_parquet() -> Path:
    S = _S()
    return S.paths.interim_path / "segments.parquet"


# ---------- Processed artifacts ----------
def processed_dataset_dir(tag: str) -> Path:
    S = _S()
    return S.paths.processed_path / tag


def manifest_parquet(tag: str) -> Path:
    return processed_dataset_dir(tag) / "manifest.parquet"


# ---------- Local artifacts ----------
def artifacts_dir() -> Path:
    S = _S()
    return S.paths.artifacts_path


def mlflow_db() -> Path:
    S = _S()
    return S.paths.mlflow_db_path



