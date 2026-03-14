from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import mlflow


def set_tracking_uri_sqlite(sqlite_path: Path) -> str:
    """Configure MLflow tracking uri sqlite:///..."""
    uri = f"sqlite:///{sqlite_path}"
    mlflow.set_tracking_uri(uri)
    return uri


def init_experiment(experiment: str = "shdb-af-ecg") -> None:
    """Sélectionne/crée l'experiment MLflow."""
    mlflow.set_experiment(experiment)


def log_json(path: Path, obj: Any, artifact_path: str) -> None:
    """Écrit un JSON localement puis loggue comme artefact MLflow."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    mlflow.log_artifact(str(path), artifact_path=artifact_path)


def log_text(path: Path, text: str, artifact_path: str) -> None:
    """Écrit un texte localement puis loggue comme artefact MLflow."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    mlflow.log_artifact(str(path), artifact_path=artifact_path)


def log_and_register_torch_model(
    model,
    *,
    artifact_path: str = "model",
    registered_model_name: str,
    tags: Optional[dict[str, str]] = None,
) -> str:
    """
    Log un modèle PyTorch (CPU) comme MLflow Model ET l'enregistre dans le Model Registry.
    Retourne le model_uri de la version enregistrée.

    - registered_model_name: nom stable dans le registry (ex: "shdbaf_resnet1d_4c")
    - tags: tags optionnels appliqués à la *model version* (très utile pour filtrer)
    """
    import mlflow.pytorch
    import torch.nn as nn

    if not isinstance(model, nn.Module):
        raise TypeError("log_and_register_torch_model attend un torch.nn.Module")

    model_cpu = model.to("cpu").eval()

    # 1) log model in current run + register into registry in one call
    model_info = mlflow.pytorch.log_model(
        model_cpu,
        artifact_path=artifact_path,
        registered_model_name=registered_model_name,
    )

    # 2) tag the registered model version (optional)
    if tags:
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            # model_info.model_uri looks like: "runs:/<run_id>/model"
            # The registered version is available via model_info.registered_model_version
            # but this attribute may vary by mlflow versions.
            ver = getattr(model_info, "registered_model_version", None)
            if ver is not None:
                # ver can be an object with .name and .version or a dict-like
                name = getattr(ver, "name", None) or getattr(ver, "registered_model_name", None) or registered_model_name
                version = getattr(ver, "version", None)
                if version is not None:
                    for k, v in tags.items():
                        client.set_model_version_tag(name, str(version), k, v)
        except Exception:
            # tags are "nice to have" — do not fail training for this
            pass

    return getattr(model_info, "model_uri", f"models:/{registered_model_name}/latest")