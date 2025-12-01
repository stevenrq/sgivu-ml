from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib

from app.core.config import Settings, get_settings


class ModelRegistry:
    """Repositorio local de modelos entrenados (artefacto + metadata).

    Gestiona versiones serializadas en disco para rollback y trazabilidad,
    manteniendo un apuntador al último modelo entrenado en ``latest.json``.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Prepara rutas de almacenamiento y configuración de modelos."""
        self.settings = settings or get_settings()
        self.model_dir: Path = self.settings.model_path()
        self.latest_metadata_path = self.model_dir / "latest.json"

    def _artifact_path(self, version: str) -> Path:
        """Construye la ruta del artefacto .joblib para una versión."""
        return self.model_dir / f"{self.settings.model_name}_{version}.joblib"

    def save(self, model: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Persiste el modelo y la metadata asociada.

        Args:
            model: Pipeline o estimador ya entrenado listo para inferencia.
            metadata: Información de entrenamiento (métricas, features, versión).

        Returns:
            Dict con la metadata enriquecida con el campo ``version`` asignado.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        merged_metadata = {**metadata, "version": timestamp}
        artifact_path = self._artifact_path(timestamp)

        joblib.dump({"model": model, "metadata": merged_metadata}, artifact_path)
        self.latest_metadata_path.write_text(json.dumps(merged_metadata, indent=2))
        return merged_metadata

    def load_latest(self) -> Tuple[Any, Dict[str, Any]]:
        """Carga el último modelo entrenado.

        Raises:
            FileNotFoundError: Si aún no existe un modelo persistido.

        Returns:
            Tuple con el modelo y la metadata asociada.
        """
        if not self.latest_metadata_path.exists():
            raise FileNotFoundError("No se encontro modelo entrenado.")

        metadata = json.loads(self.latest_metadata_path.read_text())
        artifact_path = self._artifact_path(metadata["version"])
        artifact = joblib.load(artifact_path)
        model = artifact.get("model", artifact)
        model_metadata = artifact.get("metadata", metadata)
        return model, model_metadata

    def latest_metadata(self) -> Dict[str, Any] | None:
        """Devuelve la metadata del modelo activo sin cargar el artefacto."""
        if not self.latest_metadata_path.exists():
            return None
        return json.loads(self.latest_metadata_path.read_text())
