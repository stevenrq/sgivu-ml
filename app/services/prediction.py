from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import HTTPException, status

from app.core.config import Settings, get_settings
from app.services.data_loader import DemandDatasetLoader
from app.services.normalization import (
    canonicalize_brand_model,
    canonicalize_label,
)
from app.services.model_registry import ModelRegistry
from app.services.training import TrainingService

logger = logging.getLogger(__name__)


class PredictionService:
    """Orquesta reentrenamiento e inferencia de demanda mensual.

    Integra carga de datos desde purchase-sale/inventario, construcción de
    features temporales y ejecución del modelo versionado.
    """

    def __init__(
        self,
        loader: Optional[DemandDatasetLoader],
        trainer: TrainingService,
        registry: ModelRegistry,
        settings: Settings | None = None,
    ) -> None:
        """Crear el orquestador de predicción.

        Args:
            loader: Cliente de datos; puede ser None en flujos offline donde no se llama predict.
            trainer: Servicio de entrenamiento/feature engineering.
            registry: Repositorio de modelos versionados.
            settings: Configuración global.
        """
        self.loader = loader
        self.trainer = trainer
        self.registry = registry
        self.settings = settings or get_settings()

    async def retrain(
        self, start_date: Optional[date] = None, end_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """Reentrena usando el historial más reciente disponible.

        Args:
            start_date: Fecha mínima (opcional) para acotar el rango.
            end_date: Fecha máxima (opcional).

        Raises:
            HTTPException: Si no hay datos en el rango solicitado.

        Returns:
            Metadata del modelo entrenado (métricas, versión, muestras).
        """
        if not self.loader:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No hay loader de datos configurado para reentrenar.",
            )
        raw_df = await self.loader.load_transactions(
            start_date=start_date, end_date=end_date
        )
        if raw_df.empty:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No hay datos para entrenar en el rango solicitado.",
            )
        return self.trainer.train(raw_df)

    async def predict(
        self, filters: Dict[str, Any], horizon: int, confidence: float = 0.95
    ) -> Dict[str, Any]:
        """Genera pronóstico para un segmento de vehículos.

        Args:
            filters: Segmento solicitado (vehicle_type, brand, model, opcional line).
            horizon: Meses hacia adelante a pronosticar.
            confidence: Nivel de confianza (para IC gaussiano).

        Raises:
            HTTPException: Si no existe modelo entrenado o no hay historial del segmento.

        Returns:
            Dict con predicciones, versión del modelo y métricas de entrenamiento.
        """
        if not self.loader:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No hay loader de datos configurado para predecir.",
            )
        try:
            model, metadata = self.registry.load_latest()
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Aun no existe un modelo entrenado.",
            ) from exc

        raw_df = await self.loader.load_transactions()
        feature_df = self.trainer.build_feature_table(raw_df)

        normalized_filters: Dict[str, Any] = {}
        normalized_filters["vehicle_type"] = canonicalize_label(
            filters.get("vehicle_type")
        )
        normalized_brand, normalized_model = canonicalize_brand_model(
            filters.get("brand"), filters.get("model")
        )
        normalized_filters["brand"] = normalized_brand
        normalized_filters["model"] = normalized_model
        normalized_filters["line"] = canonicalize_label(filters.get("line"))
        if not normalized_filters["line"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Falta la línea del vehículo. Incluye brand/model/line completos para predecir.",
            )

        mask_base = (
            (feature_df["vehicle_type"] == normalized_filters.get("vehicle_type"))
            & (feature_df["brand"] == normalized_filters.get("brand"))
            & (feature_df["model"] == normalized_filters.get("model"))
            & (feature_df["line"] == normalized_filters.get("line"))
        )
        history = feature_df[mask_base].copy()

        if history.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    "No se encontró historial para la combinación solicitada. "
                    "Verifica vehicle_type/brand/model/line o reentrena con datos que incluyan ese segmento."
                ),
            )

        history = history.sort_values("event_month")
        forecast = self._forecast(model, metadata, history, horizon, confidence)
        return {
            "predictions": forecast,
            "model_version": metadata["version"],
            "metrics": metadata.get("metrics"),
        }

    async def predict_with_history(
        self, filters: Dict[str, Any], horizon: int, confidence: float = 0.95
    ) -> Dict[str, Any]:
        """Pronóstico + historial mensual para graficar en frontend."""
        if not self.loader:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No hay loader de datos configurado para predecir.",
            )
        try:
            model, metadata = self.registry.load_latest()
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Aun no existe un modelo entrenado.",
            ) from exc

        raw_df = await self.loader.load_transactions()
        feature_df = self.trainer.build_feature_table(raw_df)

        normalized_filters: Dict[str, Any] = {}
        normalized_filters["vehicle_type"] = canonicalize_label(
            filters.get("vehicle_type")
        )
        normalized_brand, normalized_model = canonicalize_brand_model(
            filters.get("brand"), filters.get("model")
        )
        normalized_filters["brand"] = normalized_brand
        normalized_filters["model"] = normalized_model
        normalized_filters["line"] = canonicalize_label(filters.get("line"))
        if not normalized_filters["line"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Falta la línea del vehículo. Incluye brand/model/line completos para predecir.",
            )

        mask_base = (
            (feature_df["vehicle_type"] == normalized_filters.get("vehicle_type"))
            & (feature_df["brand"] == normalized_filters.get("brand"))
            & (feature_df["model"] == normalized_filters.get("model"))
            & (feature_df["line"] == normalized_filters.get("line"))
        )
        history = feature_df[mask_base].copy()

        if history.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    "No se encontró historial para la combinación solicitada. "
                    "Verifica vehicle_type/brand/model/line o reentrena con datos que incluyan ese segmento."
                ),
            )

        history = history.sort_values("event_month")
        forecast = self._forecast(model, metadata, history, horizon, confidence)
        history_payload = [
            {
                "month": ts.date().isoformat(),
                "sales_count": float(sc),
            }
            for ts, sc in zip(history["event_month"], history["sales_count"])
        ]

        return {
            "predictions": forecast,
            "history": history_payload,
            "segment": {
                "vehicle_type": normalized_filters["vehicle_type"],
                "brand": normalized_filters["brand"],
                "model": normalized_filters["model"],
                "line": normalized_filters["line"],
            },
            "model_version": metadata["version"],
            "trained_at": metadata.get("trained_at"),
            "metrics": metadata.get("metrics"),
        }

    def _forecast(
        self,
        model: Any,
        metadata: Dict[str, Any],
        history: pd.DataFrame,
        horizon: int,
        confidence: float,
    ) -> List[Dict[str, Any]]:
        """Pronóstico iterativo mes a mes propagando los lags calculados.

        Se usa la desviación estándar de residuales del set de prueba para
        construir bandas de confianza (aprox. normal).
        """
        residual_std = metadata.get("metrics", {}).get("residual_std", 1.0)
        z_value = self._z_value(confidence)

        working_history = history.copy()
        target_month = working_history["event_month"].max()
        results: List[Dict[str, Any]] = []

        for _ in range(horizon):
            target_month = target_month + pd.offsets.MonthBegin(1)
            future_row = self.trainer.build_future_row(working_history, target_month)
            optional_cols = [
                col
                for col in getattr(self.trainer, "optional_category_cols", [])
                if col in future_row.columns
            ]
            features = future_row[
                self.trainer.category_cols + optional_cols + self.trainer.numeric_cols
            ]
            prediction = float(model.predict(features)[0])
            lower = max(0.0, prediction - z_value * residual_std)
            upper = max(lower, prediction + z_value * residual_std)

            results.append(
                {
                    "month": target_month.date().isoformat(),
                    "demand": prediction,
                    "lower_ci": lower,
                    "upper_ci": upper,
                }
            )

            appended = future_row.copy()
            appended["sales_count"] = prediction
            working_history = pd.concat([working_history, appended], ignore_index=True)

        return results

    def _z_value(self, confidence: float) -> float:
        """Aproxima el valor-z para un nivel de confianza (distribución normal)."""
        conf = min(max(confidence, 0.5), 0.99)
        if conf >= 0.99:
            return 2.58
        if conf >= 0.95:
            return 1.96
        if conf >= 0.90:
            return 1.64
        if conf >= 0.80:
            return 1.28
        return 1.0
