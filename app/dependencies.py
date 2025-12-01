from functools import lru_cache

from app.core.config import Settings, get_settings
from app.services.data_loader import DemandDatasetLoader
from app.services.model_registry import ModelRegistry
from app.services.prediction import PredictionService
from app.services.training import TrainingService


@lru_cache
def _settings() -> Settings:
    """Singleton de configuración global."""
    return get_settings()


@lru_cache
def _registry() -> ModelRegistry:
    """Repositorio de modelos versionados compartido."""
    return ModelRegistry(_settings())


def get_loader() -> DemandDatasetLoader:
    """Cliente de datos para contratos e inventario."""
    return DemandDatasetLoader(_settings())


def get_trainer() -> TrainingService:
    """Servicio de entrenamiento de modelos de demanda."""
    return TrainingService(registry=_registry(), settings=_settings())


def get_prediction_service() -> PredictionService:
    """Servicio de predicción listo para inyección en endpoints."""
    return PredictionService(
        loader=get_loader(),
        trainer=get_trainer(),
        registry=_registry(),
        settings=_settings(),
    )
