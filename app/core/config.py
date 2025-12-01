import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import EnvSettingsSource


class LenientEnvSettingsSource(EnvSettingsSource):
    """Evita fallar en listas vacías o valores no JSON para scopes."""

    def decode_complex_value(self, field_name: str, field: Any, value: Any) -> Any:
        """Interpreta valores compuestos tolerando strings vacíos o inválidos."""
        if value is None:
            return value
        if isinstance(value, (bytes, bytearray)):
            value = value.decode()
        if isinstance(value, str) and value.strip() == "":
            return value
        try:
            return super().decode_complex_value(field_name, field, value)
        except json.JSONDecodeError:
            return value


class Settings(BaseSettings):
    """Parámetros globales del microservicio de predicción SGIVU-ML.

    Incluye endpoints de origen de datos (gateway), configuración de seguridad
    JWT/JWKS, ubicación de artefactos de modelo y ajustes de ventana temporal
    para el pipeline de demanda.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "sgivu-ml"
    app_version: str = "0.1.0"
    environment: str = "dev"

    sgivu_purchase_sale_url: str = "http://sgivu-vehicle"
    sgivu_vehicle_url: str = "http://sgivu-purchase-sale"

    sgivu_auth_discovery_url: str | None = None
    service_internal_secret_key: str | None = None

    model_dir: str = "models"
    model_name: str = "demand_forecaster"
    default_horizon_months: int = 6
    request_timeout_seconds: float = 15.0
    retrain_cron: str = "0 3 1 * *"
    retrain_timezone: str = "UTC"
    min_history_months: int = 6
    target_column: str = "sales_count"
    permissions_predict: list[str] = ["ml:predict"]
    permissions_retrain: list[str] = ["ml:retrain"]
    permissions_models: list[str] = ["ml:models"]

    @field_validator(
        "permissions_predict",
        "permissions_retrain",
        "permissions_models",
        mode="before",
    )
    @classmethod
    def _split_permissions(cls, value):
        """Normaliza listas de scopes leídas desde variables de entorno."""
        if isinstance(value, str):
            return [
                scope.strip()
                for scope in value.replace(" ", "").split(",")
                if scope.strip()
            ]
        return value

    def model_path(self) -> Path:
        """Crea (si no existe) y devuelve la ruta al directorio de modelos."""
        path = Path(self.model_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        """Inserta la fuente leniente antes de dotenv y secretos en disco."""
        # Capa leniente para valores de env que deberían ser listas (evita JSON vacío)
        return (
            init_settings,
            LenientEnvSettingsSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )


@lru_cache
def get_settings() -> Settings:
    """Devuelve la configuración cacheada (singleton)."""
    return Settings()
