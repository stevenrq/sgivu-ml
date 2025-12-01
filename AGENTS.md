# Repository Guidelines

## Project Structure & Module Organization
- Código de aplicación en `app/` (routers FastAPI, servicios, core de seguridad/config, modelos Pydantic).
- Scripts auxiliares en `scripts/` (cron de reentrenamiento) y `run.sh` (arranque uvicorn).
- Ejemplos y utilidades offline en `tests/` (demos con CSV, generación de datos sintéticos y artefactos de prueba).
- Artefactos de modelo se escriben en `models/` (creado en runtime según configuración).

## Build, Test, and Development Commands
- Instalar dependencias: `python3 -m pip install -r requirements.txt`.
- Ejecutar API local: `./run.sh` (usa `uvicorn app.main:app --host 0.0.0.0 --port 8000` por defecto).
- Desarrollar con autoreload: `UVICORN_CMD="uvicorn app.main:app --reload" ./run.sh`.
- Demo offline con CSV: `python3 tests/csv_offline_demo.py --csv <ruta> --horizon 6 --vehicle-type CAR --brand TOYOTA --model COROLLA`.
- Generar datos sintéticos: `python3 tests/generate_contracts.py --rows 5000 --output tests/data/contracts.csv`.

## Coding Style & Naming Conventions
- Python 3.11/3.12, indentación 4 espacios, tipado estático preferido (`Optional`, `Dict`, etc.).
- Docstrings en español para clases/métodos no triviales; mantener tono explicativo y conciso.
- Uso de FastAPI + Pydantic v2: preferir `model_dump`/`model_validate` y `Field` con descripciones.
- Rutas y permisos en minúsculas separadas por guion bajo (`permissions_predict`); constantes en MAYÚSCULAS.

## Testing Guidelines
- No hay suite automatizada formal; los scripts en `tests/` funcionan como pruebas manuales/funcionales.
- Convención de nombres: scripts descriptivos (`csv_offline_demo.py`, `generate_contracts.py`); si agregas tests automatizados, usa `test_*.py` con pytest.
- Para validar cambios de modelado, ejecuta el demo offline apuntando a un CSV reducido y revisa métricas/forecast generado.

## Commit & Pull Request Guidelines
- Commits claros y atómicos; prefijos útiles (`feat:`, `fix:`, `docs:`, `chore:`) facilitan lectura.
- Incluye en la descripción qué problema resuelves y el impacto en el pipeline (entrenamiento, predicción, seguridad).
- Para PRs: resume propósito, pasos de prueba manual (comandos ejecutados, payloads usados) y consideraciones de despliegue/rollback de modelos (`models/latest.json`).

## Security & Configuration Tips
- Configuración via variables de entorno (.env opcional): URLs de gateway (`SGIVU_PURCHASE_SALE_URL`, `SGIVU_VEHICLE_URL`), permisos (`PERMISSIONS_*`), y clave interna (`SERVICE_INTERNAL_SECRET_KEY`).
- OIDC: puedes apuntar `SGIVU_AUTH_DISCOVERY_URL` al `.well-known/openid-configuration`; el servicio cachea JWKS durante 1h.
- Evita commitear artefactos reales de `models/`; usa el volumen o directorio local ignorado para experimentos.

## Architecture Docs
- Service context: `docs/architecture/services/sgivu-ml-context.puml`
- Components (FastAPI, loader, training): `docs/architecture/services/sgivu-ml-components.puml`
- Data model (Pydantic requests/responses, metadata): `docs/architecture/datamodel/sgivu-ml-datamodel.puml`
