# sgivu-ml – Prediccion de demanda

Servicio FastAPI para estimar demanda mensual de vehiculos usados por tipo/marca/linea/modelo, integrado con los microservicios SGIVU.

## Fuentes de datos (via Gateway con OAuth2/JWT)

- Todas las llamadas desde sgivu-ml pasan por `sgivu-gateway` (Spring Cloud Gateway) para respetar OAuth2 y circuit breakers.
- `GET /v1/purchase-sales/search` (gateway → sgivu-purchase-sale): pagina contratos con `content`, `totalPages` (usa filtros de fecha, tipo, estado, precio).
- `GET /v1/purchase-sales/detailed` (gateway): listado enriquecido (cliente, usuario, vehiculo).
- Campos relevantes: `contractType` (PURCHASE/SALE), `contractStatus`, `purchasePrice`, `salePrice`, `paymentMethod`, `createdAt`, `updatedAt`, `vehicleId`, `clientId`, `userId`, `VehicleSummary` (type, brand, model, status).
- Enriquecimiento con inventario via gateway: `GET /v1/cars/{id}` y `GET /v1/motorcycles/{id}` para linea, anio, kilometraje, precios, estado.

## Diseño del dataset

- Objetivo: `sales_count` (demand) por mes y segmento (vehicle_type, brand, line, model).
- Variables base: precios compra/venta, margen, fechas, estado de contrato, tipo de pago, datos de vehiculo (linea, anio, kilometraje, estado).
- Ingenieria de atributos:
  - Ventas y compras por mes.
  - `margin = sale_price - purchase_price`.
  - `days_in_inventory` (venta - compra por vehicle_id).
  - `inventory_rotation = sales_count / max(purchases_count,1)`.
  - Tendencia/estacionalidad: mes, anio, seno/coseno del mes.
  - Lags y rolling: `lag_1`, `lag_3`, `lag_6`, `rolling_mean_3`, `rolling_mean_6`.

## Pipeline de entrenamiento (MLOps)

1. Carga async de contratos via gateway + enriquecimiento de inventario.
2. Normalización de categorías, fechas y cálculo de margen/días en inventario.
3. Agregación mensual por segmento (tipo + marca + modelo, línea opcional).
4. Features temporales: lags (1/3/6), rolling_mean (3/6), rotación, estacionalidad (mes/año, seno/coseno).
5. División temporal (80/20) y evaluación de candidatos con OneHot + escalado:
   - LinearRegression (baseline).
   - RandomForestRegressor (datasets medianos, no lineal).
   - XGBRegressor (si está instalado) para series cortas con no linealidad.
6. Selección por RMSE en test; se guarda `residual_std` para IC.
7. Serialización `joblib` en `models/{model_name}_{version}.joblib` + `models/latest.json`.
8. `/v1/ml/retrain` dispara entrenamiento; `scripts/cron_retrain.sh` permite cron mensual.

## Endpoints FastAPI

- `GET /health`: verificacion rapida.
- `POST /v1/ml/predict` (JWT requerido): cuerpo `vehicle_type`, `brand`, `model`, `line`, `horizon_months` (1-24), `confidence`. Devuelve lista de meses con demanda estimada e intervalos.
- `POST /v1/ml/retrain` (JWT requerido): opcional `start_date`, `end_date`. Retorna version y metricas.
- `GET /v1/ml/models/latest` (JWT requerido): metadata del ultimo modelo.
- OAuth2: sgivu-auth es el Authorization Server; usa `Authorization: Bearer <access_token>` emitido con flujo OAuth2 (password/authorization_code). El servicio valida la firma y `aud/iss` configurados.

## Documentación embebida (docstrings)

- Todo endpoint, servicio y modelo Pydantic expone docstrings en español que describen propósito, argumentos y valores de retorno.
- Explora `/app` para detalles rápidos: `app/routers/prediction_router.py` resume cada ruta; `app/services/` documenta orquestación, entrenamiento y predicción; `app/models/prediction_model.py` describe shape y restricciones de los payloads.
- Las dependencias de seguridad (`app/core/security.py`) también indican qué validan (`require_permissions`, `require_internal_or_permissions`) y cómo usar la clave interna.
- Usa `pydoc` o editores con tooltips para leer la documentación inline sin abrir este archivo.

## Reentrenamiento y cron

- Script `scripts/cron_retrain.sh` pensado para cron dentro del contenedor:
  - Variables: `TOKEN` (JWT de sgivu-auth), `API_URL` (por defecto `http://localhost:8000/v1/retrain`), `START_DATE`, `END_DATE`.
  - Programar ejemplo: `0 3 1 * * TOKEN=... /app/scripts/cron_retrain.sh`.

## Versionado y rollback

- Artefactos en `models/` con version UTC (yyyyMMddHHmmss).
- `latest.json` apunta al modelo activo; para rollback basta con restaurar ese archivo y el joblib deseado.

## Integracion con Spring Boot

- Consumir desde sgivu-gateway o servicios de compras/ventas via RestClient/HTTP Interface:
  - DTO ejemplo `PredictionRequestDto` con los campos del endpoint.
  - Cliente: `restClient.post().uri("http://sgivu-ml:8000/v1/predict").header("Authorization","Bearer "+jwt).body(request).retrieve().body(PredictionResponse.class);`
- Resilience4j:
  - Circuit breaker en el cliente (`failureRateThreshold` ajustado) y retry con backoff para /predict.
  - Timeouts alineados al `request_timeout_seconds` configurado en sgivu-ml.

## Seguridad

- Todas las rutas (excepto health) requieren `Authorization: Bearer <JWT>`.
- Actúa como Resource Server OIDC usando Authlib: valida JWT con JWKS (`AUTH_JWKS_URL`) o llave pública (`AUTH_PUBLIC_KEY`), verificando firma, `exp`, `iss` y `aud` (si se configuran `AUTH_ISSUER`/`AUTH_AUDIENCE`). También puedes apuntar `SGIVU_AUTH_DISCOVERY_URL` al `.well-known/openid-configuration` y el servicio resolverá automáticamente `jwks_uri` e `issuer`.
- Autorización por permisos (claim `rolesAndPermissions`): por defecto se exigen `ml:predict`, `ml:retrain` y `ml:models` para `/v1/predict`, `/v1/retrain` y `/v1/models/latest` respectivamente (misma convención `recurso:accion` que en los demás servicios). Puedes ajustar vía `PERMISSIONS_PREDICT`, `PERMISSIONS_RETRAIN` y `PERMISSIONS_MODELS` (listas separadas por coma); si las dejas vacías, cualquier JWT válido pasa.
- Para llamadas internas sin usuario se puede usar `SERVICE_INTERNAL_SECRET_KEY` (service-to-service).

## 📐 Diagramas
- Contexto del servicio: `docs/architecture/services/sgivu-ml-context.puml`
- Componentes (FastAPI, loaders, registry): `docs/architecture/services/sgivu-ml-components.puml`
- Modelo de datos (Pydantic requests/responses, metadata): `docs/architecture/datamodel/sgivu-ml-datamodel.puml`

## Docker

- `Dockerfile` basado en Python 3.12 slim, instala dependencias cientificas y ejecuta `run.sh`.
- En `docker-compose.dev.yml` se agregó el servicio `sgivu-ml` (puerto 8000) con volumen `sgivu-ml-models` para persistir artefactos. Variables `SGIVU_PURCHASE_SALE_URL` y `SGIVU_VEHICLE_URL` apuntan al gateway (`http://sgivu-gateway:8080`).

## Rendimiento y mejoras

- Monitoreo de drift: comparar distribucion de `sales_count` y margen vs historico; agregar alerta si MAPE > 20% en ventana movil.
- Dataset pequeno: mantener modelos ligeros (RF/XGB con profundidad limitada), aplicar smoothing en lags con rolling.
- Mejorar precision: agregar pronostico externo (feriados, macro) y modelos especializados (Prophet/SARIMA por segmento) si el historico crece.
- Multi-tenant SaaS: agregar campo `tenant_id` en dataset y entrenar modelos por tenant o con embeddings; separar rutas y modelos por namespace en `models/<tenant>/`.

## Prueba offline con CSV

- Script `tests/csv_offline_demo.py` permite entrenar y predecir sin depender de microservicios.
- Ejemplo (usando el CSV de muestra):

  ```text
  python tests/csv_offline_demo.py --csv /ruta/datos.csv --horizon 6 \
    --vehicle-type MOTORCYCLE --brand YAMAHA --model "MT-03" --line ABC-89F
  ```

- El script normaliza campos en español (Compra/Venta, Activa, Efectivo/Transferencia, Automóvil/Motocicleta), entrena con mínimo de historial y muestra pronóstico con intervalo de confianza.
- Generación de datos sintéticos: `tests/generate_contracts.py` produce CSV realistas con estacionalidad, segmentos populares y ruido; `tests/run_offline_demo.sh` entrena y genera gráfica (`tests/data/forecast.png`).
- Artefactos de prueba se guardan en `tests/models_offline/` (joblib + latest.json) sin afectar producción.
