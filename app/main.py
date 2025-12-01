from fastapi import FastAPI

from app.core.config import get_settings
from app.routers import prediction_router

settings = get_settings()
app = FastAPI(title=settings.app_name, version=settings.app_version)


@app.get("/health", tags=["health"])
async def health():
    """Endpoint de verificación de vida del microservicio."""
    return {"status": "ok", "version": settings.app_version}


app.include_router(prediction_router.router)
