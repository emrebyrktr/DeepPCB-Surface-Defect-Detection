from fastapi import APIRouter

from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.monitoring import router as monitoring_router
from app.api.v1.endpoints.predict import router as predict_router

api_v1_router = APIRouter()

api_v1_router.include_router(health_router, tags=["Health"])
api_v1_router.include_router(predict_router, tags=["Inference"])
api_v1_router.include_router(monitoring_router, tags=["Monitoring (Roadmap)"])

