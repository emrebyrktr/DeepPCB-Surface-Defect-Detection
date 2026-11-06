from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.monitoring import router as monitoring_router
from app.api.v1.endpoints.predict import router as predict_router

__all__ = ["health_router", "predict_router", "monitoring_router"]

