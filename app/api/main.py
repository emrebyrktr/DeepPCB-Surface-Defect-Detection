from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.predict import router as predict_router
from app.api.v1.router import api_v1_router
from app.core.config import settings
from app.services.model_loader import ModelLoader
from app.utils.logger import setup_logger

logger = setup_logger("api_main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Sunucu başlarken modeli belleğe bir kez yükler."""
    logger.info("PCB Defect Detection API baslatiliyor...")
    app.state.detector = ModelLoader.load_detector(
        model_path=settings.MODEL_PATH,
        config_path=str(settings.config_path),
        fallback_dummy=True,
    )
    yield
    logger.info("PCB Defect Detection API kapatiliyor...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# V1 Versioned Router
app.include_router(api_v1_router, prefix=settings.API_V1_STR)

# Geriye dönük kök seviyesi yönlendirmeleri
app.include_router(health_router, tags=["Health (Root Alias)"])
app.include_router(predict_router, tags=["Inference (Root Alias)"])


@app.get("/", tags=["General"])
async def root():
    return {
        "service": app.title,
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
        "api_v1": settings.API_V1_STR,
    }
