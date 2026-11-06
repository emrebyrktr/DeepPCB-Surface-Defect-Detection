from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import DetectionResponse, HealthResponse
from src.core import ONNXDetector
from src.utils import get_app_config

app_config = get_app_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Sunucu baslarken modeli belleğe bir kez yükler
    app.state.detector = ONNXDetector()
    yield


app = FastAPI(
    title=app_config.get("title", "PCB Defect Detection API"),
    version=app_config.get("version", "1.0.0"),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root():
    return {
        "service": app.title,
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    detector: ONNXDetector = app.state.detector
    return HealthResponse(
        status="healthy",
        model_name="DeepPCB-YOLOv8",
        model_version=app.version,
        device="cpu",
        classes=detector.classes,
    )


@app.post("/predict", response_model=DetectionResponse, tags=["Inference"])
async def predict_defects(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Gecersiz dosya formati. Lutfen bir gorsel (jpg, png) yukleyin.",
        )

    file_bytes = await file.read()
    np_array = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Gorsel verisi okunamadi veya bozuk.",
        )

    detector: ONNXDetector = app.state.detector
    output = detector.predict(image)

    return DetectionResponse(
        success=True,
        total_defects=output["total_defects"],
        detections=output["detections"],
        latency_ms=output["latency_ms"],
        filename=file.filename,
    )
