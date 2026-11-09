from fastapi import APIRouter, Depends, File, UploadFile

from app.api.dependencies import get_inference_service
from app.api.monitoring import DETECTIONS_TOTAL
from app.api.schemas.response import DetectionResponse
from app.services.inference_service import InferenceService

router = APIRouter()


@router.post("/predict", response_model=DetectionResponse, summary="PCB Kusur Tespiti")
async def predict_defects(
    file: UploadFile = File(..., description="Analiz edilecek PCB görseli (JPG/PNG)"),
    inference_service: InferenceService = Depends(get_inference_service),
):
    """Yüklenen PCB görseli üzerinde 6 sınıf için nesne tespiti gerçekleştirir."""
    output = await inference_service.predict_from_upload(file)

    response = DetectionResponse(
        success=True,
        total_defects=output["total_defects"],
        detections=output["detections"],
        latency_ms=output["latency_ms"],
        filename=output.get("filename"),
    )
    for detection in response.detections:
        DETECTIONS_TOTAL.labels(class_name=detection.class_name).inc()

    return response

