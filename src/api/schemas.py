from typing import List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float = Field(..., description="Kutucuk sol-ust X piksel koordinati")
    y1: float = Field(..., description="Kutucuk sol-ust Y piksel koordinati")
    x2: float = Field(..., description="Kutucuk sag-alt X piksel koordinati")
    y2: float = Field(..., description="Kutucuk sag-alt Y piksel koordinati")


class Detection(BaseModel):
    box: BoundingBox
    confidence: float = Field(..., ge=0.0, le=1.0, description="Tespit guven skoru (0-1)")
    class_id: int = Field(..., description="Kusur sinif ID'si (0-5)")
    class_name: str = Field(..., description="Kusur sinif adi (open, short, mousebite vb.)")


class LatencyInfo(BaseModel):
    preprocess: float = Field(..., description="On isleme suresi (ms)")
    inference: float = Field(..., description="Model cikarim suresi (ms)")
    postprocess: float = Field(..., description="Ardil isleme ve NMS suresi (ms)")
    total: float = Field(..., description="Toplam sure (ms)")


class DetectionResponse(BaseModel):
    success: bool = True
    total_defects: int = Field(..., description="Tespit edilen toplam kusur sayisi")
    detections: List[Detection] = Field(default_factory=list, description="Kusur listesi")
    latency_ms: LatencyInfo
    filename: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = Field(default="healthy", description="Servis calisma durumu")
    model_name: str
    model_version: str
    device: str
    classes: List[str]