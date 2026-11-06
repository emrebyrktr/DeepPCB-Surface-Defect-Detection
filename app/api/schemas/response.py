from pydantic import BaseModel, Field

from app.api.schemas.request import Detection


class LatencyInfo(BaseModel):
    preprocess: float = Field(..., description="Ön işleme süresi (ms)")
    inference: float = Field(..., description="Model çıkarım süresi (ms)")
    postprocess: float = Field(..., description="Ardıl işleme ve NMS süresi (ms)")
    total: float = Field(..., description="Toplam süre (ms)")


class DetectionResponse(BaseModel):
    success: bool = True
    total_defects: int = Field(..., description="Tespit edilen toplam kusur sayısı")
    detections: list[Detection] = Field(default_factory=list, description="Kusur listesi")
    latency_ms: LatencyInfo
    filename: str | None = None


class HealthResponse(BaseModel):
    status: str = Field(default="healthy", description="Servis çalışma durumu")
    model_name: str
    model_version: str
    device: str
    classes: list[str]

