from pydantic import BaseModel, Field

from app.api.schemas.request import Detection


class LatencyInfo(BaseModel):
    preprocess: float = Field(..., description="On isleme suresi (ms)")
    inference: float = Field(..., description="Model cikarim suresi (ms)")
    postprocess: float = Field(..., description="Ardil isleme ve NMS suresi (ms)")
    total: float = Field(..., description="Toplam sure (ms)")


class DetectionResponse(BaseModel):
    success: bool = True
    total_defects: int = Field(..., description="Tespit edilen toplam kusur sayisi")
    detections: list[Detection] = Field(default_factory=list, description="Kusur listesi")
    latency_ms: LatencyInfo
    filename: str | None = None
    io_read_ms: float | None = None


class HealthResponse(BaseModel):
    status: str = Field(default="healthy", description="Servis calisma durumu")
    model_name: str
    model_version: str
    device: str
    classes: list[str]


class MonitoringResponse(BaseModel):
    status: str
    uptime_seconds: float
    total_requests: int
    active_model: str
    device: str
