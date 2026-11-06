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


class PredictionParams(BaseModel):
    conf_threshold: float | None = Field(None, ge=0.0, le=1.0, description="Ozel guven esigi")
    iou_threshold: float | None = Field(None, ge=0.0, le=1.0, description="Ozel IOU NMS esigi")

