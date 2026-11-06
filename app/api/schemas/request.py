from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float = Field(..., description="Kutucuk sol-üst X piksel koordinatı")
    y1: float = Field(..., description="Kutucuk sol-üst Y piksel koordinatı")
    x2: float = Field(..., description="Kutucuk sağ-alt X piksel koordinatı")
    y2: float = Field(..., description="Kutucuk sağ-alt Y piksel koordinatı")


class Detection(BaseModel):
    box: BoundingBox
    confidence: float = Field(..., ge=0.0, le=1.0, description="Tespit güven skoru (0-1)")
    class_id: int = Field(..., description="Kusur sınıf ID'si (0-5)")
    class_name: str = Field(..., description="Kusur sınıf adı (open, short, mousebite vb.)")

