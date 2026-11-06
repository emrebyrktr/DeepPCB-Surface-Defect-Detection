from pathlib import Path

from fastapi import HTTPException, UploadFile, status

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def validate_image_extension(file: UploadFile) -> None:
    """Yüklenen dosya uzantısını kontrol eder."""
    if file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext and ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Desteklenmeyen dosya uzantisi: {ext}.",
            )

