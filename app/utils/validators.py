from fastapi import HTTPException, UploadFile, status

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


def validate_image_file(file: UploadFile) -> None:
    """Yüklenen dosyanın geçerli bir görsel olduğunu doğrular."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Gecersiz dosya formati. Lutfen bir gorsel (jpg, png) yukleyin.",
        )

    if file.filename:
        from pathlib import Path
        ext = Path(file.filename).suffix.lower()
        if ext and ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Desteklenmeyen dosya uzantisi: {ext}. Desteklenenler: {', '.join(ALLOWED_EXTENSIONS)}",
            )

