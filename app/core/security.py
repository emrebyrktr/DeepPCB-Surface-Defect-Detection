import os

from fastapi import HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(api_key: str | None = Security(api_key_header)) -> str | None:
    """İsteğe bağlı API anahtarı doğrulama.

    Ortam değişkeninde API_KEY tanımlıysa kontrol eder, tanımlı değilse serbest bırakır.
    """
    expected_api_key = os.getenv("API_KEY")
    if expected_api_key:
        if not api_key or api_key != expected_api_key:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Geçersiz veya eksik API Anahtarı.",
            )
    return api_key
