import time
from collections.abc import Callable
from functools import wraps
from typing import Any

from app.utils.logger import setup_logger

logger = setup_logger("timing")


def timeit(func: Callable) -> Callable:
    """Fonksiyonun çalışma süresini milisaniye cinsinden loglar."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.debug(f"{func.__name__} calisma suresi: {elapsed_ms:.2f} ms")
        return result
    return wrapper
