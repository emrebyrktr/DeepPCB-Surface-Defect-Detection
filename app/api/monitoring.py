from time import perf_counter

from fastapi import FastAPI, Request
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator

from app.utils.logger import setup_logger

logger = setup_logger("monitoring")
DETECTIONS_TOTAL = Counter(
    "pcb_detections_total",
    "Total number of detected PCB defects by class.",
    ["class_name"],
)

def setup_monitoring(app: FastAPI) -> None:
    Instrumentator(excluded_handlers=["/metrics"]).instrument(app).expose(app)
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = perf_counter()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            logger.info(
                "method=%s path=%s status=%s duration_ms=%.2f",
                request.method,
                request.url.path,
                status_code,
                (perf_counter() - start) * 1000,
            )
