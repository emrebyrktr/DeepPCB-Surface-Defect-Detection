import asyncio
from pathlib import Path
import pytest
import cv2
import numpy as np
from httpx import ASGITransport, AsyncClient

from src.api.app import app
from src.core.dummy_detector import DummyDetector
from src.core.onnx_detector import ONNXDetector


@pytest.fixture(autouse=True)
def setup_detector():
    """Test calismadan once app.state.detector nesnesini ayarlar."""
    if not hasattr(app.state, "detector") or app.state.detector is None:
        model_path = Path("models/pcb_defect_best.onnx")
        if model_path.exists():
            app.state.detector = ONNXDetector()
        else:
            app.state.detector = DummyDetector()


def run_async(coro):
    return asyncio.run(coro)


def test_root_endpoint():
    async def _test():
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/")
            assert resp.status_code == 200
            data = resp.json()
            assert "docs" in data
            assert "health" in data
    run_async(_test())


def test_health_endpoint():
    async def _test():
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "healthy"
            assert "classes" in data
            assert len(data["classes"]) == 6
    run_async(_test())


def test_predict_invalid_content_type():
    async def _test():
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            files = {"file": ("test.txt", b"plain text data", "text/plain")}
            resp = await client.post("/predict", files=files)
            assert resp.status_code == 400
            assert "Gecersiz dosya formati" in resp.json()["detail"]
    run_async(_test())


def test_predict_corrupted_image():
    async def _test():
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            files = {"file": ("corrupt.jpg", b"corrupted bytes", "image/jpeg")}
            resp = await client.post("/predict", files=files)
            assert resp.status_code == 400
            assert "okunamadi" in resp.json()["detail"]
    run_async(_test())


def test_predict_synthetic_image():
    async def _test():
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            img = np.zeros((640, 640, 3), dtype=np.uint8)
            cv2.rectangle(img, (100, 100), (300, 300), (0, 255, 0), -1)
            _, encoded = cv2.imencode(".jpg", img)
            files = {"file": ("synthetic.jpg", encoded.tobytes(), "image/jpeg")}

            resp = await client.post("/predict", files=files)
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert "total_defects" in data
            assert "detections" in data
            assert "latency_ms" in data
            assert "inference" in data["latency_ms"]
    run_async(_test())

