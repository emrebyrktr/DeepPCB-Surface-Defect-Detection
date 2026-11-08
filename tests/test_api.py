import asyncio
import threading

import cv2
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from app.api import app
from app.api.dependencies import get_detector, get_inference_service
from app.core import DummyDetector
from app.services.inference_service import InferenceService


@pytest.fixture(autouse=True)
def setup_detector():
    """Unit testler yerel model dosyalarindan bagimsiz, acik bir test double kullanir."""
    detector = DummyDetector()
    previous = dict(app.dependency_overrides)
    app.dependency_overrides[get_detector] = lambda: detector
    yield detector
    app.dependency_overrides.clear()
    app.dependency_overrides.update(previous)


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
            assert "api_v1" in data
    run_async(_test())


def test_health_endpoint(setup_detector):
    setup_detector.model_name = "unit-test-model"
    setup_detector.model_version = "2.3.4"

    async def _test():
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            for ep in ["/health", "/api/v1/health"]:
                resp = await client.get(ep)
                assert resp.status_code == 200
                data = resp.json()
                assert data["status"] == "healthy"
                assert data["model_name"] == "unit-test-model"
                assert data["model_version"] == "2.3.4"
                assert data["device"] == "cpu"
                assert "classes" in data
                assert len(data["classes"]) == 6
    run_async(_test())


def test_predict_invalid_content_type():
    async def _test():
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            files = {"file": ("test.txt", b"plain text data", "text/plain")}
            resp = await client.post("/api/v1/predict", files=files)
            assert resp.status_code == 400
            assert "Gecersiz dosya formati" in resp.json()["detail"]
    run_async(_test())


def test_predict_corrupted_image():
    async def _test():
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            files = {"file": ("corrupt.jpg", b"corrupted bytes", "image/jpeg")}
            resp = await client.post("/api/v1/predict", files=files)
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

            resp = await client.post("/api/v1/predict", files=files)
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert data["total_defects"] == 1
            assert data["detections"][0]["class_name"] == "open"
            assert "total_defects" in data
            assert "detections" in data
            assert "latency_ms" in data
            assert "inference" in data["latency_ms"]
    run_async(_test())


def test_prediction_runs_outside_event_loop_thread(setup_detector, monkeypatch):
    event_loop_thread = threading.get_ident()
    predict = setup_detector.predict
    worker_threads = []

    def record_thread(image):
        worker_threads.append(threading.get_ident())
        return predict(image)

    monkeypatch.setattr(setup_detector, "predict", record_thread)

    async def _test():
        _, encoded = cv2.imencode(".png", np.zeros((32, 32, 3), dtype=np.uint8))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/predict", files={"file": ("small.png", encoded.tobytes(), "image/png")},
            )
        assert resp.status_code == 200

    run_async(_test())
    assert worker_threads
    assert all(thread != event_loop_thread for thread in worker_threads)


def test_upload_size_limit(setup_detector):
    app.dependency_overrides[get_inference_service] = lambda: InferenceService(
        setup_detector, max_upload_bytes=8,
    )

    async def _test():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/predict", files={"file": ("large.jpg", b"123456789", "image/jpeg")},
            )
        assert resp.status_code == 413
        assert "Dosya boyutu" in resp.json()["detail"]

    run_async(_test())


def test_image_pixel_limit(setup_detector):
    app.dependency_overrides[get_inference_service] = lambda: InferenceService(
        setup_detector, max_image_pixels=16,
    )

    async def _test():
        _, encoded = cv2.imencode(".png", np.zeros((5, 5, 3), dtype=np.uint8))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/predict", files={"file": ("large.png", encoded.tobytes(), "image/png")},
            )
        assert resp.status_code == 413
        assert "piksel" in resp.json()["detail"]

    run_async(_test())
