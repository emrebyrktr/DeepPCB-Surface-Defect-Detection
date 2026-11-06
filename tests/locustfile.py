import io
from pathlib import Path

import cv2
import numpy as np
from locust import HttpUser, between, task

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_IMAGE_PATH = PROJECT_ROOT / "data/raw/test/images/01_PCB__31.jpg"


class PCBDefectLoadTester(HttpUser):
    """Endüstriyel PCB Kusur Tespiti API yük ve stres testi kullanıcısı."""

    wait_time = between(0.1, 0.5)

    def on_start(self):
        """Her sanal kullanıcı başlatıldığında test görselini belleğe önceden yükler."""
        if SAMPLE_IMAGE_PATH.exists():
            with open(SAMPLE_IMAGE_PATH, "rb") as f:
                self.image_bytes = f.read()
            self.image_filename = SAMPLE_IMAGE_PATH.name
        else:
            img = np.zeros((640, 640, 3), dtype=np.uint8)
            cv2.rectangle(img, (100, 100), (300, 300), (0, 255, 0), -1)
            _, buf = cv2.imencode(".jpg", img)
            self.image_bytes = buf.tobytes()
            self.image_filename = "synthetic_pcb.jpg"

    @task(5)
    def test_predict_endpoint(self):
        """Asıl çıkarım boru hattına görsel yükleyerek yük testi yapar."""
        files = {"file": (self.image_filename, io.BytesIO(self.image_bytes), "image/jpeg")}
        with self.client.post(
            "/predict",
            files=files,
            name="/predict",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("success") is True:
                    response.success()
                else:
                    response.failure(f"Prediction başarısız: {data}")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")

    @task(1)
    def test_health_endpoint(self):
        """Servis sağlık kontrolü yük testi."""
        with self.client.get("/health", name="/health", catch_response=True) as response:
            if response.status_code == 200 and response.json().get("status") == "healthy":
                response.success()
            else:
                response.failure(f"Sağlık kontrolü başarısız: {response.text}")
