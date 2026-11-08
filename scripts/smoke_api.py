"""Check a running API's readiness and real inference path using only stdlib."""

import argparse
import json
import struct
import time
import urllib.error
import urllib.request
import uuid
import zlib


def make_png(size: int = 640) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return struct.pack("!I", len(payload)) + kind + payload + struct.pack("!I", checksum)

    pixels = (b"\x00" + b"\x00" * (size * 3)) * size
    header = struct.pack("!2I5B", size, size, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b"")


def smoke_api(base_url: str, timeout: float, expected_model: str | None) -> None:
    base_url = base_url.rstrip("/")
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/api/v1/health", timeout=5) as response:
                health = json.load(response)
            if health.get("status") == "healthy":
                break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
        time.sleep(1)
    else:
        raise RuntimeError(f"API did not become ready: {last_error}")

    if expected_model and health.get("model_name") != expected_model:
        raise AssertionError(f"Unexpected model: {health.get('model_name')!r}")
    if not health.get("classes"):
        raise AssertionError("Health response does not identify model classes.")

    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="ci-smoke.png"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode() + make_png() + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        f"{base_url}/api/v1/predict", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        prediction = json.load(response)
    if prediction.get("success") is not True:
        raise AssertionError("Prediction was not successful.")
    if prediction.get("total_defects") != len(prediction.get("detections", [])):
        raise AssertionError("Detection count does not match the prediction payload.")
    if expected_model == "ci-fixture":
        detections = prediction.get("detections", [])
        if len(detections) != 1 or detections[0].get("class_name") != "open":
            raise AssertionError("ONNX fixture prediction did not reach the API response.")
    print(f"Smoke passed: model={health['model_name']}, detections={prediction['total_defects']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--expected-model")
    args = parser.parse_args()
    smoke_api(args.base_url, args.timeout, args.expected_model)


if __name__ == "__main__":
    main()
