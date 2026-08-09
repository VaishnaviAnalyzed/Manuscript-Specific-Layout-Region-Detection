"""API integration tests."""

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.api import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_detect_endpoint() -> None:
    image = np.full((320, 480, 3), 245, dtype=np.uint8)
    cv2.putText(image, "MANUSCRIPT", (80, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 10, 10), 3)
    success, encoded = cv2.imencode(".png", image)
    assert success

    response = client.post(
        "/api/detect",
        files={"file": ("page.png", encoded.tobytes(), "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["source_filename"] == "page.png"
    assert body["region_count"] >= 1
    assert body["annotated_image_url"].startswith("/results/")

