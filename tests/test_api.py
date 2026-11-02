import asyncio
import os
from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.inference import RetinalPredictor
from main import create_app


def missing_model_settings(tmp_path: Path) -> Settings:
    return Settings(tmp_path / "missing.pt", tmp_path / "metrics.json", "cpu", "127.0.0.1", 8000, "INFO", 1024)


def request(app, method: str, url: str, **kwargs):
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.request(method, url, **kwargs)
    return asyncio.run(run())


def make_test_app(settings: Settings):
    app = create_app(settings)
    predictor = RetinalPredictor(settings)
    predictor.start()
    app.state.predictor = predictor
    return app


def test_health_reports_missing_model_without_crashing(tmp_path: Path) -> None:
    response = request(make_test_app(missing_model_settings(tmp_path)), "GET", "/health")
    assert response.status_code == 200
    assert response.json()["model"] == "unavailable"


def test_predict_returns_service_unavailable_when_model_is_missing(tmp_path: Path) -> None:
    response = request(make_test_app(missing_model_settings(tmp_path)), "POST", "/predict", files={"image": ("fundus.png", b"not-an-image", "image/png")})
    assert response.status_code == 503


def test_loaded_model_health_and_prediction_when_assets_are_supplied() -> None:
    model_path = os.getenv("RETINA_TEST_MODEL_PATH")
    image_path = os.getenv("RETINA_TEST_IMAGE_PATH")
    if not model_path or not image_path:
        pytest.skip("Set RETINA_TEST_MODEL_PATH and RETINA_TEST_IMAGE_PATH to run the checkpoint integration test.")
    settings = Settings(Path(model_path), Path(os.getenv("RETINA_TEST_METRICS_PATH", "evaluation/segmentation_metrics.json")), "cpu", "127.0.0.1", 8000, "INFO", 20 * 1024 * 1024)
    response = request(make_test_app(settings), "POST", "/predict", files={"image": (Path(image_path).name, Path(image_path).read_bytes(), "image/tiff")})
    payload = response.json()
    assert response.status_code == 200
    assert payload["model_name"] == "ResNet18UNet"
    assert payload["prediction"] == "retinal_vessel_segmentation"
    assert payload["segmentation_mask"]
