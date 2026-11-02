"""Health response construction."""
from __future__ import annotations

from app.inference import RetinalPredictor

from .schemas import HealthResponse


def health_response(predictor: RetinalPredictor) -> HealthResponse:
    if predictor.available and predictor.loaded:
        return HealthResponse(status="healthy", model="loaded", model_name=predictor.loaded.model_name)
    return HealthResponse(status="degraded", model="unavailable", detail=predictor.load_error or "Model initialization has not completed.")
