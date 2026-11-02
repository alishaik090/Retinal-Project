"""Public API response schemas."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model: str
    model_name: str | None = None
    detail: str | None = None


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    prediction: str
    confidence: float = Field(ge=0, le=1)
    segmentation_mask: str
    processing_time: str
    model_name: str
    threshold: float = Field(ge=0, le=1)
    vessel_pixel_fraction: float = Field(ge=0, le=1)
