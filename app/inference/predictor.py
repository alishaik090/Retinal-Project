"""Singleton-ready prediction facade around the immutable trained model."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from app.config.settings import Settings
from app.model import LoadedSegmentationModel, ModelLoadError, load_segmentation_model

from .postprocessing import mask_to_base64_png, threshold_mask
from .preprocessing import decode_rgb, image_to_tensor


@dataclass(frozen=True)
class Prediction:
    prediction: str
    confidence: float
    segmentation_mask: str
    processing_time: str
    model_name: str
    threshold: float
    vessel_pixel_fraction: float


class RetinalPredictor:
    """Loads weights exactly once and preserves the original image preprocessing."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.loaded: LoadedSegmentationModel | None = None
        self.load_error: str | None = None
        self.threshold = 0.5

    def start(self) -> None:
        try:
            self.loaded = load_segmentation_model(self.settings.model_path, self.settings.device)
            self.threshold = _load_threshold(self.settings.metrics_path, self.threshold)
            self.load_error = None
        except ModelLoadError as error:
            self.loaded = None
            self.load_error = str(error)

    @property
    def available(self) -> bool:
        return self.loaded is not None

    def predict(self, content: bytes) -> Prediction:
        if not self.loaded:
            raise ModelLoadError(self.load_error or "The model is unavailable.")
        started = time.perf_counter()
        image = decode_rgb(content)
        tensor = image_to_tensor(image, self.loaded.image_size, self.loaded.device)
        with torch.inference_mode():
            probability = torch.sigmoid(self.loaded.model(tensor))[0, 0].cpu().numpy()
        mask = threshold_mask(probability, self.threshold)
        elapsed = time.perf_counter() - started
        return Prediction(
            prediction="retinal_vessel_segmentation",
            confidence=float(probability.mean()),
            segmentation_mask=mask_to_base64_png(mask),
            processing_time=f"{elapsed:.3f} sec",
            model_name=self.loaded.model_name,
            threshold=self.threshold,
            vessel_pixel_fraction=float(mask.mean()),
        )


def _load_threshold(metrics_path: Path, fallback: float) -> float:
    if not metrics_path.is_file():
        return fallback
    try:
        value = json.loads(metrics_path.read_text(encoding="utf-8")).get("threshold", fallback)
        return float(value) if value is not None else fallback
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return fallback
