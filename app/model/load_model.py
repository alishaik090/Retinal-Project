"""Load a compatible checkpoint once at application startup."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .model import build_segmenter


class ModelLoadError(RuntimeError):
    """Raised when a configured model cannot be loaded safely."""


@dataclass(frozen=True)
class LoadedSegmentationModel:
    model: torch.nn.Module
    device: torch.device
    image_size: int
    architecture: str
    model_name: str


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ModelLoadError("DEVICE=cuda was requested but CUDA is unavailable.")
    return torch.device(requested)


def load_segmentation_model(path: Path, device_name: str) -> LoadedSegmentationModel:
    if not path.is_file():
        raise ModelLoadError(f"Model checkpoint was not found: {path}")
    device = _resolve_device(device_name)
    try:
        checkpoint: dict[str, Any] = torch.load(path, map_location=device)
        architecture = str(checkpoint.get("architecture", "unet"))
        model = build_segmenter(architecture, int(checkpoint.get("base_channels", 24))).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return LoadedSegmentationModel(
            model=model,
            device=device,
            image_size=int(checkpoint["image_size"]),
            architecture=architecture,
            model_name=str(checkpoint.get("model", model.__class__.__name__)),
        )
    except (KeyError, RuntimeError, ValueError, TypeError) as error:
        raise ModelLoadError(f"Unable to load model checkpoint: {error}") from error
