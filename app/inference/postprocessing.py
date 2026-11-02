"""Mask thresholding and response-safe encoding."""
from __future__ import annotations

import base64

import cv2
import numpy as np


def threshold_mask(probability: np.ndarray, threshold: float) -> np.ndarray:
    return probability >= threshold


def mask_to_base64_png(mask: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", mask.astype(np.uint8) * 255)
    if not ok:
        raise RuntimeError("Could not encode the segmentation mask.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")
