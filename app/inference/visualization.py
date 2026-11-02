"""Optional server-side visualization helpers; no client/UI code."""
from __future__ import annotations

import cv2
import numpy as np


def blend_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = cv2.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)
    overlay = base.copy()
    overlay[mask.astype(bool)] = np.array([0, 235, 120], dtype=np.uint8)
    return cv2.addWeighted(base, 0.62, overlay, 0.38, 0)
