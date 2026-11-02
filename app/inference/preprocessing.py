"""Input decoding and tensor conversion; matches the original inference path."""
from __future__ import annotations

import cv2
import numpy as np
import torch


def decode_rgb(content: bytes) -> np.ndarray:
    encoded = np.frombuffer(content, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("The uploaded file is not a readable image.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def image_to_tensor(image: np.ndarray, image_size: int, device: torch.device) -> torch.Tensor:
    resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(resized.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device) / 255.0
