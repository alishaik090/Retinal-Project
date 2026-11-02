"""Original field-of-view-aware binary segmentation metrics."""
from __future__ import annotations

import numpy as np


def binary_metrics(target: np.ndarray, prediction: np.ndarray, valid_mask: np.ndarray | None = None) -> dict[str, float]:
    target = target.astype(bool).ravel()
    prediction = prediction.astype(bool).ravel()
    if valid_mask is not None:
        valid = valid_mask.astype(bool).ravel()
        target, prediction = target[valid], prediction[valid]
    tp = int(np.logical_and(target, prediction).sum())
    fp = int(np.logical_and(~target, prediction).sum())
    fn = int(np.logical_and(target, ~prediction).sum())
    tn = int(np.logical_and(~target, ~prediction).sum())
    eps = 1e-8
    return {
        "dice": float((2 * tp + eps) / (2 * tp + fp + fn + eps)),
        "iou": float((tp + eps) / (tp + fp + fn + eps)),
        "precision": float((tp + eps) / (tp + fp + eps)),
        "recall": float((tp + eps) / (tp + fn + eps)),
        "specificity": float((tn + eps) / (tn + fp + eps)),
        "pixel_accuracy": float((tp + tn + eps) / (tp + tn + fp + fn + eps)),
    }


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
