import numpy as np

from evaluation.metrics import binary_metrics


def test_binary_metrics_perfect_prediction() -> None:
    truth = np.array([[1, 0], [0, 1]])
    report = binary_metrics(truth, truth.copy())
    assert report["dice"] == 1.0
    assert report["iou"] == 1.0


def test_binary_metrics_exclude_pixels_outside_field_of_view() -> None:
    truth = np.array([[1, 0], [0, 0]])
    prediction = np.array([[1, 0], [0, 1]])
    report = binary_metrics(truth, prediction, np.array([[1, 1], [1, 0]]))
    assert report["dice"] == 1.0
