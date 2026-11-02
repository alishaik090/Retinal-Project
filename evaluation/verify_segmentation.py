"""Recalculate preserved DRIVE held-out metrics without retraining the model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from app.inference.postprocessing import threshold_mask
from app.inference.preprocessing import image_to_tensor
from app.model import load_segmentation_model
from evaluation.metrics import binary_metrics, mean_metrics


def _case_paths(drive_dir: Path, case_id: str) -> tuple[Path, Path, Path]:
    image = next((drive_dir / "training/images").glob(f"{case_id}_training.*"))
    mask = next((drive_dir / "training/1st_manual").glob(f"{case_id}_manual1.*"))
    field = next((drive_dir / "training/mask").glob(f"{case_id}_training_mask.*"))
    return image, mask, field


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--drive-dir", type=Path, required=True)
    parser.add_argument("--metrics-path", type=Path, required=True, help="Saved metrics JSON containing threshold and test split IDs.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    saved = json.loads(args.metrics_path.read_text(encoding="utf-8"))
    threshold = float(saved["threshold"])
    loaded = load_segmentation_model(args.model_path, args.device)
    rows: list[dict[str, float]] = []
    for case_id in saved["splits"]["test"]:
        image_path, mask_path, field_path = _case_paths(args.drive_dir, case_id)
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not read {image_path}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = image_to_tensor(image, loaded.image_size, loaded.device)
        with torch.inference_mode():
            probability = torch.sigmoid(loaded.model(tensor))[0, 0].cpu().numpy()
        truth = cv2.resize(np.asarray(Image.open(mask_path).convert("L")), (loaded.image_size, loaded.image_size), interpolation=cv2.INTER_NEAREST) > 127
        valid = cv2.resize(np.asarray(Image.open(field_path).convert("L")), (loaded.image_size, loaded.image_size), interpolation=cv2.INTER_NEAREST) > 127
        row = binary_metrics(truth, threshold_mask(probability, threshold), valid)
        row["id"] = case_id
        rows.append(row)
    aggregate = mean_metrics([{key: value for key, value in row.items() if key != "id"} for row in rows])
    print(json.dumps({"threshold": threshold, "test_metrics": aggregate, "per_test_image": rows}, indent=2))


if __name__ == "__main__":
    main()
