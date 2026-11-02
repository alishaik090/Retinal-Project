"""Runtime configuration loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    model_path: Path
    metrics_path: Path
    device: str
    host: str
    port: int
    log_level: str
    max_upload_bytes: int

    @classmethod
    def from_environment(cls) -> "Settings":
        load_dotenv(PROJECT_ROOT / ".env")
        return cls(
            model_path=Path(os.getenv("MODEL_PATH", str(PROJECT_ROOT / "app/model/weights/vessel_unet_best.pt"))).expanduser(),
            metrics_path=Path(os.getenv("METRICS_PATH", str(PROJECT_ROOT / "evaluation/segmentation_metrics.json"))).expanduser(),
            device=os.getenv("DEVICE", "auto").lower(),
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            max_upload_bytes=int(os.getenv("MAX_UPLOAD_MB", "20")) * 1024 * 1024,
        )
