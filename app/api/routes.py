"""REST endpoints for health and one-image prediction."""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.inference import RetinalPredictor
from app.model import ModelLoadError

from .health import health_response
from .schemas import HealthResponse, PredictionResponse


router = APIRouter()
ACCEPTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/tiff", "image/bmp", "image/webp"}


def _predictor(request: Request) -> RetinalPredictor:
    return request.app.state.predictor


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    return health_response(_predictor(request))


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, image: UploadFile = File(...)) -> PredictionResponse:
    predictor = _predictor(request)
    if not predictor.available:
        raise HTTPException(status_code=503, detail=predictor.load_error or "Model is not loaded.")
    if image.content_type not in ACCEPTED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, TIFF, BMP, or WebP image.")
    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(content) > predictor.settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="The uploaded image exceeds the configured size limit.")
    try:
        return PredictionResponse(**predictor.predict(content).__dict__)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except ModelLoadError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="Prediction failed. Consult service logs for the request context.") from error
