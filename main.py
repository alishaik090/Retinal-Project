"""ASGI entry point for the standalone Retina inference service."""
from __future__ import annotations

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from app.api.routes import router
from app.config import Settings
from app.inference import RetinalPredictor
from app.utils.logging import configure_logging


def create_app(settings: Settings | None = None) -> FastAPI:
    runtime_settings = settings or Settings.from_environment()
    configure_logging(runtime_settings.log_level)
    logger = logging.getLogger(__name__)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        predictor = RetinalPredictor(runtime_settings)
        predictor.start()
        application.state.predictor = predictor
        if predictor.available:
            logger.info("model_loaded name=%s path=%s", predictor.loaded.model_name if predictor.loaded else "unknown", runtime_settings.model_path)
        else:
            logger.error("model_unavailable detail=%s", predictor.load_error)
        yield

    application = FastAPI(title="Retina Inference Service", version="1.0.0", lifespan=lifespan)
    application.include_router(router)
    return application


app = create_app()
