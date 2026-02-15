import logging
import uuid
from fastapi import (
    FastAPI,
    Request
)
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from whisperx_api_server.dependencies import ApiKeyDependency, get_config
from whisperx_api_server.logger import setup_logger

from whisperx_api_server.models import (
    load_transcribe_pipeline,
    load_align_model,
    load_diarize_pipeline,
)

from whisperx_api_server.routers.misc import (
    router as misc_router,
)

from whisperx_api_server.routers.models import (
    router as models_router,
)

from whisperx_api_server.routers.transcriptions import (
    router as transcribe_router,
)


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_config()
    logger = logging.getLogger(__name__)
    if config.whisper.preload_model:
        logger.info(f"Preloading model {config.whisper.model}")
        try:
            await load_transcribe_pipeline(
                model_name=config.whisper.model,
            )
        except Exception:
            logger.exception(
                "Failed to preload model; will load on demand")
    try:
        if config.alignment.preload_model:
            if config.alignment.preload_model_name:
                logger.info(
                    f"Preloading alignment model for: {config.alignment.preload_model_name}")
                await load_align_model(config.alignment.preload_model_name)

            elif config.alignment.whitelist:
                for lang in config.alignment.whitelist:
                    logger.info(f"Preloading alignment model for: {lang}")
                    await load_align_model(lang)
    except Exception:
        logger.exception(
            "Failed to preload alignment model(s); will load on demand")
    try:
        if config.diarization.preload_model:
            logger.info(
                f"Preloading diarization model {config.diarization.model}")
            await load_diarize_pipeline(model_name=config.diarization.model)
    except Exception:
        logger.exception(
            "Failed to preload diarization model; will load on demand")

    yield


def create_app() -> FastAPI:
    config = get_config()
    setup_logger(config.log_level)
    logger = logging.getLogger(__name__)

    logger.debug(f"Config: {config}")

    dependencies = []
    if config.api_key is not None or config.api_keys_file is not None:
        dependencies.append(ApiKeyDependency)

    app = FastAPI(lifespan=lifespan)

    # Misc router is for not protected endpoints like healthcheck
    app.include_router(misc_router)

    app.include_router(models_router, dependencies=dependencies)
    app.include_router(transcribe_router, dependencies=dependencies)

    if config.allow_origins is not None:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.allow_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.add_middleware(RequestIDMiddleware)

    return app
