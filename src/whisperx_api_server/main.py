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

from whisperx_api_server.backends.registry import (
    get_alignment_backend,
    get_diarization_backend,
    get_transcription_backend,
    resolve_stage_backends,
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
    logger = logging.getLogger(__name__)
    selected_backends = resolve_stage_backends()
    logger.info(
        f"Configured stage backends: transcription={selected_backends.transcription}, "
        f"alignment={selected_backends.alignment}, diarization={selected_backends.diarization}"
    )
    try:
        transcription_backend = get_transcription_backend(
            selected_backends.transcription)
        await transcription_backend.preload_default()
    except Exception:
        logger.exception(
            "Failed to preload transcription backend; will try to load on demand")
    try:
        alignment_backend = get_alignment_backend(selected_backends.alignment)
        await alignment_backend.preload_default()
    except Exception:
        logger.exception(
            "Failed to preload alignment backend; will try to load on demand")
    try:
        diarization_backend = get_diarization_backend(
            selected_backends.diarization)
        await diarization_backend.preload_default()
    except Exception:
        logger.exception(
            "Failed to preload diarization backend; will try to load on demand")

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
