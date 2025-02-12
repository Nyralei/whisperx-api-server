import logging
import uuid
from fastapi import (
    FastAPI,
    Request
)
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from whisperx_api_server.dependencies import ApiKeyDependency, get_config

from whisperx_api_server.logger import setup_logger

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

def create_app() -> FastAPI:
    config = get_config()
    setup_logger(config.log_level)
    logger = logging.getLogger(__name__)

    logger.debug(f"Config: {config}")

    dependencies = []
    if config.api_key is not None:
        dependencies.append(ApiKeyDependency)

    app = FastAPI(dependencies=dependencies)

    app.include_router(misc_router)
    app.include_router(models_router)
    app.include_router(transcribe_router)

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