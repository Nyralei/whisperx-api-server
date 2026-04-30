"""Observability HTTP router: /healthcheck lives here; /metrics is registered
conditionally from main.create_app() when config.metrics.enabled is True
"""

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from whisperx_api_server.config import (
    MediaType,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/healthcheck",
    description="Check the health of the API server",
    tags=["Observability"],
)
def health_check():
    return JSONResponse(content={"status": "healthy"}, media_type=MediaType.APPLICATION_JSON)
