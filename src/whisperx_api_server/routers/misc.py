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
    tags=["Misc"],
)
def health_check():
    return JSONResponse(content={"status": "healthy"}, media_type=MediaType.APPLICATION_JSON)
