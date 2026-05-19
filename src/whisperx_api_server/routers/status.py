"""Request status router: GET /v1/audio/transcriptions/{request_id}/status."""

import logging
import re

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from whisperx_api_server import request_status
from whisperx_api_server.config import MediaType

logger = logging.getLogger(__name__)

# Mirror src/whisperx_api_server/main.py:_REQUEST_ID_SAFE so a malformed ID
# can't poison the not-found log line with control chars.
_REQUEST_ID_SAFE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

router = APIRouter()


@router.get(
    "/v1/audio/transcriptions/{request_id}/status",
    description=(
        "Live processing status for a transcription request. Clients must set "
        "X-Request-ID on the POST to learn the ID before the response arrives. "
        "Returns 404 once the entry expires (see request_status.ttl_seconds)."
    ),
    tags=["Transcription"],
)
async def get_transcription_status(request_id: str):
    if not _REQUEST_ID_SAFE.match(request_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid request_id format.",
        )
    state = request_status.get(request_id)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Request not tracked (unknown, expired, or never received).",
        )
    return JSONResponse(content=state, media_type=MediaType.APPLICATION_JSON)
