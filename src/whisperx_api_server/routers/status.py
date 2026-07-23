"""Request status router: GET /v1/audio/transcriptions/{request_id}/status
and the SSE push variant /events."""

import asyncio
import json
import logging
import re
import time

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from whisperx_api_server import request_status
from whisperx_api_server.config import MediaType
from whisperx_api_server.dependencies import get_config

logger = logging.getLogger(__name__)

_TERMINAL = ("completed", "failed")

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


@router.get(
    "/v1/audio/transcriptions/{request_id}/events",
    description=(
        "Server-Sent Events stream of processing status for a transcription "
        "request — one `data:` event per state change, `: ping` heartbeats while "
        "idle, and stream close on terminal state, client disconnect, or timeout. "
        "Replaces client-side polling of /status. 404 if the request is not tracked."
    ),
    tags=["Transcription"],
)
async def stream_transcription_events(request_id: str, request: Request):
    if not _REQUEST_ID_SAFE.match(request_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid request_id format.",
        )
    if request_status.get(request_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Request not tracked (unknown, expired, or never received).",
        )

    cfg = get_config().request_status

    async def event_stream():
        last_payload: str | None = None
        last_emit = time.monotonic()
        deadline = last_emit + cfg.sse_max_duration_seconds
        while True:
            snap = request_status.get(request_id)
            if snap is None:
                yield "event: expired\ndata: {}\n\n"
                return
            payload = json.dumps(snap, sort_keys=True)
            now = time.monotonic()
            if payload != last_payload:
                last_payload = payload
                last_emit = now
                yield f"data: {json.dumps(snap)}\n\n"
            elif now - last_emit >= cfg.sse_heartbeat_seconds:
                last_emit = now
                yield ": ping\n\n"
            if snap.get("status") in _TERMINAL:
                return
            if now >= deadline:
                yield "event: timeout\ndata: {}\n\n"
                return
            if await request.is_disconnected():
                return
            await asyncio.sleep(cfg.sse_poll_interval_seconds)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
