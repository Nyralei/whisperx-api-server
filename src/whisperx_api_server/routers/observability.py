"""Observability HTTP router: /healthcheck lives here; /metrics is registered
conditionally from main.create_app() when config.metrics.enabled is True;
/info is registered in create_app() with the same auth dependencies as the
rest of the API.
"""

import logging
import time
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Literal

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from whisperx_api_server.config import (
    DistributedMode,
    MediaType,
)
from whisperx_api_server.dependencies import get_config

logger = logging.getLogger(__name__)

_STARTED_AT = time.monotonic()

router = APIRouter()
info_router = APIRouter()


def _app_version() -> str:
    try:
        return _pkg_version("whisperx-api-server")
    except PackageNotFoundError:
        return "unknown"


@router.get(
    "/healthcheck",
    description="Check the health of the API server",
    tags=["Observability"],
)
def health_check():
    return JSONResponse(
        content={"status": "healthy"}, media_type=MediaType.APPLICATION_JSON
    )


@info_router.get(
    "/info",
    description=(
        "Application version, mode, and operational status. "
        "In kafka mode, reports live worker count and queue saturation. "
        "Add ?detail=full for extended Kafka topology (bootstrap address, per-worker assignments)."
    ),
    tags=["Observability"],
)
async def info(
    request: Request, detail: Literal["summary", "full"] = Query(default="summary")
):
    config = get_config()
    payload: dict = {
        "version": _app_version(),
        "mode": config.mode.value,
        "uptime_seconds": round(time.monotonic() - _STARTED_AT, 1),
    }

    if config.mode == DistributedMode.KAFKA:
        from whisperx_api_server import kafka_client

        discovery = await kafka_client.describe_workers()
        pending = kafka_client.pending_count()
        max_pending = config.kafka.max_pending_jobs
        admin_ok = discovery["error_type"] is None
        kafka_block: dict = {
            "request_topic": config.kafka.request_topic,
            "reply_topic": config.kafka.reply_topic,
            "worker_group_id": config.kafka.consumer_group_worker,
            "pending_jobs": pending,
            "max_pending_jobs": max_pending,
            "pending_saturation": round(pending / max_pending, 3)
            if max_pending
            else None,
            "healthy_worker_count": len(discovery["workers"]),
            "kafka_admin_ok": admin_ok,
        }
        if not admin_ok:
            # Summary: expose only error type so broker addresses don't leak.
            # Full: include the full message (gated behind detail=full).
            kafka_block["error"] = discovery["error_type"]
        if detail == "full":
            kafka_block["bootstrap_servers"] = config.kafka.bootstrap_servers
            if not admin_ok:
                kafka_block["error"] = (
                    f"{discovery['error_type']}: {discovery['error_message']}"
                )
            kafka_block["workers"] = [
                {
                    "member_id": w["member_id"],
                    "client_id": w["client_id"],
                    "host": w["host"],
                    "assigned_partition_count": sum(
                        len(a["partitions"]) for a in w["assignments"]
                    ),
                    "assignments": w["assignments"],
                }
                for w in discovery["workers"]
            ]
        payload["kafka"] = kafka_block

    else:
        from whisperx_api_server import transcriber
        from whisperx_api_server.backends import registry

        # Semaphore-derived in-flight count. None semaphore = unlimited (0 limit).
        sem = transcriber._get_concurrency_semaphore()
        max_concurrent = config.max_concurrent_transcriptions
        if sem is None:
            concurrent = {"max": max_concurrent, "unlimited": True}
        else:
            available = sem._value
            in_use = max_concurrent - available
            concurrent = {
                "in_use": in_use,
                "available": available,
                "max": max_concurrent,
                "saturation": round(in_use / max_concurrent, 3)
                if max_concurrent
                else None,
            }

        # Loaded model names per stage (empty lists before first transcription).
        try:
            selection = registry.resolve_stage_backends()
            loaded_models: dict = {
                "transcription": registry.get_transcription_backend(
                    selection.transcription
                ).list_loaded_models(),
                "alignment": registry.get_alignment_backend(
                    selection.alignment
                ).list_loaded_models(),
                "diarization": registry.get_diarization_backend(
                    selection.diarization
                ).list_loaded_models(),
            }
        except Exception as e:
            logger.warning("loaded_models enumeration failed", exc_info=True)
            loaded_models = {"error": f"{type(e).__name__}: {e}"}

        payload["backends"] = {
            "transcription": config.backends.transcription,
            "alignment": config.backends.alignment,
            "diarization": config.backends.diarization,
        }
        payload["concurrent_transcriptions"] = concurrent
        payload["loaded_models"] = loaded_models
        payload["shutting_down"] = bool(request.app.state.shutting_down)
        payload["http_in_flight"] = int(request.app.state.inflight)

    return JSONResponse(content=payload, media_type=MediaType.APPLICATION_JSON)
