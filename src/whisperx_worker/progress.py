"""Fire-and-forget per-stage progress publisher for the Kafka worker.

The worker calls publish_stage(...) at every stage boundary so the API-side
request_status tracker can surface fine-grained progress to clients polling
GET /v1/audio/transcriptions/{job_id}/status. Failure to publish never
propagates — the transcription pipeline always takes priority over progress
telemetry.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


async def publish_stage(
    producer: Any,
    topic: str,
    job_id: str,
    stage: str,
    *,
    status: str = "in_progress",
    error: str | None = None,
    error_type: str | None = None,
) -> None:
    """Publish a progress event. Best-effort — never raises."""
    if producer is None or not topic or not job_id or not stage:
        return
    event: dict[str, Any] = {
        "job_id": job_id,
        "stage": stage,
        "status": status,
        "ts": time.time(),
    }
    if error is not None:
        event["error"] = error
    if error_type is not None:
        event["error_type"] = error_type
    try:
        # send() (not send_and_wait) — we don't block the pipeline on broker ack.
        await producer.send(
            topic, key=job_id.encode(), value=json.dumps(event).encode()
        )
    except Exception:
        logger.debug(
            "progress publish failed for job %s stage %s", job_id, stage, exc_info=True
        )


async def publish_heartbeat(producer: Any, topic: str, job_id: str) -> None:
    """Prove the worker is still on the job between stage boundaries, so the
    API's inactivity timeout survives stages longer than the timeout itself
    (a multi-hour file can spend 20+ minutes inside a single stage). Stage-less
    on purpose: consumers that don't know the heartbeat status skip stage-less
    events. Best-effort — never raises."""
    if producer is None or not topic or not job_id:
        return
    event = {"job_id": job_id, "status": "heartbeat", "ts": time.time()}
    try:
        await producer.send(
            topic, key=job_id.encode(), value=json.dumps(event).encode()
        )
    except Exception:
        logger.debug("heartbeat publish failed for job %s", job_id, exc_info=True)
