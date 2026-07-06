"""Per-message Kafka handling: idempotency, poison-job DLQ, and reply publish.

Kept separate from the run loop in ``main.py`` so the redelivery / claim / DLQ
logic is testable with fakes instead of a live broker.
"""

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import whisperx_api_server.s3_client as s3_client
from whisperx_api_server.observability import kafka as _kafka
from whisperx_worker.processor import process_job, serialize_result
from whisperx_worker.progress import publish_stage

logger = logging.getLogger(__name__)


@dataclass
class WorkerContext:
    """Dependencies a single message handler needs, injected for testability."""

    producer: Any
    config: Any
    commit: Callable[[], Awaitable[None]]
    worker_id: str
    s3: Any = s3_client


async def _publish_terminal(ctx: WorkerContext, job_id: str, reply: dict) -> None:
    if reply["status"] == "ok":
        await publish_stage(
            ctx.producer,
            ctx.config.kafka.progress_topic,
            job_id,
            "completed",
            status="completed",
        )
    else:
        await publish_stage(
            ctx.producer,
            ctx.config.kafka.progress_topic,
            job_id,
            "failed",
            status="failed",
            error=reply.get("error"),
            error_type=reply.get("error_type"),
        )


async def _delete_claim(ctx: WorkerContext, job_id: str) -> None:
    try:
        await ctx.s3.delete_claim(job_id)
    except Exception:
        logger.warning("Job %s: failed to delete claim object", job_id)


async def _route_to_dlq(event: dict, ctx: WorkerContext, attempts: int) -> None:
    job_id = event.get("job_id", "<unknown>")
    kafka_cfg = ctx.config.kafka
    reason = "max_delivery_attempts exceeded"
    logger.error(
        "Job %s: %s (attempts=%d) — routing to %s",
        job_id,
        reason,
        attempts,
        kafka_cfg.dead_letter_topic,
    )
    dlq_event = {
        **event,
        "attempts": attempts,
        "reason": reason,
        "worker_id": ctx.worker_id,
        "ts": time.time(),
    }
    await ctx.producer.send_and_wait(
        kafka_cfg.dead_letter_topic,
        key=job_id.encode(),
        value=serialize_result(dlq_event).encode(),
    )
    reply = {
        "job_id": job_id,
        "status": "error",
        "error": (
            f"Job exceeded {kafka_cfg.max_delivery_attempts} delivery attempts "
            "and was routed to the dead-letter queue"
        ),
        "error_type": "RuntimeError",
    }
    envelope = serialize_result(reply)
    await ctx.s3.put_result(job_id, envelope)
    await ctx.producer.send_and_wait(
        kafka_cfg.reply_topic, key=job_id.encode(), value=envelope.encode()
    )
    await _publish_terminal(ctx, job_id, reply)
    await _delete_claim(ctx, job_id)
    await ctx.commit()
    _kafka.dlq_total.inc()


async def handle_message(event: dict, ctx: WorkerContext) -> None:
    job_id = event.get("job_id", "<unknown>")
    kafka_cfg = ctx.config.kafka

    # A stored envelope means a prior delivery already ran this job. Resend it
    # and commit without reprocessing or counting another attempt.
    cached = await ctx.s3.get_result(job_id)
    if cached is not None:
        logger.info("Job %s: result already stored, resending reply", job_id)
        await ctx.producer.send_and_wait(
            kafka_cfg.reply_topic, key=job_id.encode(), value=cached.encode()
        )
        await _delete_claim(ctx, job_id)
        await ctx.commit()
        _kafka.idempotent_skip_total.inc()
        return

    # Count this delivery before any work, so a job that kills the worker
    # mid-process is still counted and eventually retired to the DLQ.
    attempts = await ctx.s3.increment_claim(job_id)
    if attempts > kafka_cfg.max_delivery_attempts:
        await _route_to_dlq(event, ctx, attempts)
        return

    reply: dict = {"job_id": job_id}
    timeline: dict[str, dict[str, float | None]] = {}
    try:
        result = await process_job(
            event,
            progress_producer=ctx.producer,
            progress_topic=kafka_cfg.progress_topic,
            timeline_out=timeline,
        )
        reply["status"] = "ok"
        reply["result"] = result
    except Exception as exc:
        logger.exception("Job %s: failed", job_id)
        reply["status"] = "error"
        reply["error"] = str(exc)
        # Type name lets the API rehydrate typed exceptions (e.g.
        # InvalidAudioError → HTTP 422) instead of collapsing to 500.
        reply["error_type"] = type(exc).__name__
    if timeline:
        reply["timeline"] = timeline

    envelope = serialize_result(reply)

    # Envelope before reply: a crash in between replays the job, and the
    # redelivery resends from this object rather than rerunning.
    await ctx.s3.put_result(job_id, envelope)
    await _publish_terminal(ctx, job_id, reply)
    await ctx.producer.send_and_wait(
        kafka_cfg.reply_topic, key=job_id.encode(), value=envelope.encode()
    )
    logger.info("Job %s: reply published to %s", job_id, kafka_cfg.reply_topic)

    s3_key = event.get("s3_key")
    if ctx.config.s3.delete_after_download and s3_key:
        try:
            await ctx.s3.delete_audio(s3_key)
        except Exception:
            logger.warning("Job %s: failed to delete S3 object %r", job_id, s3_key)

    await _delete_claim(ctx, job_id)
    await ctx.commit()
