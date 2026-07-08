"""Per-message Kafka handling: idempotency, poison-job DLQ, and reply publish.

Kept separate from the run loop in ``main.py`` so the redelivery / claim / DLQ
logic is testable with fakes instead of a live broker.
"""

import asyncio
import contextlib
import json
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import whisperx_api_server.s3_client as s3_client
from whisperx_api_server import webhook
from whisperx_api_server.observability import kafka as _kafka
from whisperx_worker.processor import process_job, serialize_result
from whisperx_worker.progress import publish_heartbeat, publish_stage

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


async def _deliver_callback(
    event: dict, ctx: WorkerContext, job_id: str, envelope: str
) -> None:
    """Fire the completion webhook for a freshly-produced envelope, best-effort.

    Only called on the fresh-processing and DLQ paths, never on the marker-resend
    path, so a receiver is notified at most once per completion.
    """
    url = event.get("callback_url")
    if not url:
        return
    ok = await webhook.deliver_result(url, envelope, ctx.config)
    if ok:
        logger.info("Job %s: completion webhook delivered", job_id)
    else:
        logger.warning("Job %s: completion webhook not delivered", job_id)


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
    await _deliver_callback(event, ctx, job_id, envelope)
    await _delete_claim(ctx, job_id)
    await ctx.commit()
    _kafka.dlq_total.inc()


_LEASE_DEFER_SECONDS = 30.0


async def _resend_cached(ctx: WorkerContext, job_id: str, cached: str) -> None:
    """Resend a stored terminal envelope for a redelivered, already-run job."""
    await ctx.producer.send_and_wait(
        ctx.config.kafka.reply_topic, key=job_id.encode(), value=cached.encode()
    )
    await _delete_claim(ctx, job_id)
    await ctx.commit()
    _kafka.idempotent_skip_total.inc()


async def _defer_leased_job(event: dict, ctx: WorkerContext) -> None:
    """Another worker holds this job's lease (a rebalance redelivered its
    uncommitted message mid-run). Park briefly, then requeue the copy to the
    back of its key-hashed partition and commit this delivery — it keeps coming
    back until the result appears (resend path) or the lease expires (takeover).
    """
    job_id = event.get("job_id", "<unknown>")
    logger.info("Job %s: lease held by another worker — deferring", job_id)
    _kafka.lease_deferred_total.inc()
    await asyncio.sleep(_LEASE_DEFER_SECONDS)
    cached = await ctx.s3.get_result(job_id)
    if cached is not None:
        logger.info("Job %s: result appeared while deferring, resending reply", job_id)
        await _resend_cached(ctx, job_id, cached)
        return
    await ctx.producer.send_and_wait(
        ctx.config.kafka.request_topic,
        key=job_id.encode(),
        value=json.dumps(event).encode(),
    )
    await ctx.commit()


async def _job_liveness_loop(ctx: WorkerContext, job_id: str) -> None:
    """Periodic proof-of-life while a job runs: renew the S3 processing lease
    (keeps redelivered duplicates deferring) and publish a heartbeat progress
    event (keeps the API's inactivity timeout from reaping jobs whose current
    stage outlasts it). Runs until cancelled at the job's terminal step.
    """
    ttl = ctx.config.kafka.job_lease_ttl_seconds
    interval = ttl / 5.0
    lease_held = True
    while True:
        await asyncio.sleep(interval)
        await publish_heartbeat(ctx.producer, ctx.config.kafka.progress_topic, job_id)
        if not lease_held:
            continue
        try:
            if not await ctx.s3.renew_job_lease(job_id, ctx.worker_id, ttl):
                lease_held = False
                logger.error(
                    "Job %s: processing lease lost mid-run; continuing — a "
                    "duplicate runner may exist, results deduplicate via the "
                    "stored envelope",
                    job_id,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "Job %s: lease renewal failed; will retry", job_id, exc_info=True
            )


async def handle_message(event: dict, ctx: WorkerContext) -> None:
    job_id = event.get("job_id", "<unknown>")
    kafka_cfg = ctx.config.kafka

    # A stored envelope means a prior delivery already ran this job. Resend it
    # and commit without reprocessing or counting another attempt.
    cached = await ctx.s3.get_result(job_id)
    if cached is not None:
        logger.info("Job %s: result already stored, resending reply", job_id)
        await _resend_cached(ctx, job_id, cached)
        return

    # Take the processing lease before any work: a live lease elsewhere means a
    # concurrent run is in flight — defer instead of duplicating it. Acquisition
    # counts this delivery attempt, so a job that kills the worker mid-process
    # is still counted and eventually retired to the DLQ.
    acquired, attempts = await ctx.s3.acquire_job_lease(
        job_id, ctx.worker_id, kafka_cfg.job_lease_ttl_seconds
    )
    if not acquired:
        await _defer_leased_job(event, ctx)
        return
    if attempts > kafka_cfg.max_delivery_attempts:
        await _route_to_dlq(event, ctx, attempts)
        return

    # The previous holder may have finished between the cached check and the
    # acquire, leaving a fresh lease over an already-completed job.
    cached = await ctx.s3.get_result(job_id)
    if cached is not None:
        logger.info("Job %s: result stored by previous holder, resending", job_id)
        await _resend_cached(ctx, job_id, cached)
        return

    renewer = asyncio.create_task(_job_liveness_loop(ctx, job_id))
    try:
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

        await _deliver_callback(event, ctx, job_id, envelope)
        await _delete_claim(ctx, job_id)
        await ctx.commit()
    finally:
        renewer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await renewer


_REBALANCE_COMMIT_ERRORS = frozenset({"CommitFailedError", "UnknownMemberIdError"})


async def commit_safely(commit: Callable[[], Awaitable[None]]) -> None:
    """Commit offsets, tolerating the group having rebalanced this consumer out.

    The result is already durable in S3 and the reply already sent, so a commit
    lost to a rebalance is recovered by idempotent redelivery — a crash is not.
    Matched by class name to avoid importing aiokafka into this torch-free module.
    """
    try:
        await commit()
    except Exception as exc:
        if type(exc).__name__ not in _REBALANCE_COMMIT_ERRORS:
            raise
        logger.warning(
            "Offset commit skipped — consumer was rebalanced out of the group; "
            "idempotent redelivery will re-commit"
        )


_HANDOFF_THROTTLE_SECONDS = 1.0


def _foreign_partitions(consumer: Any, topic: str) -> set[int]:
    """Partitions of `topic` owned by other members of the consumer group."""
    all_partitions = consumer.partitions_for_topic(topic) or set()
    owned = {tp.partition for tp in consumer.assignment() if tp.topic == topic}
    return all_partitions - owned


async def _relay_queued_message(
    msg: Any,
    tp: Any,
    consumer: Any,
    ctx: WorkerContext,
    current_job_id: str,
    paused_for_job: list[bool],
) -> None:
    """Republish a message fetched mid-job so an idle worker picks it up instead
    of it waiting behind this worker's in-flight job.

    The original is never committed here: a mid-job commit would also commit the
    in-flight job's offset and break its crash-redelivery. It stays uncommitted
    until the job's terminal commit; a crash before then redelivers it and the
    claims/results idempotency layer deduplicates against the relayed copy.
    """
    if msg.value is None:
        return
    try:
        event = json.loads(msg.value)
    except Exception:
        preview = msg.value[:200]
        logger.error(
            "Unparseable message fetched mid-job (offset=%s, partition=%s), "
            "dropping. Raw value preview: %r",
            msg.offset,
            msg.partition,
            preview,
        )
        return
    job_id = event.get("job_id", "<unknown>")
    if job_id == current_job_id:
        # Relaying would start a concurrent duplicate on an idle worker; the
        # in-flight run stores the result this delivery would need anyway.
        logger.info("Job %s: duplicate delivery during own run, dropped", job_id)
        return
    event["handoff_hops"] = hops = int(event.get("handoff_hops", 0)) + 1
    request_topic = ctx.config.kafka.request_topic
    foreign = _foreign_partitions(consumer, request_topic)
    if not foreign:
        # Group shrank to just this worker mid-job: requeue onto our own log
        # (key-hashed, no explicit partition) and stop pulling until job end.
        paused_for_job[0] = True
        consumer.pause(*consumer.assignment())
    target = random.choice(sorted(foreign)) if foreign else None
    try:
        await ctx.producer.send_and_wait(
            request_topic,
            key=msg.key,
            value=json.dumps(event).encode(),
            partition=target,
        )
    except Exception:
        logger.warning(
            "Job %s: handoff republish failed; rewinding to retry",
            job_id,
            exc_info=True,
        )
        # Best-effort: if the partition was revoked meanwhile, its new owner
        # refetches the message anyway.
        with contextlib.suppress(Exception):
            consumer.seek(tp, msg.offset)
        return
    _kafka.handoff_total.inc()
    logger.info(
        "Job %s: handed off to partition %s while busy (hop %d)",
        job_id,
        target,
        hops,
    )


async def consume_loop(
    consumer: Any,
    ctx: WorkerContext,
    shutdown_event: asyncio.Event,
    paused_for_job: list[bool],
) -> None:
    """Drain the request topic one job at a time until shutdown is requested.

    While a job runs the loop keeps polling: each getmany resets aiokafka's
    fetcher-idle clock, so a job outlasting max_poll_interval_ms no longer evicts
    the worker from the group. Messages fetched during the job are relayed to a
    partition owned by another worker rather than queueing behind this job; a
    worker that owns every partition pauses instead (paused_for_job gates the
    rebalance listener's pause re-apply).

    The shutdown flag is only checked between jobs, so a SIGTERM mid-job lets the
    in-flight job finish (reply + commit) before the loop exits.
    """
    while not shutdown_event.is_set():
        records = await consumer.getmany(timeout_ms=1000, max_records=1)
        if not records:
            continue

        messages = [m for msgs in records.values() for m in msgs]
        for msg in messages:
            if msg.value is None:
                logger.warning(
                    "Empty Kafka message value (offset=%s, partition=%s), skipping",
                    msg.offset,
                    msg.partition,
                )
                await ctx.commit()
                continue
            try:
                event = json.loads(msg.value)
            except Exception:
                preview = msg.value[:200] if msg.value else b""
                logger.error(
                    "Failed to parse Kafka message (offset=%s, partition=%s), skipping. "
                    "Raw value preview: %r",
                    msg.offset,
                    msg.partition,
                    preview,
                )
                await ctx.commit()
                continue

            job_id = event.get("job_id", "<unknown>")
            logger.info("Job %s: received", job_id)

            if not _foreign_partitions(consumer, ctx.config.kafka.request_topic):
                paused_for_job[0] = True
                consumer.pause(*consumer.assignment())
            task = asyncio.create_task(handle_message(event, ctx))
            try:
                while not task.done():
                    queued = await consumer.getmany(timeout_ms=1000, max_records=1)
                    for q_tp, q_msgs in queued.items():
                        for q_msg in q_msgs:
                            await _relay_queued_message(
                                q_msg, q_tp, consumer, ctx, job_id, paused_for_job
                            )
                            await asyncio.sleep(_HANDOFF_THROTTLE_SECONDS)
                await task
            finally:
                paused_for_job[0] = False
                consumer.resume(*consumer.assignment())
                logger.info("Job %s: done, consumer resumed", job_id)

    logger.info("Worker shutdown requested, exiting message loop")
