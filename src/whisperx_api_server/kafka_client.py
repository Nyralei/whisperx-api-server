import asyncio
import contextlib
import importlib
import json
import logging
import os
import struct
import time
import uuid
from dataclasses import dataclass
from typing import Any

from whisperx_api_server import request_status
from whisperx_api_server.config import KafkaConfig
from whisperx_api_server.observability import kafka as _kafka

logger = logging.getLogger(__name__)


@dataclass
class _PendingJob:
    future: asyncio.Future | None
    submitted_at: float
    # Refreshed by worker progress events; reply/liveness timeouts measure
    # inactivity from here, not total elapsed time since submission.
    last_activity: float


_producer = None
_admin = None  # AIOKafkaAdminClient singleton, lives for the process lifetime
_config: KafkaConfig | None = None
_pending_jobs: dict[str, _PendingJob] = {}
_janitor_task: asyncio.Task | None = None

_discovery_cache: tuple[float, dict] | None = None
_discovery_lock: asyncio.Lock | None = None


def _rehydrate_worker_error(error_type: str | None, message: str) -> BaseException:
    """Map worker-side exception class name → API-side exception class so the
    typed-error → HTTP-code router mapping still works when the failure
    originates in a worker process. Lazy import avoids a transcriber↔kafka_client
    cycle.
    """
    if error_type:
        from whisperx_api_server.transcriber import (
            InvalidAudioError,
            UploadTooLargeError,
        )

        mapping: dict[str, type[BaseException]] = {
            "InvalidAudioError": InvalidAudioError,
            "UploadTooLargeError": UploadTooLargeError,
            "TimeoutError": TimeoutError,
            "ValueError": ValueError,
        }
        cls = mapping.get(error_type, RuntimeError)
    else:
        cls = RuntimeError
    return cls(message)


def pending_count() -> int:
    return len(_pending_jobs)


def touch_pending(job_id: str) -> None:
    """Record worker activity for a pending job, pushing its liveness deadline out."""
    entry = _pending_jobs.get(job_id)
    if entry is not None:
        entry.last_activity = time.monotonic()


def discard_pending(job_id: str) -> None:
    if _pending_jobs.pop(job_id, None) is not None:
        _kafka.pending_jobs.set(len(_pending_jobs))


async def wait_for_reply(job_id: str, future: asyncio.Future, timeout: float) -> Any:
    """Await the reply future, timing out only after `timeout` seconds of worker
    inactivity (no reply and no progress events), not total elapsed time.
    Raises TimeoutError; never cancels or discards the pending entry itself.
    """
    while not future.done():
        entry = _pending_jobs.get(job_id)
        if entry is None:
            # Entry vanished without resolving the future (reaped mid-await).
            raise TimeoutError(f"Job {job_id}: pending entry was reaped")
        remaining = entry.last_activity + timeout - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Job {job_id}: no reply or worker progress within {timeout}s"
            )
        await asyncio.wait([future], timeout=remaining)
    return future.result()


def _decode_assignment(raw: bytes | None) -> list[dict]:
    """Return per-topic partition assignments from a member_assignment blob.

    Tries the kafka-python typed decoder first (stable, handles all versions);
    falls back to a bounds-checked big-endian struct parser.
    """
    if not raw:
        return []
    try:
        # Imported dynamically: the symbol's location varies across kafka-python
        # versions (and the package may be absent), so a static import would both
        # break type-checking and hard-fail here. Any miss drops to the fallback.
        group_mod = importlib.import_module("kafka.protocol.group")
        decoded = group_mod.ConsumerProtocolMemberAssignment.decode(raw)
        return [
            {"topic": topic, "partitions": list(partitions)}
            for topic, partitions in decoded.assignment
        ]
    except Exception:
        logger.debug(
            "ConsumerProtocolMemberAssignment.decode failed; using fallback parser",
            exc_info=True,
        )
    try:
        offset = 0
        if len(raw) < 6:
            return []
        offset += 2  # skip version int16
        num_topics = struct.unpack_from(">i", raw, offset)[0]
        offset += 4
        result = []
        for _ in range(num_topics):
            if offset + 2 > len(raw):
                return result
            name_len = struct.unpack_from(">h", raw, offset)[0]
            offset += 2
            if offset + name_len + 4 > len(raw):
                return result
            topic = raw[offset : offset + name_len].decode("utf-8")
            offset += name_len
            num_parts = struct.unpack_from(">i", raw, offset)[0]
            offset += 4
            if offset + 4 * num_parts > len(raw):
                return result
            partitions = list(struct.unpack_from(f">{num_parts}i", raw, offset))
            offset += 4 * num_parts
            result.append({"topic": topic, "partitions": partitions})
        return result
    except Exception:
        logger.debug("_decode_assignment fallback parser failed", exc_info=True)
        return []


async def describe_workers(timeout: float = 5.0) -> dict:
    """Return live consumer-group membership from the Kafka admin API.

    Results are TTL-cached (config.discovery_cache_ttl_seconds). Only
    successful responses are cached — transient broker errors are never pinned,
    so the next call retries immediately.
    """
    global _discovery_cache
    if _config is None or _admin is None:
        raise RuntimeError("Kafka client not initialized")

    now = time.monotonic()
    if _discovery_cache is not None:
        cached_at, cached_payload = _discovery_cache
        if now - cached_at < _config.discovery_cache_ttl_seconds:
            return cached_payload

    lock = _discovery_lock  # set alongside _admin in start(); non-None guaranteed here
    assert lock is not None
    async with lock:
        now = time.monotonic()
        if _discovery_cache is not None:
            cached_at, cached_payload = _discovery_cache
            if now - cached_at < _config.discovery_cache_ttl_seconds:
                return cached_payload

        try:
            groups = await asyncio.wait_for(
                _admin.describe_consumer_groups([_config.consumer_group_worker]),
                timeout=timeout,
            )
        except Exception as e:
            logger.warning("describe_workers failed: %s: %s", type(e).__name__, e)
            return {
                "workers": [],
                "error_type": type(e).__name__,
                "error_message": str(e),
            }

        workers: list[dict] = []
        for response in groups or []:
            # DescribeGroupsResponse_vN: .groups is a list of plain tuples
            # (error_code, group_id, state, protocol_type, protocol, members)
            for group_tuple in getattr(response, "groups", []) or []:
                members = group_tuple[5] if len(group_tuple) > 5 else []
                for m in members or []:
                    # member tuple: (member_id, client_id, client_host, metadata, assignment)
                    workers.append(
                        {
                            "member_id": m[0] if len(m) > 0 else None,
                            "client_id": m[1] if len(m) > 1 else None,
                            "host": m[2] if len(m) > 2 else None,
                            "assignments": _decode_assignment(
                                m[4] if len(m) > 4 else None
                            ),
                        }
                    )
        result = {"workers": workers, "error_type": None, "error_message": None}
        _discovery_cache = (time.monotonic(), result)
        return result


async def _ensure_topics(cfg: KafkaConfig) -> None:
    """Pre-create request/reply topics so workers that connect before any job
    is produced still get a non-empty partition assignment. Idempotent —
    TopicAlreadyExistsError is the success path on every restart.
    """
    from aiokafka.admin import NewTopic
    from aiokafka.errors import TopicAlreadyExistsError

    assert _admin is not None  # set in start() immediately before this call
    topics = [
        NewTopic(
            name=cfg.request_topic,
            num_partitions=cfg.topic_partitions,
            replication_factor=cfg.topic_replication_factor,
        ),
        NewTopic(
            name=cfg.reply_topic,
            num_partitions=cfg.topic_partitions,
            replication_factor=cfg.topic_replication_factor,
        ),
        NewTopic(
            name=cfg.progress_topic,
            num_partitions=cfg.topic_partitions,
            replication_factor=cfg.topic_replication_factor,
        ),
        NewTopic(
            name=cfg.dead_letter_topic,
            num_partitions=1,
            replication_factor=cfg.topic_replication_factor,
        ),
    ]
    try:
        await _admin.create_topics(topics)
        logger.info(
            "Kafka topics ensured: %s, %s, %s, %s (partitions=%d, rf=%d)",
            cfg.request_topic,
            cfg.reply_topic,
            cfg.progress_topic,
            cfg.dead_letter_topic,
            cfg.topic_partitions,
            cfg.topic_replication_factor,
        )
    except TopicAlreadyExistsError:
        logger.info(
            "Kafka topics already exist: %s, %s, %s, %s",
            cfg.request_topic,
            cfg.reply_topic,
            cfg.progress_topic,
            cfg.dead_letter_topic,
        )
    except Exception:
        # Non-fatal: if topic creation fails (e.g. permissions) the broker's
        # auto.create.topics.enable still covers the produce path, and the
        # /info endpoint surfaces the resulting empty-assignment state.
        logger.warning("Kafka topic pre-creation failed", exc_info=True)


async def start(cfg: KafkaConfig) -> None:
    global _producer, _admin, _config, _discovery_lock, _janitor_task
    from aiokafka import AIOKafkaProducer
    from aiokafka.admin import AIOKafkaAdminClient

    _config = cfg
    _discovery_lock = asyncio.Lock()

    _admin = AIOKafkaAdminClient(bootstrap_servers=cfg.bootstrap_servers)
    await _admin.start()
    logger.info("Kafka admin client started (brokers: %s)", cfg.bootstrap_servers)

    await _ensure_topics(cfg)

    _producer = AIOKafkaProducer(
        bootstrap_servers=cfg.bootstrap_servers,
        max_request_size=cfg.max_message_bytes,
    )
    await _producer.start()
    logger.info("Kafka producer started (brokers: %s)", cfg.bootstrap_servers)

    _janitor_task = asyncio.create_task(
        _pending_janitor_loop(cfg), name="kafka-pending-janitor"
    )


async def stop() -> None:
    global _producer, _admin, _discovery_cache, _discovery_lock, _janitor_task
    if _janitor_task is not None:
        _janitor_task.cancel()
        try:
            await _janitor_task
        except asyncio.CancelledError:
            pass
        _janitor_task = None
        logger.info("Kafka pending-jobs janitor stopped")
    if _admin is not None:
        with contextlib.suppress(Exception):
            await _admin.close()
        _admin = None
        logger.info("Kafka admin client stopped")
    if _producer is not None:
        await _producer.stop()
        _producer = None
        logger.info("Kafka producer stopped")
    _discovery_cache = None
    _discovery_lock = None


async def submit_job(
    job_id: str,
    s3_key: str | None,
    audio_url: str | None,
    filename: str,
    params: dict[str, Any],
    *,
    track_future: bool = True,
    callback_url: str | None = None,
) -> asyncio.Future | None:
    """Publish a job to the request topic. With track_future (sync path) a future
    is registered for the reply consumer to resolve; without it (async path) the
    entry holds no awaiter — the result is fetched from S3 — and the janitor reaps
    it if no reply arrives.
    """
    if _producer is None or _config is None:
        raise RuntimeError("Kafka producer not initialized")
    if bool(s3_key) == bool(audio_url):
        raise ValueError("submit_job requires exactly one of s3_key or audio_url")
    future: asyncio.Future | None = None
    if track_future:
        future = asyncio.get_running_loop().create_future()
    now = time.monotonic()
    _pending_jobs[job_id] = _PendingJob(future, now, now)
    _kafka.pending_jobs.set(len(_pending_jobs))

    event = {
        "job_id": job_id,
        "s3_key": s3_key,
        "audio_url": audio_url,
        "filename": filename,
        "params": params,
        "callback_url": callback_url,
    }
    try:
        await _producer.send_and_wait(
            _config.request_topic,
            key=job_id.encode(),
            value=json.dumps(event).encode(),
        )
    except Exception:
        _pending_jobs.pop(job_id, None)
        _kafka.pending_jobs.set(len(_pending_jobs))
        raise
    logger.debug("Job %s: published to %s", job_id, _config.request_topic)
    return future


async def publish_submitted(
    cfg: KafkaConfig,
    job_id: str,
    *,
    filename: str | None = None,
    params: dict[str, Any] | None = None,
) -> None:
    """Announce a new job on the progress topic so other replicas start tracking
    it before the first worker stage arrives. Best-effort — never raises."""
    if _producer is None:
        return
    event = {
        "job_id": job_id,
        "stage": "submitted",
        "status": "submitted",
        "filename": filename,
        "params": params,
        "ts": time.time(),
    }
    try:
        await _producer.send(
            cfg.progress_topic, key=job_id.encode(), value=json.dumps(event).encode()
        )
    except Exception:
        logger.debug("submitted-event publish failed for job %s", job_id, exc_info=True)


def _reap_stale_pending(cfg: KafkaConfig, now: float | None = None) -> int:
    """Drop pending jobs with no reply and no worker activity; return the count reaped.

    Staleness is measured from last_activity (refreshed by progress events), so a
    worker grinding through long audio is never reaped while it reports stages.
    The +60s margin past reply_timeout_seconds means this never races a live
    awaiter (which discards its own entry on timeout). It only catches entries
    with no awaiter (async submits) or genuine leaks.
    """
    cutoff = (now if now is not None else time.monotonic()) - (
        cfg.reply_timeout_seconds + 60.0
    )
    stale = [
        job_id
        for job_id, entry in _pending_jobs.items()
        if entry.last_activity < cutoff
    ]
    for job_id in stale:
        entry = _pending_jobs.pop(job_id, None)
        if entry is None:
            continue
        msg = f"Job {job_id} reaped: no reply or worker progress received in time"
        if entry.future is not None and not entry.future.done():
            entry.future.set_exception(TimeoutError(msg))
        request_status.mark_failed(job_id, msg, "TimeoutError")
        _kafka.pending_reaped_total.inc()
    if stale:
        _kafka.pending_jobs.set(len(_pending_jobs))
    return len(stale)


async def _pending_janitor_loop(cfg: KafkaConfig) -> None:
    try:
        while True:
            await asyncio.sleep(60.0)
            try:
                n = _reap_stale_pending(cfg)
                if n:
                    logger.warning("Reaped %d pending job(s) with no worker reply", n)
            except Exception:
                logger.exception("Pending-jobs janitor sweep failed")
    except asyncio.CancelledError:
        raise


def _handle_reply_event(event: dict[str, Any]) -> None:
    """Resolve the pending future for a single decoded reply event."""
    job_id = event.get("job_id")
    if not job_id:
        return

    # Timeline first so a trailing mark_completed runs against finalized stages.
    timeline = event.get("timeline")
    if timeline:
        request_status.apply_worker_timeline(job_id, timeline)

    is_ok = event.get("status") == "ok"
    entry = _pending_jobs.pop(job_id, None)
    _kafka.pending_jobs.set(len(_pending_jobs))
    if entry is not None:
        # future is None for async submits (no awaiter); popping the entry above
        # is the only cleanup needed — the result is already durable in S3.
        if entry.future is not None and not entry.future.done():
            if is_ok:
                entry.future.set_result(event["result"])
                logger.debug("Job %s: resolved from reply", job_id)
            else:
                entry.future.set_exception(
                    _rehydrate_worker_error(
                        event.get("error_type"),
                        event.get("error", "worker error"),
                    )
                )
                logger.warning(
                    "Job %s: failed with error: %s", job_id, event.get("error")
                )
        _kafka.job_duration.labels(status="ok" if is_ok else "error").observe(
            time.monotonic() - entry.submitted_at
        )
    else:
        state = request_status.get(job_id)
        if state is not None and state.get("local"):
            # Submitted here but the future is gone (timed out / reaped / awaiter
            # cancelled). Replies for jobs owned by other replicas land on
            # non-local stubs and are ignored — counting those would just be
            # fan-out noise.
            _kafka.late_reply_total.inc()
            if is_ok:
                logger.warning(
                    "Job %s: reply arrived after the future was gone; "
                    "status reconciled — result available via /result",
                    job_id,
                )
            else:
                logger.warning(
                    "Job %s: reply arrived after the future was gone", job_id
                )

    # The reply is the authoritative terminal signal: it settles status even when
    # no awaiter remains (async submits, cancelled/timed-out sync awaiters) and
    # flips a failed-by-timeout state back to completed — the result envelope is
    # durable in S3 regardless of who is still listening.
    if is_ok:
        request_status.reconcile_completed(job_id)
    else:
        request_status.mark_failed(
            job_id, event.get("error", "worker error"), event.get("error_type")
        )


async def reply_consumer_loop(cfg: KafkaConfig) -> None:
    from aiokafka import AIOKafkaConsumer

    # Unique group per replica so the broker fans every reply out to all of
    # them; the holder of the future resolves it, the rest no-op.
    group_id = f"{cfg.reply_group_id}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    consumer = AIOKafkaConsumer(
        cfg.reply_topic,
        bootstrap_servers=cfg.bootstrap_servers,
        group_id=group_id,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        fetch_max_bytes=cfg.max_message_bytes,
        max_partition_fetch_bytes=cfg.max_message_bytes,
    )
    await consumer.start()
    logger.info(
        "Kafka reply consumer started (group: %s, topic: %s)",
        group_id,
        cfg.reply_topic,
    )
    try:
        async for msg in consumer:
            if msg.value is None:
                continue
            try:
                event = json.loads(msg.value)
            except Exception:
                logger.warning("Reply consumer: failed to parse message, skipping")
                continue
            _handle_reply_event(event)
    finally:
        await consumer.stop()
        logger.info("Kafka reply consumer stopped")


async def progress_consumer_loop(cfg: KafkaConfig) -> None:
    """Consume per-stage worker progress events and update the request_status tracker.

    Each API replica subscribes with a unique consumer group id (prefix + pid + rand)
    so every replica receives every event from the broker. A replica upserts a stub
    entry for any job_id it does not already track, so a replica that did not submit
    the job (or started mid-job) still converges its status view — this is what makes
    GET /status work behind a load balancer.

    Best-effort: parse errors are logged and skipped; commit failures are ignored
    (we use enable_auto_commit so this is a no-op anyway).
    """
    from aiokafka import AIOKafkaConsumer

    group_id = f"{cfg.progress_group_id_prefix}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    consumer = AIOKafkaConsumer(
        cfg.progress_topic,
        bootstrap_servers=cfg.bootstrap_servers,
        group_id=group_id,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        fetch_max_bytes=cfg.max_message_bytes,
        max_partition_fetch_bytes=cfg.max_message_bytes,
    )
    await consumer.start()
    logger.info(
        "Kafka progress consumer started (group: %s, topic: %s)",
        group_id,
        cfg.progress_topic,
    )
    try:
        async for msg in consumer:
            if msg.value is None:
                continue
            try:
                event = json.loads(msg.value)
            except Exception:
                logger.warning("Progress consumer: failed to parse message, skipping")
                continue
            _handle_progress_event(event)
    finally:
        await consumer.stop()
        logger.info("Kafka progress consumer stopped")


def _handle_progress_event(event: dict[str, Any]) -> None:
    job_id = event.get("job_id")
    stage = event.get("stage")
    status_val = event.get("status")
    if not job_id:
        return

    # Job announcement from the submitting replica — create/enrich the stub.
    if status_val == "submitted":
        request_status.ensure_tracked(
            job_id,
            filename=event.get("filename"),
            params=event.get("params"),
        )
        return

    # Worker-originated event: proof of life. Push the liveness deadline
    # out so long-running jobs that keep signalling are never timed out.
    touch_pending(job_id)

    # Heartbeats exist only for the touch above — no stage bookkeeping.
    if status_val == "heartbeat":
        return

    if not stage:
        return
    # Upsert so replicas that didn't submit the job still track it.
    request_status.ensure_tracked(job_id)

    try:
        if status_val == "failed":
            request_status.mark_failed(
                job_id,
                event.get("error", "worker error"),
                event.get("error_type"),
            )
        elif status_val == "completed":
            # Defensive — reply path also calls mark_completed; idempotent.
            request_status.mark_completed(job_id)
        else:
            request_status.set_stage(job_id, f"worker.{stage}")
    except Exception:
        logger.exception("Progress consumer: failed to apply event for job %s", job_id)
