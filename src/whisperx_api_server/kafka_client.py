import asyncio
import contextlib
import json
import logging
import struct
import time
from typing import Any

from whisperx_api_server.config import KafkaConfig
from whisperx_api_server.observability import kafka as _kafka

logger = logging.getLogger(__name__)

_producer = None
_admin = None  # AIOKafkaAdminClient singleton, lives for the process lifetime
_config: KafkaConfig | None = None
_pending_jobs: dict[str, tuple[asyncio.Future, float]] = {}

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
            InvalidAudioError, UploadTooLargeError,
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


def _decode_assignment(raw: bytes | None) -> list[dict]:
    """Return per-topic partition assignments from a member_assignment blob.

    Tries the kafka-python typed decoder first (stable, handles all versions);
    falls back to a bounds-checked big-endian struct parser.
    """
    if not raw:
        return []
    try:
        from kafka.protocol.group import ConsumerProtocolMemberAssignment
        decoded = ConsumerProtocolMemberAssignment.decode(raw)
        return [
            {"topic": topic, "partitions": list(partitions)}
            for topic, partitions in decoded.assignment
        ]
    except Exception:
        logger.debug("ConsumerProtocolMemberAssignment.decode failed; using fallback parser", exc_info=True)
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
            topic = raw[offset:offset + name_len].decode("utf-8")
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
                    workers.append({
                        "member_id": m[0] if len(m) > 0 else None,
                        "client_id": m[1] if len(m) > 1 else None,
                        "host": m[2] if len(m) > 2 else None,
                        "assignments": _decode_assignment(m[4] if len(m) > 4 else None),
                    })
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
    ]
    try:
        await _admin.create_topics(topics)
        logger.info(
            "Kafka topics ensured: %s, %s (partitions=%d, rf=%d)",
            cfg.request_topic, cfg.reply_topic,
            cfg.topic_partitions, cfg.topic_replication_factor,
        )
    except TopicAlreadyExistsError:
        logger.info(
            "Kafka topics already exist: %s, %s",
            cfg.request_topic, cfg.reply_topic,
        )
    except Exception:
        # Non-fatal: if topic creation fails (e.g. permissions) the broker's
        # auto.create.topics.enable still covers the produce path, and the
        # /info endpoint surfaces the resulting empty-assignment state.
        logger.warning("Kafka topic pre-creation failed", exc_info=True)


async def start(cfg: KafkaConfig) -> None:
    global _producer, _admin, _config, _discovery_lock
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


async def stop() -> None:
    global _producer, _admin, _discovery_cache, _discovery_lock
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
    job_id: str, s3_key: str, filename: str, params: dict[str, Any]
) -> asyncio.Future:
    if _producer is None or _config is None:
        raise RuntimeError("Kafka producer not initialized")
    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()
    _pending_jobs[job_id] = (future, time.monotonic())
    _kafka.pending_jobs.set(len(_pending_jobs))

    event = {
        "job_id": job_id,
        "s3_key": s3_key,
        "filename": filename,
        "params": params,
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
    logger.debug(f"Job {job_id}: published to {_config.request_topic}")
    return future


async def reply_consumer_loop(cfg: KafkaConfig) -> None:
    from aiokafka import AIOKafkaConsumer

    consumer_kwargs: dict[str, Any] = dict(
        bootstrap_servers=cfg.bootstrap_servers,
        group_id=cfg.reply_group_id,
        auto_offset_reset="latest",
        enable_auto_commit=False,
        fetch_max_bytes=cfg.max_message_bytes,
        max_partition_fetch_bytes=cfg.max_message_bytes,
    )
    if cfg.reply_group_instance_id:
        consumer_kwargs["group_instance_id"] = cfg.reply_group_instance_id

    consumer = AIOKafkaConsumer(cfg.reply_topic, **consumer_kwargs)
    await consumer.start()
    logger.info(
        "Kafka reply consumer started (group: %s, instance: %s, topic: %s)",
        cfg.reply_group_id,
        cfg.reply_group_instance_id or "<dynamic>",
        cfg.reply_topic,
    )
    try:
        async for msg in consumer:
            try:
                event = json.loads(msg.value)
            except Exception:
                logger.warning(
                    "Reply consumer: failed to parse message, skipping")
                await consumer.commit()
                continue

            job_id = event.get("job_id")
            if job_id:
                entry = _pending_jobs.pop(job_id, None)
                _kafka.pending_jobs.set(len(_pending_jobs))
                if entry is not None:
                    future, submit_time = entry
                    if not future.done():
                        duration = time.monotonic() - submit_time
                        if event.get("status") == "ok":
                            future.set_result(event["result"])
                            _kafka.job_duration.labels(status="ok").observe(duration)
                            logger.debug(f"Job {job_id}: resolved from reply")
                        else:
                            future.set_exception(_rehydrate_worker_error(
                                event.get("error_type"),
                                event.get("error", "worker error"),
                            ))
                            _kafka.job_duration.labels(status="error").observe(duration)
                            logger.warning(
                                f"Job {job_id}: failed with error: {event.get('error')}")

            try:
                await consumer.commit()
            except Exception:
                logger.exception(
                    "Reply consumer: failed to commit offset for job %s", job_id
                )
    finally:
        await consumer.stop()
        logger.info("Kafka reply consumer stopped")
