import asyncio
import json
import logging
import time
from typing import Any

from whisperx_api_server.config import KafkaConfig
from whisperx_api_server.observability import kafka as _kafka

logger = logging.getLogger(__name__)

_producer = None
_config: KafkaConfig | None = None
_pending_jobs: dict[str, tuple[asyncio.Future, float]] = {}


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


async def start(cfg: KafkaConfig) -> None:
    global _producer, _config
    from aiokafka import AIOKafkaProducer

    _config = cfg
    _producer = AIOKafkaProducer(
        bootstrap_servers=cfg.bootstrap_servers,
        max_request_size=cfg.max_message_bytes,
    )
    await _producer.start()
    logger.info(f"Kafka producer started (brokers: {cfg.bootstrap_servers})")


async def stop() -> None:
    global _producer
    if _producer is not None:
        await _producer.stop()
        _producer = None
        logger.info("Kafka producer stopped")


async def submit_job(
    job_id: str, s3_key: str, filename: str, params: dict[str, Any]
) -> asyncio.Future:
    if _producer is None or _config is None:
        raise RuntimeError("Kafka producer not initialized")
    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()
    _pending_jobs[job_id] = (future, time.monotonic())

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
