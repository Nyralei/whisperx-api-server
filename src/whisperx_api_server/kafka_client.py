import asyncio
import json
import logging
import socket
from typing import Any

from whisperx_api_server.config import KafkaConfig

logger = logging.getLogger(__name__)

_producer = None
_config: KafkaConfig | None = None
_pending_jobs: dict[str, asyncio.Future] = {}


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
    assert _producer is not None and _config is not None, "Kafka producer not initialized"
    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()
    _pending_jobs[job_id] = future

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

    group_id = f"whisperx-api-reply-{socket.gethostname()}"
    consumer = AIOKafkaConsumer(
        cfg.reply_topic,
        bootstrap_servers=cfg.bootstrap_servers,
        group_id=group_id,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        fetch_max_bytes=cfg.max_message_bytes,
        max_partition_fetch_bytes=cfg.max_message_bytes,
    )
    await consumer.start()
    logger.info(
        f"Kafka reply consumer started (group: {group_id}, topic: {cfg.reply_topic})")
    try:
        async for msg in consumer:
            try:
                event = json.loads(msg.value)
            except Exception:
                logger.warning(
                    "Reply consumer: failed to parse message, skipping")
                continue

            job_id = event.get("job_id")
            if not job_id:
                continue

            future = _pending_jobs.pop(job_id, None)
            if future is None or future.done():
                continue

            if event.get("status") == "ok":
                future.set_result(event["result"])
                logger.debug(f"Job {job_id}: resolved from reply")
            else:
                future.set_exception(RuntimeError(
                    event.get("error", "worker error")))
                logger.warning(
                    f"Job {job_id}: failed with error: {event.get('error')}")
    finally:
        await consumer.stop()
        logger.info("Kafka reply consumer stopped")
