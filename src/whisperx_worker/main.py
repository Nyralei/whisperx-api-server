import asyncio
import contextlib
import json
import logging

from whisperx_api_server.backends.registry import (
    get_alignment_backend,
    get_diarization_backend,
    get_transcription_backend,
    resolve_stage_backends,
)
from whisperx_api_server.dependencies import get_config
from whisperx_api_server.logger import setup_logger
import whisperx_api_server.s3_client as s3_client
from whisperx_worker.processor import process_job, serialize_result

logger = logging.getLogger(__name__)


async def run_worker() -> None:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

    config = get_config()
    setup_logger(config.log_level)

    gpu_task: "asyncio.Task | None" = None
    if config.metrics.enabled:
        from whisperx_api_server.observability.registry import setup_metrics, get_registry
        from whisperx_api_server.observability import gpu as _gpu
        from prometheus_client import start_http_server

        # creates _registry, runs _setup_gpu_instruments
        setup_metrics(config.metrics)

        if _gpu._pynvml_ok:
            def _on_gpu_task_done(task: asyncio.Task) -> None:
                if not task.cancelled() and task.exception() is not None:
                    logger.error(
                        "Worker GPU metrics poller died unexpectedly — VRAM/utilization gauges will stop updating",
                        exc_info=task.exception(),
                    )

            gpu_task = asyncio.create_task(
                _gpu._gpu_poll_loop(
                    config.metrics.gpu_poll_interval, _gpu._nvml_handle),
                name="worker-gpu-metrics-poller",
            )
            gpu_task.add_done_callback(_on_gpu_task_done)
            logger.info(
                "Worker GPU metrics poller started (interval=%ss)",
                config.metrics.gpu_poll_interval,
            )

        start_http_server(config.metrics.worker_port, registry=get_registry())
        logger.info(
            "Worker /metrics server started on port %s",
            config.metrics.worker_port,
        )

    await s3_client.init_client(config.s3)

    selected_backends = resolve_stage_backends()
    logger.info(
        f"Preloading backends: transcription={selected_backends.transcription}, "
        f"alignment={selected_backends.alignment}, diarization={selected_backends.diarization}"
    )
    try:
        await get_transcription_backend(selected_backends.transcription).preload_default()
    except Exception:
        logger.exception(
            "Failed to preload transcription backend; will load on first job")
    try:
        await get_alignment_backend(selected_backends.alignment).preload_default()
    except Exception:
        logger.exception(
            "Failed to preload alignment backend; will load on first job")
    try:
        await get_diarization_backend(selected_backends.diarization).preload_default()
    except Exception:
        logger.exception(
            "Failed to preload diarization backend; will load on first job")

    consumer = AIOKafkaConsumer(
        config.kafka.request_topic,
        bootstrap_servers=config.kafka.bootstrap_servers,
        group_id=config.kafka.consumer_group_worker,
        enable_auto_commit=False,
        max_poll_interval_ms=config.kafka.max_poll_interval_ms,
        auto_offset_reset="earliest",
    )
    producer = AIOKafkaProducer(
        bootstrap_servers=config.kafka.bootstrap_servers,
        max_request_size=config.kafka.max_message_bytes,
    )

    await consumer.start()
    await producer.start()
    logger.info(
        f"Worker ready — topic: {config.kafka.request_topic}, "
        f"group: {config.kafka.consumer_group_worker}, "
        f"brokers: {config.kafka.bootstrap_servers}"
    )

    try:
        async for msg in consumer:
            try:
                event = json.loads(msg.value)
            except Exception:
                preview = msg.value[:200] if msg.value else b""
                logger.error(
                    "Failed to parse Kafka message (offset=%s, partition=%s), skipping. "
                    "Raw value preview: %r",
                    msg.offset, msg.partition, preview,
                )
                await consumer.commit()
                continue

            job_id = event.get("job_id", "<unknown>")
            logger.info(f"Job {job_id}: received")

            # Pause consumer — process one job at a time
            consumer.pause(*consumer.assignment())

            reply: dict = {"job_id": job_id}
            try:
                result = await process_job(event)
                reply["status"] = "ok"
                reply["result"] = result
            except Exception as exc:
                logger.exception(f"Job {job_id}: failed")
                reply["status"] = "error"
                reply["error"] = str(exc)

            # Publish reply before committing offset
            await producer.send_and_wait(
                config.kafka.reply_topic,
                key=job_id.encode(),
                value=serialize_result(reply).encode(),
            )
            logger.info(
                f"Job {job_id}: reply published to {config.kafka.reply_topic}")

            s3_key = event.get("s3_key")
            if config.s3.delete_after_download and s3_key:
                try:
                    await s3_client.delete_audio(s3_key)
                except Exception:
                    logger.warning(
                        f"Job {job_id}: failed to delete S3 object {s3_key!r}")

            await consumer.commit()

            consumer.resume(*consumer.assignment())
            logger.info(f"Job {job_id}: done, consumer resumed")

    finally:
        if gpu_task is not None:
            gpu_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await gpu_task
            logger.info("Worker GPU metrics poller stopped")
        await consumer.stop()
        await producer.stop()
        await s3_client.close_client()
        logger.info("Worker shut down")


if __name__ == "__main__":
    asyncio.run(run_worker())
