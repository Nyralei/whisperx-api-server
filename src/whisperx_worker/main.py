import asyncio
import contextlib
import json
import logging
import signal

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

    from whisperx_api_server.transcriber import init_concurrency
    init_concurrency()

    shutdown_event = asyncio.Event()

    def _request_worker_shutdown(signame: str) -> None:
        if shutdown_event.is_set():
            return
        shutdown_event.set()
        logger.info(
            "%s received — finishing current job (if any) then shutting down",
            signame,
        )

    sigterm_registered = False
    try:
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGTERM, _request_worker_shutdown, "SIGTERM")
        sigterm_registered = True
    except NotImplementedError:
        logger.info(
            "Worker SIGTERM handler unavailable on this platform (likely Windows)"
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

    from aiokafka import ConsumerRebalanceListener

    job_in_flight = [False]

    class _PauseAwareRebalanceListener(ConsumerRebalanceListener):
        """Restores `consumer.pause()` state across partition reassignments.

        The worker pauses the consumer for the duration of a single job to
        enforce one-job-at-a-time semantics. Without this listener, a Kafka
        rebalance (broker restart, partition reassignment, etc.) silently
        clears the paused set on the newly-assigned partitions and the
        consumer would begin pulling fresh records mid-job.
        """

        async def on_partitions_revoked(self, revoked):
            return

        async def on_partitions_assigned(self, assigned):
            if job_in_flight[0] and assigned:
                consumer.pause(*assigned)
                logger.info(
                    "Rebalance: re-applied pause on %d partition(s) for in-flight job",
                    len(assigned),
                )

    await consumer.start()
    await producer.start()
    consumer.subscribe(
        topics=[config.kafka.request_topic],
        listener=_PauseAwareRebalanceListener(),
    )
    logger.info(
        f"Worker ready — topic: {config.kafka.request_topic}, "
        f"group: {config.kafka.consumer_group_worker}, "
        f"brokers: {config.kafka.bootstrap_servers}"
    )

    try:
        while not shutdown_event.is_set():
            # Poll with a bounded timeout so SIGTERM during idle periods
            # exits the loop within ~1s.
            records = await consumer.getmany(timeout_ms=1000, max_records=1)
            if not records:
                continue

            messages = [m for msgs in records.values() for m in msgs]
            for msg in messages:
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

                # Pause consumer — process one job at a time. `job_in_flight` is set
                # before pause so the rebalance listener re-applies it if assignments
                # change while this job runs.
                job_in_flight[0] = True
                consumer.pause(*consumer.assignment())

                try:
                    reply: dict = {"job_id": job_id}
                    try:
                        result = await process_job(event)
                        reply["status"] = "ok"
                        reply["result"] = result
                    except Exception as exc:
                        logger.exception(f"Job {job_id}: failed")
                        reply["status"] = "error"
                        reply["error"] = str(exc)
                        # Pass exception type so the API can rehydrate typed
                        # exceptions (e.g. InvalidAudioError → HTTP 422) instead
                        # of collapsing every worker failure to RuntimeError/500.
                        reply["error_type"] = type(exc).__name__

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
                finally:
                    job_in_flight[0] = False
                    consumer.resume(*consumer.assignment())
                    logger.info(f"Job {job_id}: done, consumer resumed")

        logger.info("Worker shutdown requested, exiting message loop")

    finally:
        if sigterm_registered:
            with contextlib.suppress(NotImplementedError, RuntimeError):
                asyncio.get_running_loop().remove_signal_handler(signal.SIGTERM)
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
