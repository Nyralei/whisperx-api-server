import asyncio
import contextlib
import logging
import uuid
from fastapi import (
    FastAPI,
    Request
)
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from whisperx_api_server.config import DistributedMode
from whisperx_api_server.dependencies import ApiKeyDependency, get_config
from whisperx_api_server.logger import setup_logger

from whisperx_api_server.backends.registry import (
    get_alignment_backend,
    get_diarization_backend,
    get_transcription_backend,
    resolve_stage_backends,
)

from whisperx_api_server.routers.observability import (
    router as observability_router,
)

from whisperx_api_server.routers.models import (
    router as models_router,
)

from whisperx_api_server.routers.transcriptions import (
    router as transcribe_router,
)


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger = logging.getLogger(__name__)
    config = get_config()

    gpu_task: "asyncio.Task | None" = None
    if config.metrics.enabled:
        from whisperx_api_server.observability import gpu as _gpu
        if _gpu._pynvml_ok:
            def _on_gpu_task_done(task: asyncio.Task) -> None:
                if not task.cancelled() and task.exception() is not None:
                    logger.error(
                        "GPU metrics poller died unexpectedly — VRAM/utilization gauges will stop updating",
                        exc_info=task.exception(),
                    )

            gpu_task = asyncio.create_task(
                _gpu._gpu_poll_loop(
                    config.metrics.gpu_poll_interval, _gpu._nvml_handle),
                name="gpu-metrics-poller",
            )
            gpu_task.add_done_callback(_on_gpu_task_done)
            logger.info("GPU metrics poller started (interval=%ss)",
                        config.metrics.gpu_poll_interval)

    try:
        if config.mode == DistributedMode.KAFKA:
            from whisperx_api_server import s3_client, kafka_client
            await s3_client.init_client(config.s3)
            await kafka_client.start(config.kafka)

            def _on_reply_task_done(task: asyncio.Task) -> None:
                if not task.cancelled() and task.exception() is not None:
                    logger.error(
                        "Kafka reply consumer died unexpectedly — pending jobs will time out",
                        exc_info=task.exception(),
                    )

            reply_task = asyncio.create_task(
                kafka_client.reply_consumer_loop(config.kafka),
                name="kafka-reply-consumer",
            )
            reply_task.add_done_callback(_on_reply_task_done)
            logger.info("Kafka mode: S3 and Kafka clients started")
            try:
                yield
            finally:
                reply_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await reply_task
                await kafka_client.stop()
                await s3_client.close_client()
        else:
            selected_backends = resolve_stage_backends()
            logger.info(
                f"Configured stage backends: transcription={selected_backends.transcription}, "
                f"alignment={selected_backends.alignment}, diarization={selected_backends.diarization}"
            )
            try:
                transcription_backend = get_transcription_backend(
                    selected_backends.transcription)
                await transcription_backend.preload_default()
            except Exception:
                logger.exception(
                    "Failed to preload transcription backend; will try to load on demand")
            try:
                alignment_backend = get_alignment_backend(
                    selected_backends.alignment)
                await alignment_backend.preload_default()
            except Exception:
                logger.exception(
                    "Failed to preload alignment backend; will try to load on demand")
            try:
                diarization_backend = get_diarization_backend(
                    selected_backends.diarization)
                await diarization_backend.preload_default()
            except Exception:
                logger.exception(
                    "Failed to preload diarization backend; will try to load on demand")
            yield
    finally:
        # -------- GPU Task shutdown --------
        if gpu_task is not None:
            gpu_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await gpu_task
            logger.info("GPU metrics poller stopped")


def create_app() -> FastAPI:
    config = get_config()
    setup_logger(config.log_level)
    logger = logging.getLogger(__name__)

    logger.debug(f"Config: {config}")

    dependencies = []
    if config.api_key is not None or config.api_keys_file is not None:
        dependencies.append(ApiKeyDependency)

    app = FastAPI(lifespan=lifespan)

    # Observability router hosts unauthenticated /healthcheck (and /metrics when
    # METRICS_ENABLED=true — /metrics is registered directly on `app` below).
    app.include_router(observability_router)

    if config.metrics.enabled:
        from whisperx_api_server.observability.registry import (
            get_registry,
            setup_metrics,
        )

        setup_metrics(config.metrics)

        from whisperx_api_server.observability.http import MetricsMiddleware
        app.add_middleware(MetricsMiddleware)

        from fastapi import Response as FastAPIResponse
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        @app.get("/metrics", include_in_schema=False)
        def metrics_endpoint() -> FastAPIResponse:
            return FastAPIResponse(
                content=generate_latest(get_registry()),
                media_type=CONTENT_TYPE_LATEST,
            )

        logger.info(
            "Observability: /metrics endpoint registered (unauthenticated)")

    app.include_router(models_router, dependencies=dependencies)
    app.include_router(transcribe_router, dependencies=dependencies)

    if config.allow_origins is not None:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.allow_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.add_middleware(RequestIDMiddleware)

    return app
