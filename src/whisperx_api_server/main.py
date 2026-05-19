import asyncio
import contextlib
import logging
import re
import signal
import uuid
from fastapi import (
    FastAPI,
    Request
)
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Scope, Receive, Send

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
    info_router,
    router as observability_router,
)

from whisperx_api_server.routers.models import (
    router as models_router,
)

from whisperx_api_server.routers.transcriptions import (
    router as transcribe_router,
)

from whisperx_api_server.routers.status import (
    router as status_router,
)

from whisperx_api_server import request_status


_REQUEST_ID_SAFE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Validate client-supplied X-Request-ID so log lines / response headers
        # cannot be poisoned with CR/LF or other control chars.
        client_id = request.headers.get("X-Request-ID")
        if client_id and _REQUEST_ID_SAFE.match(client_id):
            request_id = client_id
        else:
            request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class GracefulShutdownMiddleware:
    """ASGI middleware that refuses new requests with HTTP 503 while the app
    is draining, and tracks an in-flight counter on `app.state` so the
    lifespan can await an empty queue before tearing down.

    Health/metrics endpoints stay reachable during drain so liveness probes
    do not flip the pod to CrashLoopBackoff before drain finishes.
    """

    _DRAIN_PASSTHROUGH_PATHS: frozenset[str] = frozenset(
        {"/metrics", "/healthcheck"}
    )

    def __init__(self, app: ASGIApp, app_state) -> None:
        self.app = app
        self.app_state = app_state

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path: str = scope.get("path", "")
        if path in self._DRAIN_PASSTHROUGH_PATHS:
            await self.app(scope, receive, send)
            return

        if getattr(self.app_state, "shutting_down", False):
            response = JSONResponse(
                {"detail": "Server is shutting down."},
                status_code=503,
                headers={"Connection": "close", "Retry-After": "5"},
            )
            await response(scope, receive, send)
            return

        self.app_state.inflight += 1
        try:
            await self.app(scope, receive, send)
        finally:
            self.app_state.inflight -= 1
            if self.app_state.inflight <= 0:
                drain_event: asyncio.Event | None = getattr(
                    self.app_state, "drain_event", None
                )
                if drain_event is not None:
                    drain_event.set()


async def _drain_inflight(app_state, grace_seconds: int, logger: logging.Logger) -> None:
    """Wait up to `grace_seconds` for the in-flight counter to reach zero."""
    if app_state.inflight <= 0:
        return

    drain_event = asyncio.Event()
    app_state.drain_event = drain_event
    if app_state.inflight <= 0:
        return

    logger.info(
        "Draining %d in-flight request(s) (grace=%ds)",
        app_state.inflight,
        grace_seconds,
    )
    try:
        await asyncio.wait_for(drain_event.wait(), timeout=grace_seconds)
        logger.info("All in-flight requests drained")
    except asyncio.TimeoutError:
        logger.warning(
            "Drain grace expired with %d request(s) still in flight",
            app_state.inflight,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger = logging.getLogger(__name__)
    config = get_config()

    app.state.shutting_down = False
    app.state.inflight = 0
    app.state.drain_event = None

    sigterm_registered = False
    chained_handler = None
    try:
        loop = asyncio.get_running_loop()
        # Capture and chain to uvicorn's existing SIGTERM handler so its
        # graceful shutdown still fires after we flip the drain flag.
        prev = getattr(loop, "_signal_handlers", {}).get(signal.SIGTERM)
        if prev is not None and not getattr(prev, "_cancelled", False):
            chained_handler = (prev._callback, prev._args)

        def _request_shutdown() -> None:
            if not app.state.shutting_down:
                app.state.shutting_down = True
                logger.info(
                    "SIGTERM received — refusing new requests, draining in-flight work"
                )
            if chained_handler is not None:
                cb, args = chained_handler
                try:
                    cb(*args)
                except Exception:
                    logger.exception("Previous SIGTERM handler raised")

        loop.add_signal_handler(signal.SIGTERM, _request_shutdown)
        sigterm_registered = True
    except NotImplementedError:
        logger.info(
            "SIGTERM handler unavailable on this platform; relying on lifespan teardown"
        )

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

    from whisperx_api_server.transcriber import init_concurrency
    from whisperx_api_server.executors import shutdown_executors
    init_concurrency()

    def _on_status_cleanup_done(task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception() is not None:
            logger.error(
                "request_status cleanup task died unexpectedly — terminal entries will no longer be evicted",
                exc_info=task.exception(),
            )

    status_cleanup_task = asyncio.create_task(
        request_status.cleanup_loop(),
        name="request-status-cleanup",
    )
    status_cleanup_task.add_done_callback(_on_status_cleanup_done)

    if config.alignment.nltk_preload:
        try:
            import nltk
            await asyncio.to_thread(nltk.download, "punkt_tab", quiet=True)
            logger.info("NLTK punkt_tab preloaded")
        except Exception:
            logger.warning("NLTK punkt_tab preload failed; will be downloaded on first alignment request")

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

            def _on_progress_task_done(task: asyncio.Task) -> None:
                if not task.cancelled() and task.exception() is not None:
                    logger.error(
                        "Kafka progress consumer died unexpectedly — per-stage status updates will stop",
                        exc_info=task.exception(),
                    )

            progress_task = asyncio.create_task(
                kafka_client.progress_consumer_loop(config.kafka),
                name="kafka-progress-consumer",
            )
            progress_task.add_done_callback(_on_progress_task_done)

            logger.info("Kafka mode: S3 and Kafka clients started")
            try:
                yield
            finally:
                reply_task.cancel()
                progress_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await reply_task
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await progress_task
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
        # Flip the drain flag before teardown so any final in-flight requests
        # see HTTP 503. If SIGTERM already fired, this is a no-op.
        if not app.state.shutting_down:
            app.state.shutting_down = True
        await _drain_inflight(app.state, config.shutdown_grace_seconds, logger)

        if sigterm_registered:
            with contextlib.suppress(NotImplementedError, RuntimeError):
                asyncio.get_running_loop().remove_signal_handler(signal.SIGTERM)

        shutdown_executors()
        # -------- Request-status cleanup task shutdown --------
        status_cleanup_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await status_cleanup_task
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
    app.include_router(info_router, dependencies=dependencies)

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

    # Model-management endpoints operate on the API process's local cache.
    # In Kafka mode the API process never loads models (the worker does), so
    # /models/*, /align_models/*, /diarize_models/* would either lie about
    # cache state or wastefully load models into the API container. Skip them
    # entirely so callers see a clean 404 instead of misleading 200/500.
    if config.mode != DistributedMode.KAFKA:
        app.include_router(models_router, dependencies=dependencies)
    else:
        logger.info(
            "Kafka mode: /models/*, /align_models/*, /diarize_models/* endpoints "
            "are disabled (model lifecycle lives in worker processes)"
        )
    app.include_router(transcribe_router, dependencies=dependencies)
    app.include_router(status_router, dependencies=dependencies)

    if config.allow_origins is not None:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.allow_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.add_middleware(RequestIDMiddleware)
    # Outermost: short-circuit new requests to 503 once shutting_down is set.
    app.add_middleware(GracefulShutdownMiddleware, app_state=app.state)

    return app
