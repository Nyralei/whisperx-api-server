"""Per-app Prometheus CollectorRegistry and setup_metrics() entry point.

Follows the module-level singleton pattern from kafka_client.py and s3_client.py:
the `_registry` global is None by default; `setup_metrics()` imports
prometheus_client and creates a per-app CollectorRegistry.
"""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from prometheus_client import CollectorRegistry

logger = logging.getLogger(__name__)

_registry: "CollectorRegistry | None" = None


def get_registry() -> "CollectorRegistry | None":
    """Return the per-app CollectorRegistry or None if setup_metrics() has not run."""
    return _registry


def _setup_http_instruments(registry: "CollectorRegistry") -> None:
    """Construct real HTTP instruments and replace http.py shims in-place.

    Called ONLY from setup_metrics() after _registry is created.

    The four singletons in observability/http.py (requests_total,
    request_duration, requests_in_flight, errors_total) are replaced by
    direct attribute assignment on the imported module object, mirroring
    the kafka_client._client / s3_client._client pattern.
    """
    from prometheus_client import Counter, Histogram, Gauge
    from whisperx_api_server.observability import http as _http
    from whisperx_api_server.observability.taxonomy import HTTP_BUCKETS

    _http.requests_total = Counter(
        "whisperx_http_requests_total",
        "Total HTTP requests by endpoint, method, and status code",
        labelnames=["endpoint", "method", "status_code"],
        registry=registry,
    )
    _http.request_duration = Histogram(
        "whisperx_http_request_duration_seconds",
        "HTTP request duration in seconds",
        labelnames=["endpoint", "method", "status_code_class"],
        buckets=HTTP_BUCKETS,
        registry=registry,
    )
    _http.requests_in_flight = Gauge(
        "whisperx_http_requests_in_flight",
        "Number of currently in-flight HTTP requests",
        labelnames=["endpoint"],
        registry=registry,
    )
    _http.errors_total = Counter(
        "whisperx_http_errors_total",
        "Total HTTP error responses (status_code >= 400)",
        labelnames=["endpoint", "status_code", "error_type"],
        registry=registry,
    )
    logger.info("HTTP instruments registered in CollectorRegistry")


def _setup_pipeline_instruments(registry: "CollectorRegistry") -> None:
    """Construct real pipeline Histograms and replace pipeline.py shims in-place."""
    from prometheus_client import Histogram
    from whisperx_api_server.observability import pipeline as _pipe
    from whisperx_api_server.observability.taxonomy import PIPELINE_BUCKETS

    _pipe.stage_duration = Histogram(
        "whisperx_stage_duration_seconds",
        "Per-stage transcription pipeline duration in seconds",
        labelnames=["stage"],
        buckets=PIPELINE_BUCKETS,
        registry=registry,
    )
    _pipe.semaphore_wait = Histogram(
        "whisperx_semaphore_wait_seconds",
        "Time to acquire the GPU concurrency semaphore in seconds",
        labelnames=[],
        buckets=PIPELINE_BUCKETS,
        registry=registry,
    )
    _pipe.realtime_factor = Histogram(
        "whisperx_realtime_factor",
        "Ratio of stage processing time to audio duration (stage_duration / audio_duration_seconds)",
        labelnames=["model", "stage"],
        buckets=PIPELINE_BUCKETS,
        registry=registry,
    )
    logger.info("Pipeline instruments registered in CollectorRegistry")


def _setup_gpu_instruments(registry: "CollectorRegistry") -> None:
    """Construct real GPU Gauges and replace gpu.py shims in-place.

    Called ONLY from setup_metrics() after _registry is created. Gated on
    `torch.cuda.is_available()` AND a successful `pynvml.nvmlInit()`.
    """
    # CUDA guard
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False
    if not cuda_available:
        logger.debug("GPU metrics disabled: CUDA unavailable")
        return

    # pynvml init guard
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        logger.warning(
            "pynvml unavailable or NVIDIA driver not found — GPU metrics disabled"
        )
        from whisperx_api_server.observability import gpu as _gpu
        _gpu._pynvml_ok = False
        _gpu._nvml_handle = None
        return

    from prometheus_client import Gauge
    from whisperx_api_server.observability import gpu as _gpu

    _gpu.vram_used_bytes = Gauge(
        "whisperx_gpu_vram_used_bytes",
        "GPU VRAM used in bytes",
        registry=registry,
    )
    _gpu.vram_free_bytes = Gauge(
        "whisperx_gpu_vram_free_bytes",
        "GPU VRAM free in bytes",
        registry=registry,
    )
    _gpu.vram_total_bytes = Gauge(
        "whisperx_gpu_vram_total_bytes",
        "GPU VRAM total capacity in bytes",
        registry=registry,
    )
    _gpu.gpu_utilization = Gauge(
        "whisperx_gpu_utilization_percent",
        "GPU utilization percent (0-100)",
        registry=registry,
    )

    # Store handle + success flag so main.lifespan() can launch the poller.
    _gpu._nvml_handle = handle
    _gpu._pynvml_ok = True

    # GPU-03: loaded_models Gauge with set_function callbacks per stage.
    # Python closure-over-loop-variable bug and keeps the callbacks resilient
    # to backends that are not yet registered at create_app() time.
    _gpu.loaded_models = Gauge(
        "whisperx_loaded_models",
        "Number of ML models currently loaded in memory",
        labelnames=["stage"],
        registry=registry,
    )

    from whisperx_api_server.observability.taxonomy import ModelStage
    from whisperx_api_server.backends.registry import (
        get_transcription_backend,
        get_alignment_backend,
        get_diarization_backend,
        resolve_stage_backends,
    )

    def _make_loaded_cb(get_fn, backend_name: str):
        def _cb() -> float:
            try:
                return float(len(get_fn(backend_name).list_loaded_models()))
            except Exception:
                return 0.0
        return _cb

    selected = resolve_stage_backends()
    _gpu.loaded_models.labels(stage=ModelStage.TRANSCRIPTION.value).set_function(
        _make_loaded_cb(get_transcription_backend, selected.transcription)
    )
    _gpu.loaded_models.labels(stage=ModelStage.ALIGNMENT.value).set_function(
        _make_loaded_cb(get_alignment_backend, selected.alignment)
    )
    _gpu.loaded_models.labels(stage=ModelStage.DIARIZATION.value).set_function(
        _make_loaded_cb(get_diarization_backend, selected.diarization)
    )

    logger.info("GPU instruments registered in CollectorRegistry")


def _setup_kafka_instruments(registry: "CollectorRegistry") -> None:
    """Construct real Kafka instruments and replace kafka.py shims in-place.

    Called ONLY from setup_metrics() after _registry is created, and ONLY
    when config.mode == DistributedMode.KAFKA.
    """
    from prometheus_client import Counter, Histogram, Gauge
    from whisperx_api_server.observability import kafka as _kafka
    from whisperx_api_server.observability.taxonomy import PIPELINE_BUCKETS

    _kafka.pending_jobs = Gauge(
        "whisperx_kafka_pending_jobs",
        "Number of Kafka jobs currently waiting for a worker reply",
        registry=registry,
    )
    _kafka.job_duration = Histogram(
        "whisperx_kafka_job_duration_seconds",
        "End-to-end Kafka job duration from submit to reply",
        labelnames=["status"],
        buckets=PIPELINE_BUCKETS,
        registry=registry,
    )
    _kafka.job_timeout_total = Counter(
        "whisperx_kafka_job_timeout_total",
        "Total Kafka jobs that timed out waiting for a worker reply",
        registry=registry,
    )
    _kafka.queue_rejected_total = Counter(
        "whisperx_kafka_queue_rejected_total",
        "Total requests rejected because the pending job queue was full",
        registry=registry,
    )

    import whisperx_api_server.kafka_client as kafka_client
    _kafka.pending_jobs.set_function(
        lambda: float(len(kafka_client._pending_jobs))
    )

    logger.info("Kafka instruments registered in CollectorRegistry")


def setup_metrics(config: Any) -> None:
    """Initialize the per-app Prometheus CollectorRegistry.

    Called from create_app() only when config.metrics.enabled is True.
    Imports prometheus_client and creates a CollectorRegistry bound to this module as the singleton
    `_registry`.

    Parameters
    ----------
    config : MetricsConfig (duck-typed as Any to avoid importing MetricsConfig here)
        Must expose a boolean `.enabled` attribute.
    """
    global _registry
    if not getattr(config, "enabled", False):
        return
    if _registry is not None:
        logger.debug(
            "setup_metrics() called twice; reusing existing CollectorRegistry"
        )
        return

    from prometheus_client import CollectorRegistry

    _registry = CollectorRegistry()
    _setup_http_instruments(_registry)
    _setup_pipeline_instruments(_registry)
    _setup_gpu_instruments(_registry)
    from whisperx_api_server.config import DistributedMode
    from whisperx_api_server.dependencies import get_config
    if get_config().mode == DistributedMode.KAFKA:
        _setup_kafka_instruments(_registry)
    logger.info(
        "Prometheus metrics enabled: per-app CollectorRegistry initialized"
    )
