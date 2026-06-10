"""Label taxonomy constants for Prometheus instruments."""

from enum import Enum


class ModelStage(str, Enum):
    TRANSCRIPTION = "transcription"
    ALIGNMENT = "alignment"
    DIARIZATION = "diarization"


class ErrorType(str, Enum):
    TIMEOUT = "timeout"
    QUEUE_FULL = "queue_full"
    MODEL_ERROR = "model_error"
    PIPELINE_ERROR = "pipeline_error"
    UNKNOWN = "unknown"


# Histogram bucket sets — frozen at construction time (D-10, FOUND-06)
PIPELINE_BUCKETS: list[float] = [1, 5, 10, 30, 60, 120, 300, 600]
HTTP_BUCKETS: list[float] = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]
# Semaphore acquisition waits are typically sub-second when the GPU is idle
# and grow into multi-minute territory only under sustained backpressure;
# PIPELINE_BUCKETS bottoms out at 1s and hides the fast path entirely.
SEMAPHORE_BUCKETS: list[float] = [
    0.001,
    0.01,
    0.1,
    0.5,
    1,
    5,
    10,
    30,
    60,
    120,
    300,
]
