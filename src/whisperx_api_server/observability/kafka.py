"""Kafka instrument shims"""

import contextlib
from typing import Any


class _NoOpHistogram:
    def labels(self, **kwargs: Any) -> "_NoOpHistogram":
        return self

    def observe(self, amount: float) -> None:
        pass

    def time(self):
        return contextlib.nullcontext()


class _NoOpCounter:
    def labels(self, **kwargs: Any) -> "_NoOpCounter":
        return self

    def inc(self, amount: float = 1) -> None:
        pass


class _NoOpGauge:
    def labels(self, **kwargs: Any) -> "_NoOpGauge":
        return self

    def set(self, value: float) -> None:
        pass

    def inc(self, amount: float = 1) -> None:
        pass

    def dec(self, amount: float = 1) -> None:
        pass

    def set_function(self, f) -> None:
        pass


# Module-level singletons — shims by default, replaced by setup_metrics() when enabled.
pending_jobs: Any = _NoOpGauge()
job_duration: Any = _NoOpHistogram()
job_timeout_total: Any = _NoOpCounter()
queue_rejected_total: Any = _NoOpCounter()
