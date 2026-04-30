"""Pipeline-stage instrument shims"""

import contextlib
from typing import Any


class _NoOpHistogram:
    def labels(self, **kwargs: Any) -> "_NoOpHistogram":
        return self

    def observe(self, amount: float) -> None:
        pass

    def time(self):
        return contextlib.nullcontext()


# Module-level singletons — shims by default, replaced by setup_metrics() when enabled.
stage_duration: Any = _NoOpHistogram()
semaphore_wait: Any = _NoOpHistogram()
realtime_factor: Any = _NoOpHistogram()
