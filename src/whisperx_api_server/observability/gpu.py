"""GPU instrument shims"""

import asyncio
import logging
from typing import Any


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
vram_used_bytes: Any = _NoOpGauge()
vram_free_bytes: Any = _NoOpGauge()
vram_total_bytes: Any = _NoOpGauge()
gpu_utilization: Any = _NoOpGauge()
loaded_models: Any = _NoOpGauge()

# Populated by registry._setup_gpu_instruments when pynvml init succeeds.
# Read by main.lifespan() to decide whether to create the poller Task.
_nvml_handle: Any = None
_pynvml_ok: bool = False

logger = logging.getLogger(__name__)


async def _gpu_poll_loop(interval: int, handle: Any) -> None:
    """Background coroutine that polls pynvml and updates GPU Gauges.

    Behavior:
      - Every `interval` seconds, reads VRAM info and utilization from pynvml
        and writes them to the module-level Gauge singletons (which have been
        replaced by real prometheus_client.Gauge objects in registry._setup_gpu_instruments).
      - On ANY exception during a poll cycle, logs WARNING and continues the loop.
      - The coroutine never returns under normal operation; it is cancelled by
        the lifespan shutdown path.
    """
    import pynvml

    while True:
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_used_bytes.set(info.used)
            vram_free_bytes.set(info.free)
            vram_total_bytes.set(info.total)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_utilization.set(util.gpu)
        except Exception as exc:
            logger.warning("GPU metrics poll failed (will retry): %s", exc)
        await asyncio.sleep(interval)
