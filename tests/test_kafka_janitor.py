"""Unit tests for the defensive pending-jobs janitor (no broker)."""

import asyncio
import time

import pytest

from whisperx_api_server import kafka_client, request_status
from whisperx_api_server.config import KafkaConfig
from whisperx_api_server.observability import kafka as _kafka

pytestmark = pytest.mark.anyio


class _SpyCounter:
    def __init__(self) -> None:
        self.count = 0

    def inc(self, amount: float = 1) -> None:
        self.count += amount


class _SpyGauge:
    def __init__(self) -> None:
        self.value: float | None = None

    def set(self, value: float) -> None:
        self.value = value


@pytest.fixture
def spy_metrics(monkeypatch):
    reaped = _SpyCounter()
    gauge = _SpyGauge()
    monkeypatch.setattr(_kafka, "pending_reaped_total", reaped)
    monkeypatch.setattr(_kafka, "pending_jobs", gauge)
    return reaped, gauge


@pytest.fixture(autouse=True)
def _clear_pending():
    kafka_client._pending_jobs.clear()
    yield
    kafka_client._pending_jobs.clear()


async def test_reaps_stale_entry(spy_metrics):
    reaped, gauge = spy_metrics
    cfg = KafkaConfig(reply_timeout_seconds=0.0)
    fut = asyncio.get_running_loop().create_future()
    kafka_client._pending_jobs["stale"] = (fut, time.monotonic() - 1000.0)
    request_status.start("stale", mode="kafka")

    n = kafka_client._reap_stale_pending(cfg)

    assert n == 1
    assert "stale" not in kafka_client._pending_jobs
    assert fut.done()
    with pytest.raises(TimeoutError):
        fut.result()
    st = request_status.get("stale")
    assert st is not None
    assert st["status"] == "failed"
    assert st["error_type"] == "TimeoutError"
    assert reaped.count == 1
    assert gauge.value == 0


async def test_skips_fresh_entry(spy_metrics):
    reaped, _ = spy_metrics
    cfg = KafkaConfig(reply_timeout_seconds=3600.0)
    fut = asyncio.get_running_loop().create_future()
    kafka_client._pending_jobs["fresh"] = (fut, time.monotonic())

    n = kafka_client._reap_stale_pending(cfg)

    assert n == 0
    assert "fresh" in kafka_client._pending_jobs
    assert not fut.done()
    assert reaped.count == 0
    fut.cancel()


async def test_reaps_awaiterless_entry(spy_metrics):
    # Async submits store no future; the janitor still reaps and marks failed.
    reaped, _ = spy_metrics
    cfg = KafkaConfig(reply_timeout_seconds=0.0)
    kafka_client._pending_jobs["orphan"] = (None, time.monotonic() - 1000.0)
    request_status.start("orphan", mode="kafka")

    n = kafka_client._reap_stale_pending(cfg)

    assert n == 1
    assert "orphan" not in kafka_client._pending_jobs
    orphan_st = request_status.get("orphan")
    assert orphan_st is not None
    assert orphan_st["status"] == "failed"
    assert reaped.count == 1
