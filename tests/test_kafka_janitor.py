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


def _entry(fut, ts):
    return kafka_client._PendingJob(future=fut, submitted_at=ts, last_activity=ts)


async def test_reaps_stale_entry(spy_metrics):
    reaped, gauge = spy_metrics
    cfg = KafkaConfig(reply_timeout_seconds=0.0)
    fut = asyncio.get_running_loop().create_future()
    kafka_client._pending_jobs["stale"] = _entry(fut, time.monotonic() - 1000.0)
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
    kafka_client._pending_jobs["fresh"] = _entry(fut, time.monotonic())

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
    kafka_client._pending_jobs["orphan"] = _entry(None, time.monotonic() - 1000.0)
    request_status.start("orphan", mode="kafka")

    n = kafka_client._reap_stale_pending(cfg)

    assert n == 1
    assert "orphan" not in kafka_client._pending_jobs
    orphan_st = request_status.get("orphan")
    assert orphan_st is not None
    assert orphan_st["status"] == "failed"
    assert reaped.count == 1


async def test_progress_touch_defers_reaping(spy_metrics):
    # Submitted long ago, but a worker progress event refreshed last_activity —
    # the janitor must treat the job as live.
    reaped, _ = spy_metrics
    cfg = KafkaConfig(reply_timeout_seconds=0.0)
    kafka_client._pending_jobs["long-job"] = _entry(None, time.monotonic() - 9999.0)
    kafka_client.touch_pending("long-job")

    n = kafka_client._reap_stale_pending(cfg, now=time.monotonic() + 30.0)

    assert n == 0
    assert "long-job" in kafka_client._pending_jobs


async def test_touch_pending_unknown_id_noop():
    kafka_client.touch_pending("never-seen")
    assert "never-seen" not in kafka_client._pending_jobs


async def test_heartbeat_event_touches_without_stage_bookkeeping():
    # Heartbeats refresh liveness for the janitor but never create stages/stubs.
    kafka_client._pending_jobs["hb-job"] = _entry(None, time.monotonic() - 9999.0)
    request_status.start("hb-job", mode="kafka")

    kafka_client._handle_progress_event({"job_id": "hb-job", "status": "heartbeat"})

    entry = kafka_client._pending_jobs["hb-job"]
    assert time.monotonic() - entry.last_activity < 5.0
    st = request_status.get("hb-job")
    assert st is not None
    assert st["stages"] == []
    # Unknown job: heartbeat must not fabricate a tracking stub.
    kafka_client._handle_progress_event({"job_id": "hb-ghost", "status": "heartbeat"})
    assert request_status.get("hb-ghost") is None


async def test_stage_event_touches_and_records_stage():
    kafka_client._pending_jobs["st-job"] = _entry(None, time.monotonic() - 9999.0)
    request_status.start("st-job", mode="kafka")

    kafka_client._handle_progress_event(
        {"job_id": "st-job", "stage": "diarize", "status": "in_progress"}
    )

    entry = kafka_client._pending_jobs["st-job"]
    assert time.monotonic() - entry.last_activity < 5.0
    st = request_status.get("st-job")
    assert st is not None
    assert st["stages"][-1]["name"] == "worker.diarize"
