"""Unit tests for the Kafka reply-consumer decision logic (no broker)."""

import asyncio

import pytest

from whisperx_api_server import kafka_client, request_status
from whisperx_api_server.observability import kafka as _kafka

pytestmark = pytest.mark.anyio


class _SpyCounter:
    def __init__(self) -> None:
        self.count = 0

    def labels(self, **kwargs):
        return self

    def inc(self, amount: float = 1) -> None:
        self.count += amount


@pytest.fixture
def spy_late_reply(monkeypatch):
    spy = _SpyCounter()
    monkeypatch.setattr(_kafka, "late_reply_total", spy)
    return spy


@pytest.fixture(autouse=True)
def _clear_pending():
    kafka_client._pending_jobs.clear()
    yield
    kafka_client._pending_jobs.clear()


async def test_owner_resolves_future_no_late_metric(spy_late_reply):
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    kafka_client._pending_jobs["job-1"] = (fut, 0.0)
    request_status.start("job-1", mode="kafka")

    kafka_client._handle_reply_event(
        {"job_id": "job-1", "status": "ok", "result": {"text": "hi"}}
    )

    assert fut.done()
    assert fut.result() == {"text": "hi"}
    assert "job-1" not in kafka_client._pending_jobs
    assert spy_late_reply.count == 0


async def test_owner_resolves_error_future(spy_late_reply):
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    kafka_client._pending_jobs["job-err"] = (fut, 0.0)
    request_status.start("job-err", mode="kafka")

    kafka_client._handle_reply_event(
        {
            "job_id": "job-err",
            "status": "error",
            "error": "boom",
            "error_type": "ValueError",
        }
    )

    assert fut.done()
    with pytest.raises(ValueError, match="boom"):
        fut.result()
    assert spy_late_reply.count == 0


async def test_foreign_reply_is_silent_noop(spy_late_reply):
    # Fan-out duplicate for a job submitted elsewhere: must not count as late.
    kafka_client._handle_reply_event(
        {
            "job_id": "ghost",
            "status": "ok",
            "result": {"text": "hi"},
            "timeline": {"transcribe": {"started_at": 1.0, "completed_at": 2.0}},
        }
    )
    assert spy_late_reply.count == 0
    # apply_worker_timeline must have no-op'd for the unknown id.
    assert request_status.get("ghost") is None


async def test_late_local_reply_increments_metric(spy_late_reply):
    # Tracked locally but the future is gone — the only genuinely-late case.
    request_status.start("job-2", mode="kafka")
    assert "job-2" not in kafka_client._pending_jobs

    kafka_client._handle_reply_event(
        {"job_id": "job-2", "status": "ok", "result": {"text": "hi"}}
    )
    assert spy_late_reply.count == 1


async def test_apply_worker_timeline_unknown_id_noop():
    # No-ops for unknown ids, so the reply consumer needs no extra guard.
    request_status.apply_worker_timeline(
        "never-seen", {"transcribe": {"started_at": 1.0, "completed_at": 2.0}}
    )
    assert request_status.get("never-seen") is None
