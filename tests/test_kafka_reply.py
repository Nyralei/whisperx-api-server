"""Unit tests for the Kafka reply-consumer decision logic (no broker)."""

import asyncio
import time

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


def _entry(fut, ts=0.0):
    return kafka_client._PendingJob(future=fut, submitted_at=ts, last_activity=ts)


async def test_owner_resolves_future_no_late_metric(spy_late_reply):
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    kafka_client._pending_jobs["job-1"] = _entry(fut)
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
    kafka_client._pending_jobs["job-err"] = _entry(fut)
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


async def test_non_local_stub_reply_is_not_late(spy_late_reply):
    # A reply landing on a stub built from progress events (job owned elsewhere)
    # must not count as late.
    request_status.ensure_tracked("job-stub")
    st = request_status.get("job-stub")
    assert st is not None
    assert st["local"] is False

    kafka_client._handle_reply_event(
        {"job_id": "job-stub", "status": "ok", "result": {"text": "hi"}}
    )
    assert spy_late_reply.count == 0


async def test_async_entry_reply_pops_without_future(spy_late_reply):
    # Async submits register (None, ts) — no awaiter. A reply must clear the
    # entry without raising and without counting as late.
    kafka_client._pending_jobs["job-async"] = _entry(None)
    request_status.start("job-async", mode="kafka")

    kafka_client._handle_reply_event(
        {"job_id": "job-async", "status": "ok", "result": {"text": "hi"}}
    )

    assert "job-async" not in kafka_client._pending_jobs
    assert spy_late_reply.count == 0


async def test_apply_worker_timeline_unknown_id_noop():
    # No-ops for unknown ids, so the reply consumer needs no extra guard.
    request_status.apply_worker_timeline(
        "never-seen", {"transcribe": {"started_at": 1.0, "completed_at": 2.0}}
    )
    assert request_status.get("never-seen") is None


async def test_late_ok_reply_reconciles_failed_status(spy_late_reply):
    # Reaped/timed-out locally, but the worker finished: the ok reply must flip
    # the failed state back to completed so /status matches /result.
    request_status.start("job-late", mode="kafka")
    request_status.mark_failed("job-late", "reaped: no reply", "TimeoutError")

    kafka_client._handle_reply_event(
        {"job_id": "job-late", "status": "ok", "result": {"text": "hi"}}
    )

    st = request_status.get("job-late")
    assert st is not None
    assert st["status"] == "completed"
    assert st["error"] is None
    assert st["error_type"] is None
    assert spy_late_reply.count == 1


async def test_reply_settles_status_without_awaiter(spy_late_reply):
    # Async submits have no awaiter to call mark_completed/mark_failed; the
    # reply itself must settle the status.
    kafka_client._pending_jobs["job-a-ok"] = _entry(None)
    request_status.start("job-a-ok", mode="kafka")
    kafka_client._handle_reply_event(
        {"job_id": "job-a-ok", "status": "ok", "result": {"text": "hi"}}
    )
    st = request_status.get("job-a-ok")
    assert st is not None
    assert st["status"] == "completed"

    kafka_client._pending_jobs["job-a-err"] = _entry(None)
    request_status.start("job-a-err", mode="kafka")
    kafka_client._handle_reply_event(
        {
            "job_id": "job-a-err",
            "status": "error",
            "error": "boom",
            "error_type": "ValueError",
        }
    )
    st = request_status.get("job-a-err")
    assert st is not None
    assert st["status"] == "failed"
    assert st["error"] == "boom"


async def test_late_error_reply_does_not_override_completed():
    request_status.start("job-done", mode="kafka")
    request_status.mark_completed("job-done")

    kafka_client._handle_reply_event(
        {"job_id": "job-done", "status": "error", "error": "dup worker failed"}
    )

    st = request_status.get("job-done")
    assert st is not None
    assert st["status"] == "completed"


async def test_wait_for_reply_returns_result():
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    kafka_client._pending_jobs["job-w1"] = _entry(fut, time.monotonic())
    loop.call_later(0.05, fut.set_result, {"text": "hi"})

    result = await kafka_client.wait_for_reply("job-w1", fut, timeout=5.0)
    assert result == {"text": "hi"}


async def test_wait_for_reply_times_out_on_inactivity():
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    kafka_client._pending_jobs["job-w2"] = _entry(fut, time.monotonic() - 10.0)

    with pytest.raises(TimeoutError):
        await kafka_client.wait_for_reply("job-w2", fut, timeout=0.05)
    assert not fut.done()
    fut.cancel()


async def test_wait_for_reply_extended_by_progress():
    # Inactivity clock restarts on touch_pending: total elapsed time exceeds the
    # timeout, but the job stays live because progress keeps arriving.
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    kafka_client._pending_jobs["job-w3"] = _entry(fut, time.monotonic())

    async def _worker():
        for _ in range(12):
            await asyncio.sleep(0.05)
            kafka_client.touch_pending("job-w3")
        fut.set_result({"text": "done"})

    worker = asyncio.create_task(_worker())
    # Total run (~0.6s) far exceeds the timeout; touch gaps (~0.05s) stay far
    # inside it, so only a stalled touch loop could time out.
    result = await kafka_client.wait_for_reply("job-w3", fut, timeout=0.4)
    await worker
    assert result == {"text": "done"}
