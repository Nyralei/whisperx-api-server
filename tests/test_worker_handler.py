"""Unit tests for the worker message handler (no broker, no S3)."""

import asyncio
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from whisperx_api_server.config import KafkaConfig, S3Config
from whisperx_api_server.observability import kafka as _kafka
from whisperx_worker import handler

pytestmark = pytest.mark.anyio


class _Crash(BaseException):
    """Stands in for a worker-killing failure (OOM/segfault): not caught by the
    handler's ``except Exception``, so it aborts before commit like a real crash.
    """


@dataclass
class _Send:
    topic: str
    key: bytes
    value: bytes
    partition: int | None = None


class FakeProducer:
    def __init__(self, order):
        self.sends: list[_Send] = []
        self.fail_next_sends = 0
        self._order = order

    async def send_and_wait(self, topic, *, key, value, partition=None):
        if self.fail_next_sends > 0:
            self.fail_next_sends -= 1
            raise RuntimeError("broker unavailable")
        self.sends.append(_Send(topic, key, value, partition))
        self._order.append(("send_and_wait", topic))

    async def send(self, topic, *, key, value, partition=None):
        self.sends.append(_Send(topic, key, value, partition))
        self._order.append(("send", topic))


class FakeS3:
    def __init__(self, order):
        self.results: dict[str, str] = {}
        self.claims: dict[str, dict] = {}
        self.deleted_audio: list[str] = []
        self.claim_deletes = 0
        self.renewals: list[str] = []
        self._order = order

    async def get_result(self, job_id):
        return self.results.get(job_id)

    async def put_result(self, job_id, envelope):
        self.results[job_id] = envelope
        self._order.append(("put_result", job_id))

    async def acquire_job_lease(self, job_id, worker_id, ttl_seconds):
        now = time.time()
        lease = self.claims.get(job_id)
        if (
            lease is not None
            and lease["expires_at"] > now
            and lease["owner"] != worker_id
        ):
            return False, lease["attempts"]
        attempts = (lease["attempts"] if lease else 0) + 1
        self.claims[job_id] = {
            "attempts": attempts,
            "owner": worker_id,
            "expires_at": now + ttl_seconds,
        }
        return True, attempts

    async def renew_job_lease(self, job_id, worker_id, ttl_seconds):
        self.renewals.append(job_id)
        lease = self.claims.get(job_id)
        if lease is None or lease["owner"] != worker_id:
            return False
        lease["expires_at"] = time.time() + ttl_seconds
        return True

    async def delete_claim(self, job_id):
        self.claim_deletes += 1
        self.claims.pop(job_id, None)

    async def delete_audio(self, key):
        self.deleted_audio.append(key)


class _SpyCounter:
    def __init__(self):
        self.count = 0

    def inc(self, amount: float = 1):
        self.count += amount


@dataclass
class Harness:
    ctx: handler.WorkerContext
    producer: FakeProducer
    s3: FakeS3
    order: list
    commits: list
    dlq: _SpyCounter
    skip: _SpyCounter
    deferred: _SpyCounter


@pytest.fixture
def harness(monkeypatch):
    order: list = []
    commits: list = []
    producer = FakeProducer(order)
    s3 = FakeS3(order)

    async def _commit():
        commits.append(True)
        order.append(("commit", None))

    config = SimpleNamespace(kafka=KafkaConfig(), s3=S3Config())
    ctx = handler.WorkerContext(
        producer=producer,
        config=config,
        commit=_commit,
        worker_id="test-worker",
        s3=s3,
    )

    dlq = _SpyCounter()
    skip = _SpyCounter()
    deferred = _SpyCounter()
    monkeypatch.setattr(_kafka, "dlq_total", dlq)
    monkeypatch.setattr(_kafka, "idempotent_skip_total", skip)
    monkeypatch.setattr(_kafka, "lease_deferred_total", deferred)
    monkeypatch.setattr(handler, "_LEASE_DEFER_SECONDS", 0)
    return Harness(ctx, producer, s3, order, commits, dlq, skip, deferred)


async def _fake_ok(event, *, progress_producer, progress_topic, timeline_out):
    timeline_out["worker.transcribe"] = {"started_at": 1.0, "completed_at": 2.0}
    return {"text": "done", "segments": []}


def _reply_sends(h):
    return [s for s in h.producer.sends if s.topic == h.ctx.config.kafka.reply_topic]


async def test_success_writes_result_before_reply_then_commits(harness, monkeypatch):
    monkeypatch.setattr(handler, "process_job", _fake_ok)
    event = {"job_id": "j1", "s3_key": "audio/j1/a.wav", "params": {}}

    await handler.handle_message(event, harness.ctx)

    sends = _reply_sends(harness)
    assert len(sends) == 1
    # Stored envelope is byte-identical to the reply payload.
    assert sends[0].value.decode() == harness.s3.results["j1"]
    assert json.loads(sends[0].value)["status"] == "ok"

    kinds = [step[0] for step in harness.order]
    assert (
        kinds.index("put_result") < kinds.index("send_and_wait") < kinds.index("commit")
    )
    assert harness.s3.deleted_audio == ["audio/j1/a.wav"]
    assert harness.s3.claim_deletes == 1
    assert harness.commits == [True]
    assert harness.dlq.count == 0


async def test_handled_error_replies_and_commits(harness, monkeypatch):
    async def _fake_err(event, **kwargs):
        raise ValueError("bad audio")

    monkeypatch.setattr(handler, "process_job", _fake_err)
    event = {"job_id": "j2", "s3_key": "audio/j2/a.wav", "params": {}}

    await handler.handle_message(event, harness.ctx)

    env = json.loads(harness.s3.results["j2"])
    assert env["status"] == "error"
    assert env["error_type"] == "ValueError"
    assert "bad audio" in env["error"]
    assert json.loads(_reply_sends(harness)[0].value)["status"] == "error"
    assert harness.commits == [True]
    assert harness.s3.claim_deletes == 1
    assert harness.dlq.count == 0


async def test_redelivered_completed_job_resends_without_reprocessing(
    harness, monkeypatch
):
    def _boom(*args, **kwargs):
        raise AssertionError("process_job must not run on the resend path")

    monkeypatch.setattr(handler, "process_job", _boom)
    harness.s3.results["j3"] = json.dumps(
        {"job_id": "j3", "status": "ok", "result": {"text": "cached"}}
    )
    event = {"job_id": "j3", "s3_key": "audio/j3/a.wav", "params": {}}

    await handler.handle_message(event, harness.ctx)

    sends = _reply_sends(harness)
    assert len(sends) == 1
    assert json.loads(sends[0].value)["result"]["text"] == "cached"
    assert harness.commits == [True]
    assert harness.skip.count == 1
    assert harness.dlq.count == 0
    # No attempt counted on the resend path.
    assert harness.s3.claims == {}


async def test_completion_webhook_fires_on_success(harness, monkeypatch):
    monkeypatch.setattr(handler, "process_job", _fake_ok)
    calls: list = []

    async def _spy(url, envelope, config):
        calls.append((url, envelope))
        return True

    monkeypatch.setattr(handler.webhook, "deliver_result", _spy)
    event = {
        "job_id": "cb1",
        "s3_key": "audio/cb1/a.wav",
        "params": {},
        "callback_url": "http://hook.example/x",
    }

    await handler.handle_message(event, harness.ctx)

    assert len(calls) == 1
    url, envelope = calls[0]
    assert url == "http://hook.example/x"
    # The webhook carries the exact stored envelope.
    assert envelope == harness.s3.results["cb1"]
    assert json.loads(envelope)["status"] == "ok"


async def test_completion_webhook_not_sent_without_callback_url(harness, monkeypatch):
    monkeypatch.setattr(handler, "process_job", _fake_ok)
    calls: list = []

    async def _spy(url, envelope, config):
        calls.append(url)
        return True

    monkeypatch.setattr(handler.webhook, "deliver_result", _spy)
    event = {"job_id": "cb0", "s3_key": "audio/cb0/a.wav", "params": {}}

    await handler.handle_message(event, harness.ctx)

    assert calls == []


async def test_completion_webhook_skipped_on_resend(harness, monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("process_job must not run on the resend path")

    monkeypatch.setattr(handler, "process_job", _boom)
    calls: list = []

    async def _spy(url, envelope, config):
        calls.append(url)
        return True

    monkeypatch.setattr(handler.webhook, "deliver_result", _spy)
    harness.s3.results["cb2"] = json.dumps(
        {"job_id": "cb2", "status": "ok", "result": {"text": "cached"}}
    )
    event = {
        "job_id": "cb2",
        "s3_key": "audio/cb2/a.wav",
        "params": {},
        "callback_url": "http://hook.example/x",
    }

    await handler.handle_message(event, harness.ctx)

    # A redelivery resends the reply but must never re-fire the webhook.
    assert calls == []
    assert harness.skip.count == 1


async def test_completion_webhook_fires_on_dlq(harness, monkeypatch):
    async def _crash(event, **kwargs):
        raise _Crash("killed")

    monkeypatch.setattr(handler, "process_job", _crash)
    calls: list = []

    async def _spy(url, envelope, config):
        calls.append((url, envelope))
        return True

    monkeypatch.setattr(handler.webhook, "deliver_result", _spy)
    event = {
        "job_id": "cb3",
        "s3_key": "audio/cb3/a.wav",
        "params": {},
        "callback_url": "http://hook.example/x",
    }

    for _ in (1, 2, 3):
        with pytest.raises(_Crash):
            await handler.handle_message(event, harness.ctx)
    assert calls == []  # no webhook while crash-redelivering pre-DLQ

    await handler.handle_message(event, harness.ctx)  # fourth delivery → DLQ

    assert len(calls) == 1
    assert json.loads(calls[0][1])["status"] == "error"


async def test_poison_job_routes_to_dlq_after_max_attempts(harness, monkeypatch):
    async def _crash(event, **kwargs):
        raise _Crash("killed")

    monkeypatch.setattr(handler, "process_job", _crash)
    event = {"job_id": "poison", "s3_key": "audio/poison/a.wav", "params": {}}

    # Three crash deliveries: attempts advance (the crashed worker's own stale
    # lease is retaken), never commits, never DLQs.
    for attempt in (1, 2, 3):
        with pytest.raises(_Crash):
            await handler.handle_message(event, harness.ctx)
        assert harness.s3.claims["poison"]["attempts"] == attempt
        assert harness.commits == []
        assert harness.dlq.count == 0

    # Fourth delivery exceeds max_delivery_attempts (default 3) → DLQ.
    await handler.handle_message(event, harness.ctx)

    dlq_sends = [
        s
        for s in harness.producer.sends
        if s.topic == harness.ctx.config.kafka.dead_letter_topic
    ]
    assert len(dlq_sends) == 1
    payload = json.loads(dlq_sends[0].value)
    assert payload["attempts"] == 4
    assert payload["reason"] == "max_delivery_attempts exceeded"
    assert payload["worker_id"] == "test-worker"
    assert payload["job_id"] == "poison"

    assert json.loads(_reply_sends(harness)[-1].value)["status"] == "error"
    assert "poison" in harness.s3.results
    assert harness.s3.claims.get("poison") is None
    assert harness.commits == [True]
    assert harness.dlq.count == 1


def _live_lease(owner: str, attempts: int = 1) -> dict:
    return {"attempts": attempts, "owner": owner, "expires_at": time.time() + 300.0}


async def test_live_foreign_lease_defers_and_requeues(harness, monkeypatch):
    """A redelivered copy of a job running on another worker must be requeued,
    not processed concurrently."""

    def _boom(*args, **kwargs):
        raise AssertionError("process_job must not run while another worker leases")

    monkeypatch.setattr(handler, "process_job", _boom)
    harness.s3.claims["j-busy"] = _live_lease("other-worker")
    event = {"job_id": "j-busy", "s3_key": "audio/j-busy/a.wav", "params": {}}

    await handler.handle_message(event, harness.ctx)

    requeued = [
        s
        for s in harness.producer.sends
        if s.topic == harness.ctx.config.kafka.request_topic
    ]
    assert len(requeued) == 1
    assert json.loads(requeued[0].value)["job_id"] == "j-busy"
    assert _reply_sends(harness) == []
    assert harness.commits == [True]
    assert harness.deferred.count == 1
    # The holder's lease is untouched.
    assert harness.s3.claims["j-busy"]["owner"] == "other-worker"
    assert harness.s3.claims["j-busy"]["attempts"] == 1


async def test_defer_resends_result_that_appears_while_parked(harness, monkeypatch):
    """If the lease holder finishes during the defer pause, resend its result
    instead of requeueing another copy."""

    def _boom(*args, **kwargs):
        raise AssertionError("process_job must not run")

    monkeypatch.setattr(handler, "process_job", _boom)
    harness.s3.claims["j-fin"] = _live_lease("other-worker")
    envelope = json.dumps({"job_id": "j-fin", "status": "ok", "result": {"text": "x"}})

    calls = {"n": 0}
    real_get = harness.s3.get_result

    async def _get_result(job_id):
        calls["n"] += 1
        if calls["n"] >= 2:
            harness.s3.results.setdefault(job_id, envelope)
        return await real_get(job_id)

    monkeypatch.setattr(harness.s3, "get_result", _get_result)
    event = {"job_id": "j-fin", "s3_key": "audio/j-fin/a.wav", "params": {}}

    await handler.handle_message(event, harness.ctx)

    sends = _reply_sends(harness)
    assert len(sends) == 1
    assert json.loads(sends[0].value)["result"]["text"] == "x"
    requeued = [
        s
        for s in harness.producer.sends
        if s.topic == harness.ctx.config.kafka.request_topic
    ]
    assert requeued == []
    assert harness.commits == [True]
    assert harness.skip.count == 1


async def test_expired_foreign_lease_taken_over(harness, monkeypatch):
    """A lease whose holder crashed (expired, never renewed) must be taken over
    and the job processed, advancing the attempts counter."""
    monkeypatch.setattr(handler, "process_job", _fake_ok)
    harness.s3.claims["j-dead"] = {
        "attempts": 1,
        "owner": "crashed-worker",
        "expires_at": time.time() - 10.0,
    }
    event = {"job_id": "j-dead", "s3_key": "audio/j-dead/a.wav", "params": {}}

    await handler.handle_message(event, harness.ctx)

    assert json.loads(_reply_sends(harness)[0].value)["status"] == "ok"
    assert harness.commits == [True]
    assert harness.s3.claims.get("j-dead") is None  # released at terminal
    assert harness.deferred.count == 0


async def test_result_stored_between_check_and_acquire_resends(harness, monkeypatch):
    """The holder finishing between the cached check and our acquire leaves us a
    fresh lease over a done job — resend, never reprocess."""

    def _boom(*args, **kwargs):
        raise AssertionError("process_job must not run")

    monkeypatch.setattr(handler, "process_job", _boom)
    envelope = json.dumps({"job_id": "j-race", "status": "ok", "result": {"text": "r"}})

    calls = {"n": 0}
    real_get = harness.s3.get_result

    async def _get_result(job_id):
        calls["n"] += 1
        if calls["n"] >= 2:
            harness.s3.results.setdefault(job_id, envelope)
        return await real_get(job_id)

    monkeypatch.setattr(harness.s3, "get_result", _get_result)
    event = {"job_id": "j-race", "s3_key": "audio/j-race/a.wav", "params": {}}

    await handler.handle_message(event, harness.ctx)

    assert len(_reply_sends(harness)) == 1
    assert harness.skip.count == 1
    assert harness.commits == [True]
    assert harness.s3.claims.get("j-race") is None


async def test_lease_renewed_and_heartbeat_during_long_job(harness, monkeypatch):
    """The liveness loop must fire while process_job runs: the lease outlives
    jobs longer than the TTL, and heartbeats keep the API's inactivity clock
    fresh through stages longer than the reply timeout."""
    harness.ctx.config.kafka = KafkaConfig(job_lease_ttl_seconds=0.05)

    async def _slow_ok(event, *, progress_producer, progress_topic, timeline_out):
        await asyncio.sleep(0.1)
        return {"text": "done", "segments": []}

    monkeypatch.setattr(handler, "process_job", _slow_ok)
    event = {"job_id": "j-slow", "s3_key": "audio/j-slow/a.wav", "params": {}}

    await handler.handle_message(event, harness.ctx)

    assert "j-slow" in harness.s3.renewals
    heartbeats = [
        s
        for s in harness.producer.sends
        if s.topic == harness.ctx.config.kafka.progress_topic
        and json.loads(s.value).get("status") == "heartbeat"
    ]
    assert heartbeats and json.loads(heartbeats[0].value)["job_id"] == "j-slow"
    assert "stage" not in json.loads(heartbeats[0].value)
    assert json.loads(_reply_sends(harness)[0].value)["status"] == "ok"


@dataclass
class _Msg:
    value: bytes
    offset: int = 0
    partition: int = 0
    key: bytes | None = None


@dataclass(frozen=True)
class _TP:
    topic: str
    partition: int


class FakeConsumer:
    """Minimal getmany/pause/resume stand-in for the run loop."""

    def __init__(
        self,
        batches,
        *,
        topic,
        partitions: Iterable[int] = frozenset({0}),
        owned: Iterable[int] = frozenset({0}),
    ):
        self._batches = list(batches)  # one getmany return value per entry
        self.getmany_calls = 0
        self.pause_calls = 0
        self.resume_calls = 0
        self.seeks: list[tuple[_TP, int]] = []
        self._topic = topic
        self._partitions = set(partitions)
        self._owned = {_TP(topic, p) for p in owned}

    def assignment(self):
        return set(self._owned)

    def partitions_for_topic(self, topic):
        return set(self._partitions)

    def pause(self, *partitions):
        self.pause_calls += 1

    def resume(self, *partitions):
        self.resume_calls += 1

    def seek(self, tp, offset):
        self.seeks.append((tp, offset))

    async def getmany(self, timeout_ms, max_records):
        self.getmany_calls += 1
        await asyncio.sleep(0)  # yield like a real broker poll
        if self._batches:
            return self._batches.pop(0)
        return {}

    async def commit(self):
        await asyncio.sleep(0)


def _job_msg(job_id: str, offset: int = 0, **extra) -> _Msg:
    return _Msg(
        json.dumps({"job_id": job_id, **extra}).encode(),
        offset=offset,
        key=job_id.encode(),
    )


async def _run_loop_with_slow_job(harness, monkeypatch, consumer, *, ticks=10):
    """Run consume_loop with a handle_message that spins `ticks` event-loop
    iterations, then requests shutdown. Returns (handled_job_ids, paused_flag)."""
    monkeypatch.setattr(handler, "_HANDOFF_THROTTLE_SECONDS", 0)
    shutdown_event = asyncio.Event()
    paused_for_job = [False]
    handled: list[str] = []

    async def _slow_handle(event, ctx):
        handled.append(event["job_id"])
        for _ in range(ticks):
            await asyncio.sleep(0)
        shutdown_event.set()

    monkeypatch.setattr(handler, "handle_message", _slow_handle)
    await asyncio.wait_for(
        handler.consume_loop(consumer, harness.ctx, shutdown_event, paused_for_job),
        timeout=5,
    )
    return handled, paused_for_job


async def test_consume_loop_keeps_polling_while_job_runs(harness, monkeypatch):
    """A long job must not starve the poll loop: getmany has to keep firing while
    handle_message runs, or aiokafka evicts the worker past max_poll_interval_ms."""
    monkeypatch.setattr(handler, "_HANDOFF_THROTTLE_SECONDS", 0)
    shutdown_event = asyncio.Event()
    paused_for_job = [False]
    polls_during: list[int] = []
    topic = harness.ctx.config.kafka.request_topic

    async def _slow_handle(event, ctx):
        start = consumer.getmany_calls
        for _ in range(5):
            await asyncio.sleep(0)
        polls_during.append(consumer.getmany_calls - start)
        shutdown_event.set()

    monkeypatch.setattr(handler, "handle_message", _slow_handle)

    consumer = FakeConsumer([{_TP(topic, 0): [_job_msg("live-1")]}], topic=topic)

    await asyncio.wait_for(
        handler.consume_loop(consumer, harness.ctx, shutdown_event, paused_for_job),
        timeout=5,
    )

    assert polls_during and polls_during[0] >= 1  # polled during the in-flight job
    # Sole owner of every partition: nothing to relay to, so it pauses.
    assert consumer.pause_calls == 1
    assert consumer.resume_calls == 1
    assert paused_for_job == [False]


async def test_queued_job_relayed_to_foreign_partition(harness, monkeypatch):
    """A message arriving mid-job must be republished to another worker's
    partition — uncommitted — instead of waiting behind the in-flight job."""
    topic = harness.ctx.config.kafka.request_topic
    tp = _TP(topic, 0)
    consumer = FakeConsumer(
        [{tp: [_job_msg("A", offset=10)]}, {tp: [_job_msg("B", offset=11)]}],
        topic=topic,
        partitions={0, 1, 2},
        owned={0},
    )

    handled, paused_for_job = await _run_loop_with_slow_job(
        harness, monkeypatch, consumer
    )

    assert handled == ["A"]  # B was never processed here
    relayed = [s for s in harness.producer.sends if s.topic == topic]
    assert len(relayed) == 1
    event = json.loads(relayed[0].value)
    assert event["job_id"] == "B"
    assert event["handoff_hops"] == 1
    assert relayed[0].partition in {1, 2}
    assert consumer.pause_calls == 0  # relay mode never pauses
    assert harness.commits == []  # relay never commits mid-job
    assert paused_for_job == [False]


async def test_duplicate_of_inflight_job_dropped_not_relayed(harness, monkeypatch):
    """A redelivery of the job currently being processed must not be relayed —
    that would start a concurrent duplicate run on an idle worker."""
    topic = harness.ctx.config.kafka.request_topic
    tp = _TP(topic, 0)
    consumer = FakeConsumer(
        [{tp: [_job_msg("A", offset=10)]}, {tp: [_job_msg("A", offset=11)]}],
        topic=topic,
        partitions={0, 1, 2},
        owned={0},
    )

    handled, _ = await _run_loop_with_slow_job(harness, monkeypatch, consumer)

    assert handled == ["A"]
    assert [s for s in harness.producer.sends if s.topic == topic] == []


async def test_relay_publish_failure_rewinds_for_retry(harness, monkeypatch):
    """If the republish fails, the consumer seeks back so the message is
    refetched instead of being silently consumed past."""
    topic = harness.ctx.config.kafka.request_topic
    tp = _TP(topic, 0)
    consumer = FakeConsumer(
        [
            {tp: [_job_msg("A", offset=10)]},
            {tp: [_job_msg("B", offset=11)]},
            {tp: [_job_msg("B", offset=11)]},  # refetch after seek
        ],
        topic=topic,
        partitions={0, 1, 2},
        owned={0},
    )
    harness.producer.fail_next_sends = 1

    handled, _ = await _run_loop_with_slow_job(harness, monkeypatch, consumer)

    assert handled == ["A"]
    assert consumer.seeks == [(tp, 11)]
    relayed = [s for s in harness.producer.sends if s.topic == topic]
    assert len(relayed) == 1
    assert json.loads(relayed[0].value)["job_id"] == "B"


async def test_garbage_mid_job_dropped_without_relay(harness, monkeypatch):
    """Unparseable messages fetched mid-job are dropped, not republished."""
    topic = harness.ctx.config.kafka.request_topic
    tp = _TP(topic, 0)
    consumer = FakeConsumer(
        [{tp: [_job_msg("A", offset=10)]}, {tp: [_Msg(b"not json", offset=11)]}],
        topic=topic,
        partitions={0, 1, 2},
        owned={0},
    )

    handled, _ = await _run_loop_with_slow_job(harness, monkeypatch, consumer)

    assert handled == ["A"]
    assert [s for s in harness.producer.sends if s.topic == topic] == []
    assert consumer.seeks == []


async def test_commit_safely_swallows_rebalance_error():
    class CommitFailedError(Exception):
        pass

    async def _rebalanced():
        raise CommitFailedError("group already rebalanced")

    await handler.commit_safely(_rebalanced)  # must not raise


async def test_commit_safely_propagates_unrelated_error():
    async def _boom():
        raise ValueError("real bug")

    with pytest.raises(ValueError):
        await handler.commit_safely(_boom)


async def test_commit_safely_commits_when_healthy():
    calls: list = []

    async def _ok():
        calls.append(True)

    await handler.commit_safely(_ok)
    assert calls == [True]
