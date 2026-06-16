"""Unit tests for the worker message handler (no broker, no S3)."""

import json
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


class FakeProducer:
    def __init__(self, order):
        self.sends: list[_Send] = []
        self._order = order

    async def send_and_wait(self, topic, *, key, value):
        self.sends.append(_Send(topic, key, value))
        self._order.append(("send_and_wait", topic))

    async def send(self, topic, *, key, value):
        self.sends.append(_Send(topic, key, value))
        self._order.append(("send", topic))


class FakeS3:
    def __init__(self, order):
        self.results: dict[str, str] = {}
        self.claims: dict[str, int] = {}
        self.deleted_audio: list[str] = []
        self.claim_deletes = 0
        self._order = order

    async def get_result(self, job_id):
        return self.results.get(job_id)

    async def put_result(self, job_id, envelope):
        self.results[job_id] = envelope
        self._order.append(("put_result", job_id))

    async def increment_claim(self, job_id):
        self.claims[job_id] = self.claims.get(job_id, 0) + 1
        return self.claims[job_id]

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
    monkeypatch.setattr(_kafka, "dlq_total", dlq)
    monkeypatch.setattr(_kafka, "idempotent_skip_total", skip)
    return Harness(ctx, producer, s3, order, commits, dlq, skip)


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


async def test_poison_job_routes_to_dlq_after_max_attempts(harness, monkeypatch):
    async def _crash(event, **kwargs):
        raise _Crash("killed")

    monkeypatch.setattr(handler, "process_job", _crash)
    event = {"job_id": "poison", "s3_key": "audio/poison/a.wav", "params": {}}

    # Three crash deliveries: claim increments, never commits, never DLQs.
    for attempt in (1, 2, 3):
        with pytest.raises(_Crash):
            await handler.handle_message(event, harness.ctx)
        assert harness.s3.claims["poison"] == attempt
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
