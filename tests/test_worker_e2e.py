"""Worker handler end-to-end against a real Kafka broker + real MinIO (S3).

Closes the gap the in-process fakes cannot reach: the worker's idempotency
marker resend and poison-job dead-letter routing run here through real S3
objects (``claims/`` and ``results/`` in MinIO) and real Kafka offset
redelivery, plus the broker-level reply fan-out that makes multi-replica reply
routing correct.

The worker's message core is ``handle_message`` (extracted for exactly this);
these tests drive it against real infrastructure with real consumer offset
semantics — a crash before commit is reproduced by seeking the partition back,
which is what a worker restart does.

Marked ``kafka``; needs Docker. Run with ``pytest -m kafka``.
"""

import asyncio
import json
import os
import time
import uuid

import numpy as np
import pytest

import whisperx_worker.handler as worker_handler
import whisperx_worker.processor as processor
from fake_backends import fake_transcription
from whisperx_api_server import kafka_client, request_status
from whisperx_api_server import s3_client as s3_client
from whisperx_api_server.dependencies import get_config
from whisperx_api_server.transcriber import init_concurrency
from whisperx_worker.handler import WorkerContext, handle_message

pytestmark = [pytest.mark.anyio, pytest.mark.kafka]


@pytest.fixture(scope="module")
def kafka_bootstrap():
    from testcontainers.kafka import KafkaContainer

    container = KafkaContainer().with_kraft()
    try:
        container.start()
    except Exception as e:
        pytest.skip(f"Kafka broker unavailable (Docker required): {e}")
    try:
        yield container.get_bootstrap_server()
    finally:
        container.stop()


@pytest.fixture(scope="module")
def minio_endpoint():
    import re

    from testcontainers.core.container import DockerContainer
    from testcontainers.core.wait_strategies import LogMessageWaitStrategy

    container = (
        DockerContainer("minio/minio:latest")
        .with_env("MINIO_ROOT_USER", "minioadmin")
        .with_env("MINIO_ROOT_PASSWORD", "minioadmin")
        .with_exposed_ports(9000)
        .with_command("server /data")
        .waiting_for(LogMessageWaitStrategy(re.compile(r"API:|Status:")))
    )
    try:
        container.start()
    except Exception as e:
        pytest.skip(f"MinIO unavailable (Docker required): {e}")
    try:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(9000)
        yield f"http://{host}:{port}"
    finally:
        container.stop()


@pytest.fixture
async def worker_env(kafka_bootstrap, minio_endpoint, monkeypatch):
    uid = uuid.uuid4().hex[:8]
    env = {
        "MODE": "kafka",
        "BACKENDS__TRANSCRIPTION": "fake",
        "BACKENDS__ALIGNMENT": "fake",
        "BACKENDS__DIARIZATION": "fake",
        "S3__ENDPOINT_URL": minio_endpoint,
        "S3__BUCKET": f"wx-{uid}",
        "S3__ACCESS_KEY_ID": "minioadmin",
        "S3__SECRET_ACCESS_KEY": "minioadmin",
        "S3__REGION": "us-east-1",
        "S3__DELETE_AFTER_DOWNLOAD": "true",
        "KAFKA__BOOTSTRAP_SERVERS": kafka_bootstrap,
        "KAFKA__REQUEST_TOPIC": f"req-{uid}",
        "KAFKA__REPLY_TOPIC": f"rep-{uid}",
        "KAFKA__PROGRESS_TOPIC": f"prog-{uid}",
        "KAFKA__DEAD_LETTER_TOPIC": f"dlq-{uid}",
        "KAFKA__TOPIC_PARTITIONS": "1",
        "KAFKA__TOPIC_REPLICATION_FACTOR": "1",
        "KAFKA__CONSUMER_GROUP_WORKER": f"worker-{uid}",
        "KAFKA__MAX_DELIVERY_ATTEMPTS": "3",
        "KAFKA__REPLY_TIMEOUT_SECONDS": "30",
        # The webhook test posts to a loopback capture server.
        "URL_FETCH_ALLOW_PRIVATE_HOSTS": "true",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    get_config.cache_clear()
    cfg = get_config()

    await s3_client.init_client(cfg.s3)
    await kafka_client.start(cfg.kafka)
    init_concurrency()
    try:
        yield cfg
    finally:
        await kafka_client.stop()
        await s3_client.close_client()
        kafka_client._pending_jobs.clear()
        get_config.cache_clear()


class _SpyCounter:
    """Stand-in for a metric shim that records how often the branch fires."""

    def __init__(self) -> None:
        self.count = 0

    def labels(self, **_: object) -> "_SpyCounter":
        return self

    def inc(self, amount: float = 1) -> None:
        self.count += amount


class _FlakyCommit:
    """Commit callable that raises on its first N calls — a crash-before-commit."""

    def __init__(self, real, fail_times: int) -> None:
        self._real = real
        self._fail_times = fail_times
        self.calls = 0

    async def __call__(self) -> None:
        self.calls += 1
        if self.calls <= self._fail_times:
            raise RuntimeError("simulated crash before offset commit")
        await self._real()


async def _fake_load_audio(file_path, request_id, sample_rate=16000):
    # Keep the handler contract (S3 download + real inference stages) in the test
    # while skipping the ffmpeg decode, which is covered by the direct-mode suite.
    return np.zeros(sample_rate, dtype="float32")


async def _producer(cfg):
    from aiokafka import AIOKafkaProducer

    p = AIOKafkaProducer(bootstrap_servers=cfg.kafka.bootstrap_servers)
    await p.start()
    return p


async def _worker_consumer(cfg):
    from aiokafka import AIOKafkaConsumer, TopicPartition

    c = AIOKafkaConsumer(
        bootstrap_servers=cfg.kafka.bootstrap_servers,
        group_id=cfg.kafka.consumer_group_worker,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
    )
    await c.start()
    tp = TopicPartition(cfg.kafka.request_topic, 0)
    c.assign([tp])
    await c.seek_to_beginning(tp)
    return c, tp


async def _produce_job(producer, cfg, event):
    await producer.send_and_wait(
        cfg.kafka.request_topic,
        key=event["job_id"].encode(),
        value=json.dumps(event).encode(),
    )


async def _drain_topic(cfg, topic, job_id, timeout=10.0):
    from aiokafka import AIOKafkaConsumer

    c = AIOKafkaConsumer(
        topic,
        bootstrap_servers=cfg.kafka.bootstrap_servers,
        group_id=f"drain-{uuid.uuid4().hex[:8]}",
        auto_offset_reset="earliest",
        enable_auto_commit=False,
    )
    await c.start()
    out: list[dict] = []
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            batch = await c.getmany(timeout_ms=500)
            got = False
            for _tp, msgs in batch.items():
                for m in msgs:
                    if m.value:
                        got = True
                        decoded = json.loads(m.value)
                        if decoded.get("job_id") == job_id:
                            out.append(decoded)
            if out and not got:
                break
    finally:
        await c.stop()
    return out


def _job_event(job_id, key, filename="a.wav"):
    return {
        "job_id": job_id,
        "s3_key": key,
        "audio_url": None,
        "filename": filename,
        "params": {"model_name": "fake-tiny", "align": False, "diarize": False},
    }


async def test_marker_resend_skips_reprocess_through_real_worker(
    worker_env, monkeypatch
):
    """1.4: a completed job redelivered (crash before commit) resends the stored
    envelope from S3 without rerunning inference."""
    cfg = worker_env
    monkeypatch.setattr(processor, "load_audio_from_path", _fake_load_audio)
    skip_spy = _SpyCounter()
    monkeypatch.setattr(
        "whisperx_api_server.observability.kafka.idempotent_skip_total", skip_spy
    )

    job_id = "job-marker-1"
    key = await s3_client.upload_audio(b"rawbytes", job_id, "a.wav")
    event = _job_event(job_id, key)

    producer = await _producer(cfg)
    consumer, tp = await _worker_consumer(cfg)
    try:
        await _produce_job(producer, cfg, event)
        flaky = _FlakyCommit(consumer.commit, fail_times=1)
        ctx = WorkerContext(
            producer=producer, config=cfg, commit=flaky, worker_id="w-test"
        )

        # First delivery: processes fully, then the commit crashes.
        msg = await asyncio.wait_for(consumer.getone(), timeout=20)
        assert msg.value is not None
        with pytest.raises(RuntimeError):
            await handle_message(json.loads(msg.value), ctx)
        assert len(fake_transcription.calls) == 1
        assert await s3_client.get_result(job_id) is not None

        # Worker restart → the uncommitted record is redelivered.
        consumer.seek(tp, msg.offset)
        msg2 = await asyncio.wait_for(consumer.getone(), timeout=20)
        assert msg2.value is not None
        await handle_message(json.loads(msg2.value), ctx)
    finally:
        await consumer.stop()
        await producer.stop()

    # No second inference; the stored reply was resent, and commit finally stuck.
    assert len(fake_transcription.calls) == 1
    assert skip_spy.count == 1
    assert flaky.calls == 2

    replies = await _drain_topic(cfg, cfg.kafka.reply_topic, job_id)
    assert len(replies) == 2
    assert all(r["status"] == "ok" for r in replies)
    assert all(r["result"]["text"] == "hello world" for r in replies)


async def test_poison_job_routed_to_dlq_and_future_fails_fast(worker_env, monkeypatch):
    """1.3: a job that keeps killing the worker is retired to the DLQ after
    max_delivery_attempts, and the error reply fails the submitter's future."""
    cfg = worker_env
    monkeypatch.setattr(processor, "load_audio_from_path", _fake_load_audio)
    dlq_spy = _SpyCounter()
    monkeypatch.setattr("whisperx_api_server.observability.kafka.dlq_total", dlq_spy)

    class Poison(BaseException):
        """Not an Exception — models an OOM-kill / native crash the handler
        cannot catch, so the delivery is never committed."""

    fake_transcription.raise_exc = Poison("worker-killing job")

    job_id = "job-poison-1"
    key = await s3_client.upload_audio(b"rawbytes", job_id, "a.wav")
    event = _job_event(job_id, key)

    producer = await _producer(cfg)
    consumer, tp = await _worker_consumer(cfg)
    crashes = 0
    try:
        await _produce_job(producer, cfg, event)
        ctx = WorkerContext(
            producer=producer, config=cfg, commit=consumer.commit, worker_id="w-test"
        )
        for _ in range(cfg.kafka.max_delivery_attempts + 2):
            msg = await asyncio.wait_for(consumer.getone(), timeout=20)
            assert msg.value is not None
            try:
                await handle_message(json.loads(msg.value), ctx)
            except Poison:
                crashes += 1
                consumer.seek(tp, msg.offset)  # crash w/o commit → redelivery
                continue
            break  # DLQ path ran and committed
    finally:
        await consumer.stop()
        await producer.stop()

    # The job crashed the worker exactly max_delivery_attempts times before the
    # next delivery crossed the threshold and routed it to the DLQ.
    assert crashes == cfg.kafka.max_delivery_attempts
    assert dlq_spy.count == 1

    dlq_events = await _drain_topic(cfg, cfg.kafka.dead_letter_topic, job_id)
    assert len(dlq_events) == 1
    dlq = dlq_events[0]
    assert dlq["attempts"] == cfg.kafka.max_delivery_attempts + 1
    assert dlq["reason"] == "max_delivery_attempts exceeded"
    assert dlq["worker_id"] == "w-test"

    stored = await s3_client.get_result(job_id)
    assert stored is not None
    assert json.loads(stored)["status"] == "error"

    replies = await _drain_topic(cfg, cfg.kafka.reply_topic, job_id)
    assert len(replies) == 1
    assert replies[0]["status"] == "error"

    # The submitter's future fails fast off the real DLQ reply bytes.
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    request_status.start(job_id, mode="kafka")
    now = time.monotonic()
    kafka_client._pending_jobs[job_id] = kafka_client._PendingJob(future, now, now)
    kafka_client._handle_reply_event(replies[0])
    assert future.done()
    with pytest.raises(RuntimeError):
        future.result()


async def test_completion_webhook_delivered_through_real_worker(
    worker_env, monkeypatch
):
    """3.3: a job carrying callback_url has its result envelope POSTed once to the
    callback, through the real worker and a real HTTP round-trip."""
    from aiohttp import web

    cfg = worker_env
    monkeypatch.setattr(processor, "load_audio_from_path", _fake_load_audio)

    received: list[dict] = []

    async def _hook(request):
        received.append(await request.json())
        return web.Response(status=204)

    app = web.Application()
    app.router.add_post("/hook", _hook)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    callback_url = f"http://127.0.0.1:{port}/hook"

    job_id = "job-webhook-1"
    key = await s3_client.upload_audio(b"rawbytes", job_id, "a.wav")
    event = _job_event(job_id, key)
    event["callback_url"] = callback_url

    producer = await _producer(cfg)
    consumer, _tp = await _worker_consumer(cfg)
    try:
        await _produce_job(producer, cfg, event)
        ctx = WorkerContext(
            producer=producer, config=cfg, commit=consumer.commit, worker_id="w-test"
        )
        msg = await asyncio.wait_for(consumer.getone(), timeout=20)
        assert msg.value is not None
        await handle_message(json.loads(msg.value), ctx)
    finally:
        await consumer.stop()
        await producer.stop()
        await runner.cleanup()

    # Delivery is awaited inside handle_message (before commit), so it's arrived.
    assert len(received) == 1
    assert received[0]["job_id"] == job_id
    assert received[0]["status"] == "ok"
    assert received[0]["result"]["text"] == "hello world"


async def test_graceful_shutdown_finishes_inflight_and_skips_next(
    worker_env, monkeypatch
):
    """3.5: a shutdown request (what the SIGTERM handler does) landing mid-job lets
    the in-flight job finish — reply + commit — then the loop exits without
    consuming the next queued job."""
    cfg = worker_env
    monkeypatch.setattr(processor, "load_audio_from_path", _fake_load_audio)

    j1, j2 = "job-drain-1", "job-drain-2"
    k1 = await s3_client.upload_audio(b"raw1", j1, "a.wav")
    k2 = await s3_client.upload_audio(b"raw2", j2, "a.wav")

    shutdown_event = asyncio.Event()
    paused_for_job = [False]
    processed: list[str] = []
    real_handle = worker_handler.handle_message

    async def _spy_handle(event, ctx):
        shutdown_event.set()  # SIGTERM arrives while the first job is running
        await real_handle(event, ctx)
        processed.append(event["job_id"])

    monkeypatch.setattr(worker_handler, "handle_message", _spy_handle)

    from aiokafka import AIOKafkaConsumer

    producer = await _producer(cfg)
    consumer = AIOKafkaConsumer(
        bootstrap_servers=cfg.kafka.bootstrap_servers,
        group_id=cfg.kafka.consumer_group_worker,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
    )
    await consumer.start()
    consumer.subscribe(topics=[cfg.kafka.request_topic])
    ctx = WorkerContext(
        producer=producer, config=cfg, commit=consumer.commit, worker_id="w-test"
    )
    try:
        await _produce_job(producer, cfg, _job_event(j1, k1))
        await _produce_job(producer, cfg, _job_event(j2, k2))
        await asyncio.wait_for(
            worker_handler.consume_loop(consumer, ctx, shutdown_event, paused_for_job),
            timeout=30,
        )
    finally:
        await consumer.stop()
        await producer.stop()

    # Only the in-flight job ran and was persisted; the queued job is untouched.
    assert processed == [j1]
    assert await s3_client.get_result(j1) is not None
    assert await s3_client.get_result(j2) is None

    replies = await _drain_topic(cfg, cfg.kafka.reply_topic, j1)
    assert len(replies) == 1
    assert replies[0]["status"] == "ok"


async def test_reply_fanned_out_to_every_replica_group(worker_env):
    """1.2: the broker delivers each reply to every replica's unique group, so
    whichever replica holds the future is guaranteed to receive it."""
    cfg = worker_env
    from aiokafka import AIOKafkaConsumer

    groups = [
        f"{cfg.kafka.reply_group_id}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        for _ in range(2)
    ]
    consumers = []
    for g in groups:
        c = AIOKafkaConsumer(
            cfg.kafka.reply_topic,
            bootstrap_servers=cfg.kafka.bootstrap_servers,
            group_id=g,
            auto_offset_reset="latest",
            enable_auto_commit=True,
        )
        await c.start()
        consumers.append(c)

    producer = await _producer(cfg)
    job_id = "job-fanout-1"
    reply = {"job_id": job_id, "status": "ok", "result": {"text": "hi"}}
    received = [False, False]
    try:
        # Re-send until both latest-offset consumers are positioned and see it.
        for _ in range(40):
            await producer.send_and_wait(
                cfg.kafka.reply_topic,
                key=job_id.encode(),
                value=json.dumps(reply).encode(),
            )
            for i, c in enumerate(consumers):
                if received[i]:
                    continue
                batch = await c.getmany(timeout_ms=500)
                for _tp, msgs in batch.items():
                    for m in msgs:
                        if m.value and json.loads(m.value).get("job_id") == job_id:
                            received[i] = True
            if all(received):
                break
    finally:
        for c in consumers:
            await c.stop()
        await producer.stop()

    assert all(received), "reply was not fanned out to both replica groups"
