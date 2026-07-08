"""End-to-end Kafka tests against a real broker (testcontainers, KRaft mode).

Marked `kafka` so the default fast suite skips them; run with `pytest -m kafka`
(needs Docker). These cover the wiring fakes can't reach: real topic creation
(including the dead-letter topic), the produce path, and the reply consumer
resolving pending futures over the broker.
"""

import asyncio
import contextlib
import json
import uuid

import pytest

from whisperx_api_server import kafka_client, request_status
from whisperx_api_server.config import KafkaConfig

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


@pytest.fixture
async def kcfg(kafka_bootstrap):
    uid = uuid.uuid4().hex[:8]
    cfg = KafkaConfig(
        bootstrap_servers=kafka_bootstrap,
        request_topic=f"req-{uid}",
        reply_topic=f"rep-{uid}",
        progress_topic=f"prog-{uid}",
        dead_letter_topic=f"dlq-{uid}",
        topic_partitions=1,
        topic_replication_factor=1,
        reply_timeout_seconds=30.0,
    )
    await kafka_client.start(cfg)
    try:
        yield cfg
    finally:
        await kafka_client.stop()
        kafka_client._pending_jobs.clear()


async def _make_producer(cfg):
    from aiokafka import AIOKafkaProducer

    producer = AIOKafkaProducer(bootstrap_servers=cfg.bootstrap_servers)
    await producer.start()
    return producer


async def _send(producer, topic, key, value):
    await producer.send_and_wait(
        topic, key=key.encode(), value=json.dumps(value).encode()
    )


async def _resolve_via_replies(future, send_reply, *, attempts=40, delay=0.5):
    """Re-send the reply until the latest-offset consumer is positioned to see it."""
    for _ in range(attempts):
        if future.done():
            return
        await send_reply()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(future), timeout=delay)
            return


async def test_start_creates_topics(kcfg):
    from aiokafka import AIOKafkaConsumer

    consumer = AIOKafkaConsumer(bootstrap_servers=kcfg.bootstrap_servers)
    await consumer.start()
    try:
        topics = await consumer.topics()
    finally:
        await consumer.stop()

    assert {
        kcfg.request_topic,
        kcfg.reply_topic,
        kcfg.progress_topic,
        kcfg.dead_letter_topic,
    } <= topics


async def test_submit_job_produces_request(kcfg):
    from aiokafka import AIOKafkaConsumer

    consumer = AIOKafkaConsumer(
        kcfg.request_topic,
        bootstrap_servers=kcfg.bootstrap_servers,
        group_id=f"test-{uuid.uuid4().hex[:8]}",
        auto_offset_reset="earliest",
    )
    await consumer.start()
    try:
        job_id = "job-submit-1"
        future = await kafka_client.submit_job(
            job_id,
            s3_key=None,
            audio_url="http://example.com/a.wav",
            filename="a.wav",
            params={"model_name": None},
        )
        assert future is not None
        future.cancel()
        msg = await asyncio.wait_for(consumer.getone(), timeout=20)
        assert msg.value is not None
        event = json.loads(msg.value)
    finally:
        await consumer.stop()

    assert event["job_id"] == job_id
    assert event["audio_url"] == "http://example.com/a.wav"
    assert event["params"]["model_name"] is None


async def test_async_submit_registers_no_future(kcfg):
    from aiokafka import AIOKafkaConsumer

    consumer = AIOKafkaConsumer(
        kcfg.request_topic,
        bootstrap_servers=kcfg.bootstrap_servers,
        group_id=f"test-{uuid.uuid4().hex[:8]}",
        auto_offset_reset="earliest",
    )
    await consumer.start()
    try:
        job_id = "job-async-1"
        future = await kafka_client.submit_job(
            job_id,
            s3_key=None,
            audio_url="http://example.com/a.wav",
            filename="a.wav",
            params={},
            track_future=False,
        )
        assert future is None
        assert kafka_client._pending_jobs[job_id].future is None
        msg = await asyncio.wait_for(consumer.getone(), timeout=20)
        assert msg.value is not None
        event = json.loads(msg.value)
    finally:
        await consumer.stop()

    assert event["job_id"] == job_id


async def test_reply_resolves_future_and_applies_timeline(kcfg):
    reply_task = asyncio.create_task(kafka_client.reply_consumer_loop(kcfg))
    producer = await _make_producer(kcfg)
    try:
        job_id = "job-reply-1"
        request_status.start(job_id, mode="kafka")
        request_status.set_stage(job_id, "awaiting_worker")
        future = await kafka_client.submit_job(
            job_id,
            s3_key=None,
            audio_url="http://example.com/a.wav",
            filename="a.wav",
            params={},
        )
        assert future is not None
        reply = {
            "job_id": job_id,
            "status": "ok",
            "result": {"text": "hello", "segments": [], "language": "en"},
            "timeline": {"transcribe": {"started_at": 1000.0, "completed_at": 1003.0}},
        }

        async def _send_reply():
            await _send(producer, kcfg.reply_topic, job_id, reply)

        await _resolve_via_replies(future, _send_reply)
        result = await asyncio.wait_for(future, timeout=5)
    finally:
        reply_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reply_task
        await producer.stop()

    assert result["text"] == "hello"
    st = request_status.get(job_id)
    assert st is not None
    stages = [s["name"] for s in st["stages"]]
    assert "worker.transcribe" in stages


async def test_progress_builds_status_on_non_submitting_replica(kcfg):
    # Models instance B: it never submitted this job and learns it purely from
    # the progress topic, converging through stages to a terminal state.
    progress_task = asyncio.create_task(kafka_client.progress_consumer_loop(kcfg))
    producer = await _make_producer(kcfg)
    try:
        # Warm up until the latest-offset consumer is positioned (a throwaway
        # entry appearing proves it), so the real sequence is delivered in full.
        warm = "warmup-job"
        for _ in range(60):
            await _send(
                producer,
                kcfg.progress_topic,
                warm,
                {"job_id": warm, "stage": "submitted", "status": "submitted"},
            )
            await asyncio.sleep(0.3)
            if request_status.get(warm) is not None:
                break
        assert request_status.get(warm) is not None, (
            "progress consumer never positioned"
        )

        job_id = "job-crossreplica-1"
        for event in (
            {
                "job_id": job_id,
                "stage": "submitted",
                "status": "submitted",
                "filename": "a.wav",
                "params": {"model": None},
            },
            {"job_id": job_id, "stage": "transcribe", "status": "in_progress"},
            {"job_id": job_id, "stage": "align", "status": "in_progress"},
            {"job_id": job_id, "stage": "finalize", "status": "completed"},
        ):
            await _send(producer, kcfg.progress_topic, job_id, event)

        for _ in range(40):
            st = request_status.get(job_id)
            if st is not None and st["status"] == "completed":
                break
            await asyncio.sleep(0.2)
        st = request_status.get(job_id)
    finally:
        progress_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await progress_task
        await producer.stop()

    assert st is not None
    assert st["local"] is False
    assert st["filename"] == "a.wav"
    assert st["status"] == "completed"
    names = [s["name"] for s in st["stages"]]
    assert "worker.transcribe" in names
    assert "worker.align" in names


async def test_reply_for_untracked_job_does_not_break_loop(kcfg):
    reply_task = asyncio.create_task(kafka_client.reply_consumer_loop(kcfg))
    producer = await _make_producer(kcfg)
    try:
        job_id = "job-after-ghost"
        future = await kafka_client.submit_job(
            job_id,
            s3_key=None,
            audio_url="http://example.com/a.wav",
            filename="a.wav",
            params={},
        )
        assert future is not None
        ghost = {"job_id": "ghost", "status": "ok", "result": {"text": "x"}}
        real = {"job_id": job_id, "status": "ok", "result": {"text": "real"}}

        async def _send_replies():
            await _send(producer, kcfg.reply_topic, "ghost", ghost)
            await _send(producer, kcfg.reply_topic, job_id, real)

        await _resolve_via_replies(future, _send_replies)
        result = await asyncio.wait_for(future, timeout=5)
    finally:
        reply_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reply_task
        await producer.stop()

    assert result["text"] == "real"
