"""Cancelling direct-mode jobs when the HTTP client disconnects."""

import asyncio
import gc
import threading
import time

import anyio
import httpx
import pytest

import whisperx_api_server.transcriber as transcriber
from fake_backends import fake_transcription
from whisperx_api_server import request_status, webhook
from whisperx_api_server.executors import (
    ModelCallStopped,
    await_to_completion,
    run_stoppable,
)

pytestmark = pytest.mark.anyio

TRANSCRIPTIONS = "/v1/audio/transcriptions"
TRANSLATIONS = "/v1/audio/translations"


async def test_run_stoppable_returns_result():
    assert await run_stoppable(lambda checkpoint: (checkpoint(50.0), 42)[1]) == 42


async def test_run_stoppable_propagates_errors():
    def boom(checkpoint):
        raise ValueError("bad audio")

    with pytest.raises(ValueError, match="bad audio"):
        await run_stoppable(boom)


class _StopRecorder:
    def __init__(self):
        self.started = threading.Event()
        self.exits: list[BaseException] = []

    def __call__(self, checkpoint):
        self.started.set()
        try:
            while True:
                checkpoint(1.0)
                time.sleep(0.01)
        except BaseException as e:
            time.sleep(0.2)
            self.exits.append(e)
            raise


async def test_cancel_waits_for_thread_to_stop():
    fn = _StopRecorder()
    loop = asyncio.get_running_loop()
    unhandled: list[dict] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
    try:
        task = asyncio.ensure_future(run_stoppable(fn))
        await asyncio.to_thread(fn.started.wait, 5)
        task.cancel()
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(fn.exits) == 1 and isinstance(fn.exits[0], ModelCallStopped)
        del task
        gc.collect()
        assert unhandled == []
    finally:
        loop.set_exception_handler(previous_handler)


async def test_cancel_scope_waits_for_thread_without_spinning():
    fn = _StopRecorder()
    async with anyio.create_task_group() as tg:
        tg.start_soon(run_stoppable, fn)
        await asyncio.to_thread(fn.started.wait, 5)
        cpu_before = time.process_time()
        tg.cancel_scope.cancel()
    cpu_spent = time.process_time() - cpu_before

    assert len(fn.exits) == 1 and isinstance(fn.exits[0], ModelCallStopped)
    assert cpu_spent < 0.1


async def test_cancelled_caller_waits_for_task_to_finish():
    release = asyncio.Event()

    async def load():
        await release.wait()
        return "model"

    inner = asyncio.ensure_future(load())
    caller = asyncio.ensure_future(await_to_completion(inner))
    await asyncio.sleep(0.01)
    caller.cancel()
    await asyncio.sleep(0.05)
    assert not caller.done()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert inner.result() == "model"


def _request(path: str, audio: bytes, request_id: str, **fields: str):
    req = httpx.Request(
        "POST",
        f"http://test{path}",
        files={"file": ("a.wav", audio, "audio/wav")},
        data={"response_format": "json", **fields},
    )
    headers = [(k.lower().encode(), v.encode()) for k, v in req.headers.items()]
    headers.append((b"x-request-id", request_id.encode()))
    return path, req.read(), headers


async def _call(app, request, disconnect: asyncio.Event) -> int:
    path, body, headers = request
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": ("127.0.0.1", 50000),
        "server": ("test", 80),
    }
    body_sent = False

    async def receive():
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    messages: list[dict] = []

    async def send(message):
        messages.append(message)

    await app(scope, receive, send)
    return next(m for m in messages if m["type"] == "http.response.start")["status"]


class _Gate:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False
        self.calls = 0

    async def transcribe(self, **kwargs):
        self.calls += 1
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hi"}],
            "language": "en",
        }


async def _disconnect_mid_job(app, gate, request, release_after: bool) -> int:
    disconnect = asyncio.Event()
    call = asyncio.ensure_future(_call(app, request, disconnect))
    await asyncio.wait_for(gate.started.wait(), 10)
    disconnect.set()
    if release_after:
        await asyncio.sleep(0.1)
        gate.release.set()
    return await asyncio.wait_for(call, 10)


@pytest.mark.parametrize("path", [TRANSCRIPTIONS, TRANSLATIONS])
async def test_disconnect_cancels_job(make_app, monkeypatch, sine_wav, path):
    gate = _Gate()
    monkeypatch.setattr(fake_transcription, "transcribe", gate.transcribe)
    app = make_app(CANCEL_ON_DISCONNECT="true")
    request = _request(path, sine_wav, "cancel-1")

    status = await _disconnect_mid_job(app, gate, request, release_after=False)

    assert gate.cancelled
    assert status == 499
    snap = _status("cancel-1")
    assert snap["status"] == "failed"
    assert snap["error_type"] == "Cancelled"


def _status(request_id: str) -> dict:
    snap = request_status.get(request_id)
    assert snap is not None
    return snap


async def _reach_stage(request_id: str, stage: str) -> None:
    while (request_status.get(request_id) or {}).get("stage") != stage:
        await asyncio.sleep(0.01)


async def test_disconnect_while_queued_skips_the_job(make_app, monkeypatch, sine_wav):
    gate = _Gate()
    monkeypatch.setattr(fake_transcription, "transcribe", gate.transcribe)
    monkeypatch.setattr(transcriber, "_concurrency_semaphore", asyncio.Semaphore(1))
    app = make_app(CANCEL_ON_DISCONNECT="true")
    first = asyncio.ensure_future(
        _call(app, _request(TRANSCRIPTIONS, sine_wav, "q-1"), asyncio.Event())
    )
    await asyncio.wait_for(gate.started.wait(), 10)
    disconnect = asyncio.Event()
    second = asyncio.ensure_future(
        _call(app, _request(TRANSCRIPTIONS, sine_wav, "q-2"), disconnect)
    )
    await asyncio.wait_for(_reach_stage("q-2", "awaiting_concurrency"), 10)

    disconnect.set()
    assert await asyncio.wait_for(second, 10) == 499
    gate.release.set()
    assert await asyncio.wait_for(first, 10) == 200
    assert gate.calls == 1
    assert _status("q-2")["error_type"] == "Cancelled"


async def test_cancelled_request_waits_for_model_thread(
    make_app, monkeypatch, sine_wav
):
    fn = _StopRecorder()

    async def transcribe(**kwargs):
        return await run_stoppable(fn)

    monkeypatch.setattr(fake_transcription, "transcribe", transcribe)
    app = make_app(CANCEL_ON_DISCONNECT="true")
    call = asyncio.ensure_future(
        _call(app, _request(TRANSCRIPTIONS, sine_wav, "shutdown-1"), asyncio.Event())
    )
    await asyncio.to_thread(fn.started.wait, 5)
    call.cancel()
    await asyncio.wait({call})

    assert len(fn.exits) == 1


async def test_enabled_without_disconnect_returns_result(make_app, sine_wav):
    app = make_app(CANCEL_ON_DISCONNECT="true")

    status = await _call(
        app, _request(TRANSCRIPTIONS, sine_wav, "ok-1"), asyncio.Event()
    )
    await asyncio.sleep(0.05)

    assert status == 200
    assert _status("ok-1")["status"] == "completed"
    watchers = [
        t
        for t in asyncio.all_tasks()
        if getattr(t.get_coro(), "__name__", "") == "_wait_for_disconnect"
    ]
    assert watchers == []


async def test_enabled_job_error_maps_to_http_error(make_app, sine_wav):
    fake_transcription.raise_exc = RuntimeError("model crashed")
    app = make_app(CANCEL_ON_DISCONNECT="true")

    status = await _call(
        app, _request(TRANSCRIPTIONS, sine_wav, "err-1"), asyncio.Event()
    )

    assert status == 500
    assert _status("err-1")["error_type"] == "RuntimeError"


async def test_disconnect_ignored_when_disabled(make_app, monkeypatch, sine_wav):
    gate = _Gate()
    monkeypatch.setattr(fake_transcription, "transcribe", gate.transcribe)
    app = make_app()
    request = _request(TRANSCRIPTIONS, sine_wav, "keep-1")

    status = await _disconnect_mid_job(app, gate, request, release_after=True)

    assert not gate.cancelled
    assert status == 200
    assert _status("keep-1")["status"] == "completed"


async def test_callback_url_jobs_are_not_cancelled(make_app, monkeypatch, sine_wav):
    gate = _Gate()
    monkeypatch.setattr(fake_transcription, "transcribe", gate.transcribe)
    delivered: list[str] = []

    async def _spy(url, envelope, config):
        delivered.append(url)
        return True

    monkeypatch.setattr(webhook, "deliver_result", _spy)
    app = make_app(CANCEL_ON_DISCONNECT="true", URL_FETCH_ALLOW_PRIVATE_HOSTS="true")
    request = _request(
        TRANSCRIPTIONS,
        sine_wav,
        "hook-1",
        callback_url="http://hook.example/done",
    )

    status = await _disconnect_mid_job(app, gate, request, release_after=True)

    assert not gate.cancelled
    assert status == 200
    assert delivered == ["http://hook.example/done"]
