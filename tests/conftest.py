import io
import os
import tempfile
import wave

import httpx
import numpy as np
import pytest

from fake_backends import fake_diarization, fake_transcription, register_fake_backends
from whisperx_api_server import request_status
from whisperx_api_server.dependencies import get_config

register_fake_backends()

_RESULT_STORE_DIR = tempfile.mkdtemp(prefix="whisperx-test-results-")

TEST_ENV_DEFAULTS = {
    "MODE": "direct",
    "BACKENDS__TRANSCRIPTION": "fake",
    "BACKENDS__ALIGNMENT": "fake",
    "BACKENDS__DIARIZATION": "fake",
    "RESULT_STORE__DIR": _RESULT_STORE_DIR,
}


def _clear_result_store_dir() -> None:
    try:
        for name in os.listdir(_RESULT_STORE_DIR):
            if name.endswith(".json"):
                os.remove(os.path.join(_RESULT_STORE_DIR, name))
    except OSError:
        pass


CONFIG_ENV_KEYS = (
    "API_KEY",
    "API_KEYS_FILE",
    "AUTH_REQUIRED",
    "METRICS_ENABLED",
    "METRICS__ENABLED",
    "WEBUI__ENABLED",
    "WEBUI__DIST_DIR",
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def reset_state():
    request_status._reset_for_tests()
    fake_transcription.calls.clear()
    fake_transcription.raise_exc = None
    fake_diarization.calls.clear()
    _clear_result_store_dir()
    yield
    request_status._reset_for_tests()
    _clear_result_store_dir()
    get_config.cache_clear()


@pytest.fixture
def make_app(monkeypatch):
    def _make(**env):
        from whisperx_api_server.main import create_app

        merged = {**TEST_ENV_DEFAULTS, **env}
        for key in CONFIG_ENV_KEYS:
            if key not in merged:
                monkeypatch.delenv(key, raising=False)
        for key, value in merged.items():
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, str(value))
        get_config.cache_clear()
        return create_app()

    return _make


@pytest.fixture
async def client(make_app):
    transport = httpx.ASGITransport(app=make_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def sine_wav_bytes(
    duration: float = 1.0, freq: float = 440.0, rate: int = 16000
) -> bytes:
    t = np.arange(int(duration * rate)) / rate
    samples = (np.sin(2 * np.pi * freq * t) * 0.5 * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


@pytest.fixture(scope="session")
def sine_wav() -> bytes:
    return sine_wav_bytes()
