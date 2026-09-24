"""Align/diarize are skipped when transcription yields no speech segments."""

import numpy as np
import pytest

import whisperx_worker.processor as processor
from fake_backends import fake_diarization, fake_transcription
from whisperx_api_server.dependencies import get_config
from whisperx_api_server.storage import service as storage

pytestmark = pytest.mark.anyio


@pytest.fixture
def worker_config(monkeypatch):
    for key, value in {
        "MODE": "kafka",
        "BACKENDS__TRANSCRIPTION": "fake",
        "BACKENDS__ALIGNMENT": "fake",
        "BACKENDS__DIARIZATION": "fake",
    }.items():
        monkeypatch.setenv(key, value)
    get_config.cache_clear()
    yield get_config()
    get_config.cache_clear()


async def _run_job(monkeypatch, *, segments):
    async def _fake_download(key, suffix=""):
        return "/tmp/whisperx-skip-empty-nonexistent.wav"

    async def _fake_load(file_path, request_id, sample_rate=16000):
        return np.zeros(sample_rate, dtype="float32")

    async def _fake_transcribe(**kwargs):
        return {"segments": segments, "language": "en"}

    monkeypatch.setattr(storage, "download_audio_to_temp", _fake_download)
    monkeypatch.setattr(processor, "load_audio_from_path", _fake_load)
    monkeypatch.setattr(fake_transcription, "transcribe", _fake_transcribe)

    event = {
        "job_id": "skip-empty-test",
        "s3_key": "audio/skip-empty-test/a.wav",
        "audio_url": None,
        "filename": "a.wav",
        "params": {"align": True, "diarize": True},
    }
    return await processor.process_job(event)


async def test_empty_transcription_skips_diarize(worker_config, monkeypatch):
    fake_diarization.calls.clear()
    result = await _run_job(monkeypatch, segments=[])
    assert fake_diarization.calls == []
    assert result["text"] == ""


async def test_nonempty_transcription_runs_diarize(worker_config, monkeypatch):
    fake_diarization.calls.clear()
    result = await _run_job(
        monkeypatch, segments=[{"start": 0.0, "end": 1.0, "text": "hi"}]
    )
    assert len(fake_diarization.calls) == 1
    assert result["text"] == "hi"
