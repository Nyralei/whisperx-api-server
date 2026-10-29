import copy
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

pytest.importorskip("whisperx")

from whisperx_api_server.backends import whisperx_backend as backend

pytestmark = pytest.mark.anyio


@pytest.fixture
def alignment_config(monkeypatch):
    config = SimpleNamespace(alignment=SimpleNamespace(models={}))
    monkeypatch.setattr(backend, "get_config", lambda: config)
    return config.alignment


@pytest.mark.parametrize("language", ["bg", "yo", "th", "yue"])
async def test_unsupported_language_preserves_segments(
    language, alignment_config, monkeypatch, caplog
):
    result = {
        "language": language,
        "segments": [{"start": 0.0, "end": 1.0, "text": "test"}],
    }
    original = copy.deepcopy(result)
    load = AsyncMock()
    monkeypatch.setattr(backend, "load_align_model", load)
    with caplog.at_level(logging.WARNING):
        actual = await backend.WhisperXAlignmentBackend().align(
            result=result, audio=np.zeros(16000), request_id="fallback-test"
        )
    assert actual == original
    load.assert_not_awaited()
    assert f"language={language}" in caplog.text
    assert "fallback-test" in caplog.text
    assert "ALIGNMENT__MODELS" in caplog.text


@pytest.mark.parametrize(
    ("language", "models"),
    [
        ("en", {}),
        ("ru", {}),
        ("bg", {"bg": "custom"}),
        ("yo", {"multilingual": "custom"}),
    ],
)
async def test_available_model_failures_are_not_suppressed(
    language, models, alignment_config, monkeypatch
):
    alignment_config.models = models
    load = AsyncMock(side_effect=ValueError("model loading failed"))
    monkeypatch.setattr(backend, "load_align_model", load)
    with pytest.raises(ValueError, match="model loading failed"):
        await backend.WhisperXAlignmentBackend().align(
            result={"language": language, "segments": []},
            audio=np.zeros(16000),
            request_id="load-failure-test",
        )
    load.assert_awaited_once_with(language_code=language)


async def test_missing_language_still_fails(alignment_config):
    with pytest.raises(ValueError, match="requires 'language'"):
        await backend.WhisperXAlignmentBackend().align(
            result={"segments": []},
            audio=np.zeros(16000),
            request_id="missing-language",
        )


async def test_diarization_assigns_speakers_after_alignment_fallback(alignment_config):
    import pandas as pd

    result = await backend.WhisperXAlignmentBackend().align(
        result={
            "language": "bg",
            "segments": [{"start": 0.0, "end": 1.0, "text": "test"}],
        },
        audio=np.zeros(16000),
        request_id="diarization-fallback",
    )
    diarization = pd.DataFrame([{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}])
    assigned = backend.whisperx_diarize.assign_word_speakers(diarization, result)
    assert assigned["segments"][0]["speaker"] == "SPEAKER_00"
    assert assigned["segments"][0]["text"] == "test"
    assert "words" not in assigned["segments"][0]
