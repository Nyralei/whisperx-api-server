"""Non-finite floats must never reach a JSON payload (issue #36)."""

import copy
import json
import logging
import math

import numpy as np
import pytest

from whisperx_api_server.formatters import format_transcription
from whisperx_worker.processor import serialize_result

try:
    import whisperx  # noqa: F401

    HAS_WHISPERX = True
except Exception:
    HAS_WHISPERX = False

requires_whisperx = pytest.mark.skipif(
    not HAS_WHISPERX, reason="whisperx (ML extras) not installed"
)

NAN_TRANSCRIPT = {
    "text": "you",
    "language": "en",
    "segments": {
        "segments": [
            {
                "start": math.nan,
                "end": math.inf,
                "text": "you",
                "avg_logprob": math.nan,
                "words": [{"word": "you", "start": 1.9, "score": math.nan}],
            }
        ],
        "speaker_embeddings": {"SPEAKER_00": None},
    },
}


def _strict_loads(payload):
    """Parse rejecting the NaN/Infinity literals stdlib json would otherwise accept."""

    def reject(constant):
        raise AssertionError(f"non-compliant JSON literal: {constant}")

    return json.loads(payload, parse_constant=reject)


def test_verbose_json_renders_non_finite_as_null():
    resp = format_transcription(copy.deepcopy(NAN_TRANSCRIPT), "verbose_json")
    payload = _strict_loads(bytes(resp.body))
    segment = payload["segments"]["segments"][0]
    assert segment["start"] is None
    assert segment["end"] is None
    assert segment["avg_logprob"] is None
    assert segment["words"][0]["score"] is None
    assert segment["words"][0]["start"] == 1.9
    assert payload["segments"]["speaker_embeddings"] == {"SPEAKER_00": None}


def test_serialize_result_renders_non_finite_as_null():
    envelope = serialize_result(
        {"job_id": "j", "status": "ok", "result": copy.deepcopy(NAN_TRANSCRIPT)}
    )
    assert isinstance(envelope, bytes)
    assert b"NaN" not in envelope
    segment = _strict_loads(envelope)["result"]["segments"]["segments"][0]
    assert segment["start"] is None
    assert segment["avg_logprob"] is None


def test_serialize_result_handles_numpy():
    envelope = serialize_result(
        {
            "scalar": np.float32("nan"),
            "count": np.int64(3),
            "vector": np.zeros(2, dtype=np.float32),
        }
    )
    assert _strict_loads(envelope) == {
        "scalar": None,
        "count": 3,
        "vector": [0.0, 0.0],
    }


@requires_whisperx
def test_unusable_speaker_embedding_becomes_null(caplog):
    from whisperx_api_server.backends.whisperx_backend import _drop_unusable_embeddings

    embeddings = {"SPEAKER_00": [math.nan] * 3, "SPEAKER_01": [0.1, -0.2, 0.3]}
    with caplog.at_level(logging.WARNING):
        kept = _drop_unusable_embeddings(embeddings, "req-1")
    assert kept == {"SPEAKER_00": None, "SPEAKER_01": [0.1, -0.2, 0.3]}
    assert "SPEAKER_00" in caplog.text
    assert "SPEAKER_01" not in caplog.text


@requires_whisperx
def test_usable_speaker_embeddings_pass_through():
    from whisperx_api_server.backends.whisperx_backend import _drop_unusable_embeddings

    embeddings = {"SPEAKER_00": [0.1, -0.2]}
    assert _drop_unusable_embeddings(embeddings, "req-2") == embeddings
    assert _drop_unusable_embeddings(None, "req-3") is None
