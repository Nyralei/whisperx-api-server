"""format_transcription across every response_format + the segments branches."""

import json

import pytest

from whisperx_api_server.formatters import format_transcription, handle_whisperx_format

try:
    import whisperx.utils  # noqa: F401

    HAS_WHISPERX = True
except Exception:
    HAS_WHISPERX = False

requires_whisperx = pytest.mark.skipif(
    not HAS_WHISPERX, reason="whisperx (ML extras) not installed"
)

TRANSCRIPT = {
    "text": "hello world",
    "language": "en",
    "segments": [{"start": 0.0, "end": 1.0, "text": "hello world"}],
}

_WRITER_OPTIONS = {
    "max_line_width": 1000,
    "max_line_count": None,
    "highlight_words": False,
}


def test_json_format():
    resp = format_transcription(dict(TRANSCRIPT), "json")
    assert resp.media_type == "application/json"
    assert json.loads(bytes(resp.body)) == {"text": "hello world"}


def test_verbose_json_format():
    resp = format_transcription(dict(TRANSCRIPT), "verbose_json")
    assert resp.media_type == "application/json"
    assert json.loads(bytes(resp.body))["segments"][0]["text"] == "hello world"


def test_text_format():
    resp = format_transcription(dict(TRANSCRIPT), "text")
    assert resp.media_type == "text/plain"
    assert resp.body == b"hello world"


def test_unsupported_format_raises():
    with pytest.raises(ValueError, match="Unsupported format"):
        format_transcription(dict(TRANSCRIPT), "bogus")


@requires_whisperx
def test_srt_format():
    resp = format_transcription(dict(TRANSCRIPT), "srt")
    assert resp.media_type == "text/plain"
    assert b"-->" in resp.body
    assert b"hello world" in resp.body


@requires_whisperx
def test_vtt_format():
    resp = format_transcription(dict(TRANSCRIPT), "vtt")
    assert resp.media_type == "text/vtt"
    assert b"WEBVTT" in resp.body


@requires_whisperx
def test_vtt_json_format():
    resp = format_transcription(dict(TRANSCRIPT), "vtt_json")
    assert resp.media_type == "application/json"
    payload = json.loads(bytes(resp.body))
    assert "WEBVTT" in payload["vtt_text"]


@requires_whisperx
def test_aud_format():
    resp = format_transcription(dict(TRANSCRIPT), "aud")
    assert resp.media_type == "text/plain"
    assert b"hello world" in resp.body


@requires_whisperx
def test_handle_whisperx_format_dict_segments():
    transcript = {
        "segments": {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hi"}],
            "language": "en",
        }
    }
    out = handle_whisperx_format(transcript, "srt", _WRITER_OPTIONS)
    assert "hi" in out


@requires_whisperx
def test_handle_whisperx_format_invalid_payload():
    with pytest.raises(ValueError, match="must be a list or dict"):
        handle_whisperx_format({"segments": None}, "srt", _WRITER_OPTIONS)
