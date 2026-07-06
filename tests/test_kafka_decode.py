"""kafka_client pure-decode helpers: assignment parsing and error rehydration."""

import struct

from whisperx_api_server.kafka_client import (
    _decode_assignment,
    _rehydrate_worker_error,
)
from whisperx_api_server.transcriber import InvalidAudioError, UploadTooLargeError


def _assignment_blob(assignments, version: int = 1) -> bytes:
    out = struct.pack(">h", version)
    out += struct.pack(">i", len(assignments))
    for topic, parts in assignments:
        tb = topic.encode("utf-8")
        out += struct.pack(">h", len(tb)) + tb
        out += struct.pack(">i", len(parts))
        out += struct.pack(f">{len(parts)}i", *parts)
    out += struct.pack(">i", 0)  # empty user_data
    return out


def test_decode_assignment_valid():
    blob = _assignment_blob([("transcription-requests", [0, 1, 2])])
    assert _decode_assignment(blob) == [
        {"topic": "transcription-requests", "partitions": [0, 1, 2]}
    ]


def test_decode_assignment_multiple_topics():
    blob = _assignment_blob([("a", [0]), ("b", [1, 2])])
    decoded = _decode_assignment(blob)
    assert {"topic": "a", "partitions": [0]} in decoded
    assert {"topic": "b", "partitions": [1, 2]} in decoded


def test_decode_assignment_none_and_empty():
    assert _decode_assignment(None) == []
    assert _decode_assignment(b"") == []


def test_decode_assignment_truncated_never_raises():
    blob = _assignment_blob([("a", [0, 1])])
    for cut in range(1, len(blob)):
        result = _decode_assignment(blob[:cut])
        assert isinstance(result, list)


def test_decode_assignment_garbage_never_raises():
    garbage = [
        b"\xff",
        b"\x00\x00\x00",
        b"\x00\x01" + struct.pack(">i", 1_000_000) + b"junk",
        bytes(range(32)),
    ]
    for raw in garbage:
        assert isinstance(_decode_assignment(raw), list)


def test_rehydrate_mapped_types():
    cases = {
        "InvalidAudioError": InvalidAudioError,
        "UploadTooLargeError": UploadTooLargeError,
        "TimeoutError": TimeoutError,
        "ValueError": ValueError,
    }
    for name, cls in cases.items():
        err = _rehydrate_worker_error(name, "boom")
        assert type(err) is cls
        assert str(err) == "boom"


def test_rehydrate_unknown_type_falls_back_to_runtime_error():
    err = _rehydrate_worker_error("SomethingExotic", "boom")
    assert type(err) is RuntimeError
    assert str(err) == "boom"


def test_rehydrate_none_type_is_runtime_error():
    err = _rehydrate_worker_error(None, "boom")
    assert type(err) is RuntimeError
