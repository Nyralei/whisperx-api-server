"""Filename sanitization and URL SSRF policy helpers."""

import re

import pytest

from whisperx_api_server.transcriber import _safe_filename
from whisperx_api_server.url_fetch import (
    InvalidAudioError,
    _ip_is_unsafe,
    filename_from_url,
    validate_url_for_fetch,
)

_SAFE = re.compile(r"^[A-Za-z0-9._-]+$")


def test_safe_filename_strips_path_traversal():
    out = _safe_filename("../../etc/passwd")
    assert out == "passwd"
    assert "/" not in out


def test_safe_filename_sanitizes_separators_and_unicode():
    out = _safe_filename("..\\..\\café.wav")
    assert _SAFE.match(out)
    assert "\\" not in out


def test_safe_filename_empty_and_none_use_default():
    assert _safe_filename("") == "audio"
    assert _safe_filename(None) == "audio"
    assert _safe_filename(None, default="x") == "x"


def test_safe_filename_truncates_overlong():
    out = _safe_filename("a" * 200 + ".wav")
    assert len(out) <= 64


def test_filename_from_url_basic():
    assert filename_from_url("http://host/path/to/file.wav") == "file.wav"


def test_filename_from_url_empty_path_default():
    assert filename_from_url("http://host/") == "audio"


def test_filename_from_url_unquotes_and_drops_traversal():
    assert filename_from_url("http://host/%2e%2e%2ffoo.wav") == "foo.wav"


def test_filename_from_url_ignores_query():
    assert filename_from_url("http://host/a.wav?token=1") == "a.wav"


def test_ip_is_unsafe_classifies_addresses():
    for unsafe in (
        "127.0.0.1",
        "10.0.0.1",
        "192.168.1.1",
        "169.254.1.1",
        "::1",
        "224.0.0.1",
        "0.0.0.0",
        "not-an-ip",
    ):
        assert _ip_is_unsafe(unsafe) is True
    for safe in ("8.8.8.8", "1.1.1.1"):
        assert _ip_is_unsafe(safe) is False


@pytest.mark.anyio
async def test_validate_url_rejects_non_http_scheme():
    with pytest.raises(InvalidAudioError, match="scheme"):
        await validate_url_for_fetch(
            "ftp://host/x", allow_private_hosts=False, allowed_hosts=None
        )


@pytest.mark.anyio
async def test_validate_url_rejects_userinfo():
    with pytest.raises(InvalidAudioError, match="credentials"):
        await validate_url_for_fetch(
            "http://user:pass@host/x", allow_private_hosts=False, allowed_hosts=None
        )


@pytest.mark.anyio
async def test_validate_url_rejects_missing_host():
    with pytest.raises(InvalidAudioError, match="no host"):
        await validate_url_for_fetch(
            "http:///x", allow_private_hosts=False, allowed_hosts=None
        )


@pytest.mark.anyio
async def test_validate_url_rejects_loopback_ip():
    with pytest.raises(InvalidAudioError, match="not permitted"):
        await validate_url_for_fetch(
            "http://127.0.0.1/x", allow_private_hosts=False, allowed_hosts=None
        )


@pytest.mark.anyio
async def test_validate_url_allowed_host_bypasses_policy():
    await validate_url_for_fetch(
        "http://127.0.0.1/x",
        allow_private_hosts=False,
        allowed_hosts=["127.0.0.1"],
    )


@pytest.mark.anyio
async def test_validate_url_allow_private_hosts_bypasses_policy():
    await validate_url_for_fetch(
        "http://127.0.0.1/x", allow_private_hosts=True, allowed_hosts=None
    )
