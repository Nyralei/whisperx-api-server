"""Shared URL fetch helpers for the audio_url input path.

Both the API (direct mode) and the worker (Kafka mode) use these to:
  * validate the URL (scheme, no userinfo, host not private/loopback/etc.)
  * stream the body with a hard size cap and a total timeout
  * surface bad inputs as InvalidAudioError / UploadTooLargeError so the
    router's existing error mapper returns HTTP 422 / 413 instead of 500.

DNS rebinding is out of scope: getaddrinfo runs once and aiohttp re-resolves
on connect. Defense in depth (resolve-then-pin-IP with explicit Host header)
is a follow-up if the threat model demands it.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import logging
import os
import socket
import tempfile
from urllib.parse import unquote, urlsplit

import aiohttp

from whisperx_api_server.transcriber import (
    _UPLOAD_STREAM_CHUNK_SIZE,
    _UPLOAD_WRITE_BUFFER_SIZE,
    InvalidAudioError,
    UploadTooLargeError,
    _safe_filename,
    _safe_filename_suffix,
)

logger = logging.getLogger(__name__)

_REJECTED_HOST_MSG = "Source URL host is not permitted"
_FETCH_FAILED_MSG = "Could not fetch source URL"


def filename_from_url(url: str, default: str = "audio") -> str:
    """Derive a sanitized filename from a URL's path, falling back to `default`."""
    path = urlsplit(url).path
    base = os.path.basename(unquote(path)) if path else ""
    return _safe_filename(base or None, default=default)


def _redact_url(url: str) -> str:
    """Return a URL with any user:password@ stripped, for safe logging."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return "<unparseable url>"
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return f"{parts.scheme}://{host}{parts.path}"


def _ip_is_unsafe(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


async def _resolve_host(hostname: str) -> list[str]:
    """Resolve a hostname to its IP addresses off the event loop."""
    infos = await asyncio.to_thread(
        socket.getaddrinfo, hostname, None, 0, socket.SOCK_STREAM
    )
    return [str(info[4][0]) for info in infos]


async def validate_url_for_fetch(
    url: str,
    *,
    allow_private_hosts: bool,
    allowed_hosts: list[str] | None,
) -> None:
    """Reject URLs that don't pass the SSRF policy.

    Raises InvalidAudioError on rejection. Never includes the URL in the
    exception message — callers can log a redacted form themselves.
    """
    try:
        parts = urlsplit(url)
    except ValueError as e:
        raise InvalidAudioError("Source URL is malformed") from e

    if parts.scheme not in ("http", "https"):
        raise InvalidAudioError("Source URL scheme must be http or https")

    if parts.username or parts.password:
        raise InvalidAudioError("Source URL must not contain credentials")

    hostname = parts.hostname
    if not hostname:
        raise InvalidAudioError("Source URL has no host")

    allowed = set(h.lower() for h in (allowed_hosts or []))
    if hostname.lower() in allowed:
        return
    if allow_private_hosts:
        return

    try:
        ips = await _resolve_host(hostname)
    except socket.gaierror as e:
        logger.warning(
            "URL validation: DNS resolution failed for %s: %s", _redact_url(url), e
        )
        raise InvalidAudioError("Source URL host could not be resolved") from e

    if any(_ip_is_unsafe(ip) for ip in ips):
        logger.warning(
            "URL validation: rejected host for %s (resolved IPs include private/loopback/link-local)",
            _redact_url(url),
        )
        raise InvalidAudioError(_REJECTED_HOST_MSG)


def _client_timeout(
    connect_timeout: float, total_timeout: float
) -> aiohttp.ClientTimeout:
    return aiohttp.ClientTimeout(total=total_timeout, connect=connect_timeout)


@contextlib.asynccontextmanager
async def _open_get(
    url: str,
    *,
    connect_timeout: float,
    total_timeout: float,
):
    """Open a GET response, mapping errors to InvalidAudioError without leaking the URL.

    Redirects are not followed: a 3xx response is rejected up-front so a public
    allowlisted host cannot trampoline the fetcher to a private/loopback target
    (e.g. 169.254.169.254 cloud metadata). Per-hop revalidation is intentionally
    out of scope.
    """
    timeout = _client_timeout(connect_timeout, total_timeout)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, allow_redirects=False) as response:
                if 300 <= response.status < 400:
                    raise InvalidAudioError(
                        f"Source URL returned a redirect (HTTP {response.status}); redirects are not followed"
                    )
                if response.status >= 400:
                    raise InvalidAudioError(
                        f"Source URL returned HTTP {response.status}"
                    )
                yield response
    except (InvalidAudioError, UploadTooLargeError):
        raise
    except asyncio.TimeoutError as e:
        logger.warning("URL fetch timed out for %s", _redact_url(url))
        raise InvalidAudioError(f"{_FETCH_FAILED_MSG} (timeout)") from e
    except aiohttp.ClientError as e:
        logger.warning(
            "URL fetch failed for %s: %s", _redact_url(url), type(e).__name__
        )
        raise InvalidAudioError(_FETCH_FAILED_MSG) from e


def _check_content_length(headers, max_bytes: int) -> None:
    if max_bytes <= 0:
        return
    raw = headers.get("Content-Length")
    if raw is None:
        return
    try:
        declared = int(raw)
    except ValueError:
        return  # malformed; fall back to streamed enforcement
    if declared > max_bytes:
        raise UploadTooLargeError(
            f"Source exceeds max_upload_size_bytes ({max_bytes} bytes)."
        )


async def download_url_to_temp(
    url: str,
    request_id: str,
    *,
    max_bytes: int,
    connect_timeout: float,
    total_timeout: float,
    allow_private_hosts: bool,
    allowed_hosts: list[str] | None,
) -> str:
    """Stream a validated URL into a temp file. Returns the temp file path."""
    await validate_url_for_fetch(
        url,
        allow_private_hosts=allow_private_hosts,
        allowed_hosts=allowed_hosts,
    )

    suffix = _safe_filename_suffix(filename_from_url(url))
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file_path = tmp.name

    total = 0
    try:
        async with _open_get(
            url,
            connect_timeout=connect_timeout,
            total_timeout=total_timeout,
        ) as response:
            _check_content_length(response.headers, max_bytes)
            with open(file_path, "wb", buffering=_UPLOAD_WRITE_BUFFER_SIZE) as f:
                async for chunk in response.content.iter_chunked(
                    _UPLOAD_STREAM_CHUNK_SIZE
                ):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if max_bytes > 0 and total > max_bytes:
                        raise UploadTooLargeError(
                            f"Source exceeds max_upload_size_bytes ({max_bytes} bytes)."
                        )
                    f.write(chunk)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(file_path)
        raise

    if total == 0:
        with contextlib.suppress(OSError):
            os.remove(file_path)
        raise InvalidAudioError("Source URL returned an empty response.")

    logger.info(
        "Request ID: %s - URL fetch wrote %s bytes to temp file", request_id, total
    )
    return file_path
