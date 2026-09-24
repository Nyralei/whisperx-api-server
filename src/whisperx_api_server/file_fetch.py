"""Read audio that another service placed on the shared mount.

Read-only by construction: there is no write or delete path anywhere in this
module, so a future caller cannot be talked into modifying data this service does
not own. The path arrives on the Kafka wire, so it is untrusted input and gets
the same treatment as an inbound URL: validate first, confine to a root, cap the
size, and never let the locator itself select what gets opened.

The confinement is `realpath()` under the configured root. Files this service
owns are excluded too, so an external producer cannot hand the worker a lease or
a result envelope and have it treated as audio.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import stat
import tempfile

from whisperx_api_server.transcriber import (
    InvalidAudioError,
    UploadTooLargeError,
    _safe_filename_suffix,
)

logger = logging.getLogger(__name__)

_COPY_CHUNK_SIZE = 1024 * 1024  # 1 MiB

# Absent on Windows; the dev platform falls back to the realpath check alone.
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_BINARY = getattr(os, "O_BINARY", 0)

_REJECTED_MSG = "Source file path is not permitted"


def _is_under(path: str, directory: str) -> bool:
    directory = directory.rstrip(os.sep)
    if not directory:
        return False
    return path == directory or path.startswith(directory + os.sep)


def validate_input_path(
    candidate: str,
    *,
    root: str,
    owned_subtree: str,
    allowed_dirs: list[str] | None,
) -> str:
    """Return the resolved path, or raise InvalidAudioError.

    Never includes the rejected path in the message; callers log it themselves.
    """
    if not isinstance(candidate, str) or not candidate.strip():
        raise InvalidAudioError("Source file path is empty")
    if "\x00" in candidate:
        raise InvalidAudioError(_REJECTED_MSG)
    if not root.strip():
        raise InvalidAudioError(
            "Filesystem inputs are enabled but no root is configured; set "
            "INPUT_FS__ROOT or STORAGE__FS__ROOT"
        )
    if not os.path.isabs(candidate):
        raise InvalidAudioError("Source file path must be absolute")

    real_root = os.path.realpath(root)
    real = os.path.realpath(candidate)

    if real == real_root or not _is_under(real, real_root):
        logger.warning(
            "Filesystem input rejected: resolved path is outside the configured root"
        )
        raise InvalidAudioError(_REJECTED_MSG)

    if owned_subtree and _is_under(real, os.path.realpath(owned_subtree)):
        logger.warning(
            "Filesystem input rejected: path is inside this service's own subtree"
        )
        raise InvalidAudioError(_REJECTED_MSG)

    if allowed_dirs and not any(
        _is_under(real, os.path.realpath(d)) for d in allowed_dirs
    ):
        logger.warning("Filesystem input rejected: path is outside allowed_dirs")
        raise InvalidAudioError(_REJECTED_MSG)

    return real


def _open_regular_file(path: str) -> int:
    """Open `path` read-only, proving it is still the regular file we validated."""
    try:
        before = os.lstat(path)
    except OSError as e:
        raise InvalidAudioError("Source file does not exist") from e

    if not stat.S_ISREG(before.st_mode):
        # Opening a FIFO blocks until a writer appears, which would park the
        # worker indefinitely; devices, sockets and directories are not audio.
        raise InvalidAudioError("Source path is not a regular file")

    fd = os.open(path, os.O_RDONLY | _O_NOFOLLOW | _O_BINARY)
    try:
        after = os.fstat(fd)
        if not stat.S_ISREG(after.st_mode) or (after.st_dev, after.st_ino) != (
            before.st_dev,
            before.st_ino,
        ):
            # A component was swapped between the check and the open.
            raise InvalidAudioError(_REJECTED_MSG)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _copy_to_temp_sync(real_path: str, suffix: str, max_bytes: int) -> str:
    fd = _open_regular_file(real_path)
    source = os.fdopen(fd, "rb")
    try:
        size = os.fstat(source.fileno()).st_size
        if max_bytes > 0 and size > max_bytes:
            raise UploadTooLargeError(
                f"Source exceeds max_upload_size_bytes ({max_bytes} bytes)."
            )
        if size == 0:
            raise InvalidAudioError("Source file is empty")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            dest = tmp.name
        total = 0
        try:
            with open(dest, "wb") as out:
                while True:
                    chunk = source.read(_COPY_CHUNK_SIZE)
                    if not chunk:
                        break
                    total += len(chunk)
                    if max_bytes > 0 and total > max_bytes:
                        raise UploadTooLargeError(
                            f"Source exceeds max_upload_size_bytes ({max_bytes} bytes)."
                        )
                    out.write(chunk)
        except BaseException:
            with contextlib.suppress(OSError):
                os.remove(dest)
            raise
    finally:
        source.close()
    return dest


async def copy_to_temp(
    candidate: str,
    request_id: str,
    *,
    root: str,
    owned_subtree: str,
    allowed_dirs: list[str] | None,
    max_bytes: int,
) -> str:
    """Copy a validated external file into a temp file. Returns the temp path.

    Always a copy. The pipeline deletes the file it is handed once the audio is
    decoded, and that file must never be the producer's original.
    """
    real_path = validate_input_path(
        candidate,
        root=root,
        owned_subtree=owned_subtree,
        allowed_dirs=allowed_dirs,
    )
    suffix = _safe_filename_suffix(os.path.basename(real_path))
    path = await asyncio.to_thread(_copy_to_temp_sync, real_path, suffix, max_bytes)
    logger.info("Request ID: %s - copied filesystem input into a temp file", request_id)
    return path
