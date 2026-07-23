"""On-disk store of finished direct-mode transcription results.

Direct mode returns synchronously and has no durable S3 store like Kafka mode.
Persisting the verbose result to a small JSON file lets GET .../result re-format a
finished job (e.g. export srt/aud) without re-running the pipeline, while keeping
resident memory flat. Bounded on disk by entry count and TTL; best-effort,
single-replica, and not expected to survive a reboot.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import tempfile
import threading
import time
from typing import Any

from whisperx_api_server.dependencies import get_config

logger = logging.getLogger(__name__)

_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_lock = threading.Lock()


def _json_default(obj: Any) -> Any:
    item = getattr(obj, "item", None)
    if callable(item):
        with contextlib.suppress(Exception):
            return obj.item()
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _resolve_dir(cfg: Any) -> str:
    directory = cfg.dir or os.path.join(tempfile.gettempdir(), "whisperx-results")
    os.makedirs(directory, exist_ok=True)
    return directory


def _enforce_capacity(directory: str, cap: int) -> None:
    if cap <= 0:
        return
    with _lock:
        try:
            paths = [
                os.path.join(directory, n)
                for n in os.listdir(directory)
                if n.endswith(".json")
            ]
        except OSError:
            return
        if len(paths) <= cap:
            return
        dated = []
        for p in paths:
            with contextlib.suppress(OSError):
                dated.append((os.path.getmtime(p), p))
        dated.sort()
        for _, p in dated[: len(dated) - cap]:
            with contextlib.suppress(OSError):
                os.remove(p)


def put(request_id: str, result: dict[str, Any]) -> None:
    """Persist a finished result when the store is enabled. Best-effort; never raises."""
    if not request_id or not _SAFE_ID.match(request_id):
        return
    cfg = get_config().result_store
    if not cfg.enabled:
        return
    directory = _resolve_dir(cfg)
    path = os.path.join(directory, f"{request_id}.json")
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(result, f, default=_json_default)
        os.replace(tmp, path)
    except Exception:
        logger.warning("result_store: failed to persist %s", request_id, exc_info=True)
        with contextlib.suppress(OSError):
            os.remove(tmp)
        return
    _enforce_capacity(directory, cfg.max_entries)


def get(request_id: str) -> dict[str, Any] | None:
    """Return the stored result, or None if absent/expired. TTL enforced on read."""
    if not request_id or not _SAFE_ID.match(request_id):
        return None
    cfg = get_config().result_store
    path = os.path.join(_resolve_dir(cfg), f"{request_id}.json")
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    if cfg.ttl_seconds > 0 and time.time() - mtime > cfg.ttl_seconds:
        with contextlib.suppress(OSError):
            os.remove(path)
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.warning("result_store: failed to read %s", request_id, exc_info=True)
        return None


def evict_expired(now: float | None = None) -> int:
    """Delete result files older than ttl_seconds. Returns count removed."""
    cfg = get_config().result_store
    if cfg.ttl_seconds <= 0:
        return 0
    directory = _resolve_dir(cfg)
    cutoff = (now if now is not None else time.time()) - cfg.ttl_seconds
    removed = 0
    with _lock:
        try:
            names = os.listdir(directory)
        except OSError:
            return 0
        for n in names:
            if not n.endswith(".json"):
                continue
            p = os.path.join(directory, n)
            try:
                if os.path.getmtime(p) <= cutoff:
                    os.remove(p)
                    removed += 1
            except OSError:
                continue
    return removed


def _reset_for_tests() -> None:
    directory = _resolve_dir(get_config().result_store)
    with contextlib.suppress(OSError):
        for n in os.listdir(directory):
            if n.endswith(".json"):
                with contextlib.suppress(OSError):
                    os.remove(os.path.join(directory, n))
