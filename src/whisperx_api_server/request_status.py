"""In-memory per-request lifecycle tracker.

Records stage transitions for each transcription request so callers can poll
GET /v1/audio/transcriptions/{request_id}/status. Terminal states (completed,
failed) are retained for `request_status.ttl_seconds` so polling clients that
arrive after the POST resolves can still confirm the outcome.

Single-replica scope: state lives in this process only. In Kafka mode, worker
stages arrive over a progress topic and update the tracker on the replica that
submitted the job. Other replicas never see the state for that job_id.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import threading
import time
from typing import Any

from whisperx_api_server.dependencies import get_config

logger = logging.getLogger(__name__)

_STATUS_QUEUED = "queued"
_STATUS_IN_PROGRESS = "in_progress"
_STATUS_COMPLETED = "completed"
_STATUS_FAILED = "failed"

# Plain (non-async) lock — state mutations are quick and may be invoked from
# both async and sync paths (e.g., transcriber finally blocks). asyncio.Lock
# would force callers to be awaitable; threading.Lock keeps the API ergonomic.
_lock = threading.Lock()
_states: dict[str, dict[str, Any]] = {}


def _now() -> float:
    return time.time()


def _is_terminal(state: dict[str, Any]) -> bool:
    return state.get("status") in (_STATUS_COMPLETED, _STATUS_FAILED)


def _close_active_stage(state: dict[str, Any], at: float) -> None:
    """Mark the most recent stage as done and record its duration."""
    stages = state.get("stages") or []
    if not stages:
        return
    last = stages[-1]
    if last.get("in_progress"):
        last["in_progress"] = False
        last["completed_at"] = at
        started = last.get("started_at")
        if started is not None:
            last["duration_seconds"] = round(at - started, 4)


def _enforce_capacity_locked(cap: int) -> None:
    """Drop entries when over capacity. Caller holds _lock."""
    if cap <= 0 or len(_states) <= cap:
        return
    # Prefer to drop terminal entries (oldest first); fall back to in-flight
    # only if we still need room. updated_at is the eviction key.
    items = sorted(_states.items(), key=lambda kv: kv[1].get("updated_at", 0.0))
    terminal_ids = [rid for rid, st in items if _is_terminal(st)]
    other_ids = [rid for rid, st in items if not _is_terminal(st)]
    to_remove = len(_states) - cap
    for rid in terminal_ids:
        if to_remove <= 0:
            break
        _states.pop(rid, None)
        to_remove -= 1
    for rid in other_ids:
        if to_remove <= 0:
            break
        _states.pop(rid, None)
        to_remove -= 1


def start(
    request_id: str,
    *,
    mode: str,
    filename: str | None = None,
    params: dict[str, Any] | None = None,
) -> None:
    """Begin tracking a request. Idempotent — repeat calls reset the state."""
    if not request_id:
        return
    cfg = get_config().request_status
    now = _now()
    state: dict[str, Any] = {
        "request_id": request_id,
        "status": _STATUS_QUEUED,
        "mode": mode,
        "stage": _STATUS_QUEUED,
        "submitted_at": now,
        "updated_at": now,
        "completed_at": None,
        "filename": filename,
        "params": params,
        "stages": [],
        "error": None,
        "error_type": None,
    }
    with _lock:
        _states[request_id] = state
        _enforce_capacity_locked(cfg.max_entries)


def set_stage(request_id: str, stage: str) -> None:
    """Mark a new stage as in-progress. Closes the previous stage with its duration."""
    if not request_id or not stage:
        return
    now = _now()
    with _lock:
        state = _states.get(request_id)
        if state is None or _is_terminal(state):
            return
        stages = state.setdefault("stages", [])
        # Idempotency: if this stage was already authoritatively recorded (e.g.
        # apply_worker_timeline ran ahead of a late progress event), skip.
        if any(s.get("name") == stage and not s.get("in_progress") for s in stages):
            return
        _close_active_stage(state, now)
        stages.append({"name": stage, "started_at": now, "in_progress": True})
        state["stage"] = stage
        state["status"] = _STATUS_IN_PROGRESS
        state["updated_at"] = now


def mark_completed(request_id: str) -> None:
    if not request_id:
        return
    now = _now()
    with _lock:
        state = _states.get(request_id)
        if state is None:
            return
        if _is_terminal(state):
            return
        _close_active_stage(state, now)
        state["status"] = _STATUS_COMPLETED
        state["stage"] = _STATUS_COMPLETED
        state["completed_at"] = now
        state["updated_at"] = now


def mark_failed(request_id: str, error: str, error_type: str | None = None) -> None:
    if not request_id:
        return
    now = _now()
    with _lock:
        state = _states.get(request_id)
        if state is None:
            return
        if _is_terminal(state):
            return
        _close_active_stage(state, now)
        state["status"] = _STATUS_FAILED
        state["stage"] = _STATUS_FAILED
        state["error"] = error
        state["error_type"] = error_type
        state["completed_at"] = now
        state["updated_at"] = now


def apply_worker_timeline(
    request_id: str,
    timeline: dict[str, dict[str, float]],
    *,
    prefix: str = "worker.",
) -> None:
    """Reconcile the local stages array with an authoritative worker timeline.

    Called by the API reply consumer when a Kafka worker reply lands. The
    worker's per-stage wall-clock timestamps are the source of truth for
    everything that happened inside the worker — they don't depend on the
    progress topic winning the race against the reply topic.

    Behavior:
      - Removes any existing entries with names starting with ``prefix``
        (provisional entries written by the progress consumer with API-side
        approximate timing).
      - Closes any preceding in-progress API-side stages (e.g. ``awaiting_worker``)
        at the first worker stage's started_at, since that's when control actually
        passed to the worker.
      - Appends authoritative worker stages in insertion order from ``timeline``.
      - Idempotent — safe to call multiple times or with partial timelines.
    """
    if not request_id or not timeline:
        return
    with _lock:
        state = _states.get(request_id)
        if state is None:
            return

        stages: list[dict[str, Any]] = state.setdefault("stages", [])

        # Find the earliest worker stage start — that's when API-side
        # "awaiting_worker" (or any prior in-progress stage) effectively ended.
        worker_starts = [
            e["started_at"]
            for e in timeline.values()
            if e.get("started_at") is not None
        ]
        if not worker_starts:
            return
        handover_ts = min(worker_starts)

        # Close any preceding API-side in-progress stages at the handover boundary.
        for s in stages:
            if s.get("in_progress") and not s.get("name", "").startswith(prefix):
                s["in_progress"] = False
                s["completed_at"] = handover_ts
                started = s.get("started_at")
                if started is not None:
                    s["duration_seconds"] = round(handover_ts - started, 4)

        # Drop any existing worker.* entries; reinsert from the authoritative timeline.
        state["stages"] = [
            s for s in stages if not s.get("name", "").startswith(prefix)
        ]
        for stage_name, entry in timeline.items():
            started_at = entry.get("started_at")
            completed_at = entry.get("completed_at")
            if started_at is None:
                continue
            new_entry: dict[str, Any] = {
                "name": f"{prefix}{stage_name}",
                "started_at": started_at,
                "in_progress": completed_at is None,
            }
            if completed_at is not None:
                new_entry["completed_at"] = completed_at
                new_entry["duration_seconds"] = round(completed_at - started_at, 4)
            state["stages"].append(new_entry)

        state["updated_at"] = _now()


def get(request_id: str) -> dict[str, Any] | None:
    """Return a defensive copy of the state, or None if unknown."""
    if not request_id:
        return None
    with _lock:
        state = _states.get(request_id)
        if state is None:
            return None
        return copy.deepcopy(state)


def evict_expired(now: float | None = None) -> int:
    """Drop terminal entries whose completed_at + ttl < now. Returns count evicted."""
    cfg = get_config().request_status
    ttl = cfg.ttl_seconds
    if ttl <= 0:
        # ttl=0 means terminal states are dropped immediately when seen.
        cutoff = float("inf")
    else:
        cutoff = (now if now is not None else _now()) - ttl
    removed = 0
    with _lock:
        for rid in list(_states.keys()):
            state = _states.get(rid)
            if state is None:
                continue
            completed_at = state.get("completed_at")
            if (
                _is_terminal(state)
                and completed_at is not None
                and completed_at <= cutoff
            ):
                _states.pop(rid, None)
                removed += 1
    return removed


async def cleanup_loop() -> None:
    """Background task: periodically evict expired entries. Runs until cancelled."""
    cfg = get_config().request_status
    interval = cfg.cleanup_interval_seconds
    try:
        while True:
            await asyncio.sleep(interval)
            try:
                n = evict_expired()
                if n:
                    logger.debug("request_status: evicted %d expired entries", n)
            except Exception:
                logger.exception("request_status cleanup sweep failed")
    except asyncio.CancelledError:
        raise


def _reset_for_tests() -> None:
    """Test helper — clear all state."""
    with _lock:
        _states.clear()
