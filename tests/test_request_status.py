"""Lifecycle tracker: stage timing, terminal stickiness, TTL, capacity, timeline."""

import threading

from whisperx_api_server import request_status
from whisperx_api_server.dependencies import get_config

WORKER_TIMELINE = {
    "transcribe": {"started_at": 1000.0, "completed_at": 1003.0},
    "align": {"started_at": 1003.0, "completed_at": 1004.0},
}


class _Clock:
    def __init__(self, t: float):
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_stage_open_close_duration(monkeypatch):
    clock = _Clock(1000.0)
    monkeypatch.setattr(request_status, "_now", clock)

    request_status.start("r", mode="direct")
    clock.t = 1005.0
    request_status.set_stage("r", "transcribe")
    clock.t = 1008.0
    request_status.set_stage("r", "align")
    clock.t = 1010.0
    request_status.mark_completed("r")

    st = request_status.get("r")
    assert st is not None
    assert st["status"] == "completed"
    assert st["stages"][0]["name"] == "transcribe"
    assert st["stages"][0]["in_progress"] is False
    assert st["stages"][0]["duration_seconds"] == 3.0
    assert st["stages"][1]["name"] == "align"
    assert st["stages"][1]["duration_seconds"] == 2.0


def test_terminal_states_are_sticky():
    request_status.start("r", mode="direct")
    request_status.mark_completed("r")
    request_status.set_stage("r", "late")
    request_status.mark_failed("r", "boom", "RuntimeError")

    st = request_status.get("r")
    assert st is not None
    assert st["status"] == "completed"
    assert st["error"] is None
    assert all(s["name"] != "late" for s in st["stages"])


def test_ttl_retains_then_evicts():
    request_status.start("r", mode="direct")
    request_status.mark_completed("r")
    st = request_status.get("r")
    assert st is not None
    done = st["completed_at"]

    assert request_status.evict_expired(now=done + 299.0) == 0
    assert request_status.get("r") is not None
    assert request_status.evict_expired(now=done + 301.0) == 1
    assert request_status.get("r") is None


def test_inflight_never_ttl_evicted():
    request_status.start("r", mode="direct")
    assert request_status.evict_expired(now=1e12) == 0
    assert request_status.get("r") is not None


def test_capacity_prefers_terminal(monkeypatch):
    monkeypatch.setenv("REQUEST_STATUS__MAX_ENTRIES", "2")
    get_config.cache_clear()

    request_status.start("a", mode="direct")
    request_status.mark_completed("a")
    request_status.start("b", mode="direct")
    request_status.start("c", mode="direct")

    assert request_status.get("a") is None
    assert request_status.get("b") is not None
    assert request_status.get("c") is not None


def test_capacity_falls_back_to_inflight(monkeypatch):
    monkeypatch.setenv("REQUEST_STATUS__MAX_ENTRIES", "1")
    get_config.cache_clear()

    request_status.start("x", mode="direct")
    request_status.start("y", mode="direct")

    assert request_status.get("x") is None
    assert request_status.get("y") is not None


def test_apply_worker_timeline_unknown_id_noop():
    request_status.apply_worker_timeline("ghost", WORKER_TIMELINE)
    assert request_status.get("ghost") is None


def test_timeline_then_complete():
    request_status.start("r", mode="kafka")
    request_status.set_stage("r", "awaiting_worker")
    request_status.apply_worker_timeline("r", WORKER_TIMELINE)
    request_status.mark_completed("r")

    st = request_status.get("r")
    assert st is not None
    names = [s["name"] for s in st["stages"]]
    assert st["status"] == "completed"
    assert "worker.transcribe" in names
    assert "worker.align" in names
    wt = next(s for s in st["stages"] if s["name"] == "worker.transcribe")
    assert wt["duration_seconds"] == 3.0


def test_complete_then_timeline():
    request_status.start("r", mode="kafka")
    request_status.set_stage("r", "awaiting_worker")
    request_status.mark_completed("r")
    request_status.apply_worker_timeline("r", WORKER_TIMELINE)

    st = request_status.get("r")
    assert st is not None
    names = [s["name"] for s in st["stages"]]
    assert st["status"] == "completed"
    assert "worker.transcribe" in names


def test_timeline_is_idempotent():
    request_status.start("r", mode="kafka")
    request_status.apply_worker_timeline("r", WORKER_TIMELINE)
    request_status.apply_worker_timeline("r", WORKER_TIMELINE)

    st = request_status.get("r")
    assert st is not None
    assert sum(s["name"] == "worker.transcribe" for s in st["stages"]) == 1


def test_timeline_closes_preceding_stage_at_handover(monkeypatch):
    clock = _Clock(900.0)
    monkeypatch.setattr(request_status, "_now", clock)

    request_status.start("r", mode="kafka")
    clock.t = 950.0
    request_status.set_stage("r", "awaiting_worker")
    request_status.apply_worker_timeline("r", WORKER_TIMELINE)

    st = request_status.get("r")
    assert st is not None
    aw = next(s for s in st["stages"] if s["name"] == "awaiting_worker")
    assert aw["in_progress"] is False
    assert aw["completed_at"] == 1000.0
    assert aw["duration_seconds"] == 50.0


def test_thread_safety_smoke():
    ids = [f"job-{i}" for i in range(20)]

    def work(rid: str) -> None:
        request_status.start(rid, mode="direct")
        request_status.set_stage(rid, "transcribe")
        request_status.mark_completed(rid)

    threads = [threading.Thread(target=work, args=(r,)) for r in ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for r in ids:
        st = request_status.get(r)
        assert st is not None and st["status"] == "completed"
