"""Filesystem backend against a real shared Docker volume.

The fs backend's target topology is several containers sharing one mount, and two
of its load-bearing assumptions can only be checked there:

* ``check_mount`` must not refuse an ordinary Docker named volume. If it did, a
  correctly configured deployment would fail to start — and the host test suite
  cannot catch it, because it reads the host's mount table, not a container's.
* the ``link()`` create-if-absent idiom must pick exactly one winner when the
  racers are in *different containers*, not merely different processes.

Marked ``kafka``: needs Docker. Run with ``pytest -m kafka``.
"""

import json
import subprocess
import uuid

import pytest

pytestmark = pytest.mark.kafka

_IMAGE = "python:3.12-slim"
_SRC = "/src/whisperx_api_server/storage"
_MOUNT = "/shared"

# Classify the mount the volume is on, using the real module inside the container.
_CLASSIFY = f"""
import json, sys
sys.path.insert(0, "/src")
from whisperx_api_server.storage.mount_check import (
    check_mount, classify, find_mount_for, parse_mountinfo,
)
from whisperx_api_server.storage.contracts import StorageCapabilityError

entries = parse_mountinfo(open("/proc/self/mountinfo", encoding="utf-8").read())
entry = find_mount_for("{_MOUNT}", entries)
verdict, reason = classify(entry) if entry else ("no-entry", "")
try:
    check_mount("{_MOUNT}", allow_unsafe=False)
    refused = None
except StorageCapabilityError as e:
    refused = str(e)
print("RESULT" + json.dumps({{
    "fs_type": entry.fs_type if entry else None,
    "verdict": verdict,
    "reason": reason,
    "refused": refused,
}}))
"""

# The link() idiom from FsObjectStore._put_if_absent_sync, raced across containers.
_RACE = f"""
import os, sys, time, uuid
target = "{_MOUNT}/claims/" + sys.argv[1]
os.makedirs(os.path.dirname(target), exist_ok=True)
tmp = target + ".tmp." + uuid.uuid4().hex
with open(tmp, "wb") as f:
    f.write(b'{{"probe": 1}}')
# Start together so the creates actually overlap.
deadline = float(sys.argv[2])
while time.time() < deadline:
    time.sleep(0.001)
try:
    os.link(tmp, target)
except FileExistsError:
    pass
won = os.stat(tmp).st_nlink == 2
os.unlink(tmp)
print("WON" if won else "LOST")
"""


def _docker(*args, **kwargs):
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=180, **kwargs
    )


@pytest.fixture(scope="module")
def shared_volume():
    probe = _docker("version", "--format", "{{.Server.Version}}")
    if probe.returncode != 0:
        pytest.skip(f"Docker unavailable: {probe.stderr.strip()}")
    if _docker("image", "inspect", _IMAGE).returncode != 0:
        pull = _docker("pull", "--quiet", _IMAGE)
        if pull.returncode != 0:
            pytest.skip(f"cannot pull {_IMAGE}: {pull.stderr.strip()}")

    name = f"wx-shared-{uuid.uuid4().hex[:8]}"
    created = _docker("volume", "create", name)
    if created.returncode != 0:
        pytest.skip(f"cannot create volume: {created.stderr.strip()}")
    try:
        yield name
    finally:
        _docker("volume", "rm", "-f", name)


def _run_in_container(volume: str, script: str, *argv: str, src: str) -> str:
    result = _docker(
        "run",
        "--rm",
        "-v",
        f"{volume}:{_MOUNT}",
        "-v",
        f"{src}:/src:ro",
        _IMAGE,
        "python",
        "-c",
        script,
        *argv,
    )
    assert result.returncode == 0, f"container failed: {result.stderr}"
    return result.stdout


@pytest.fixture(scope="module")
def src_dir(pytestconfig):
    return str(pytestconfig.rootpath / "src")


def test_docker_volume_is_not_refused_by_the_mount_check(shared_volume, src_dir):
    out = _run_in_container(shared_volume, _CLASSIFY, src=src_dir)
    payload = json.loads(out.split("RESULT", 1)[1].strip())

    assert payload["refused"] is None, (
        "check_mount refused an ordinary Docker named volume, which would stop a "
        f"correctly configured deployment from starting: {payload['refused']}"
    )
    # 'warn' is tolerated (an unrecognised type still reaches the probe); a
    # refusal is not. Record the type so a regression is diagnosable.
    assert payload["verdict"] in ("ok", "warn"), payload
    print(f"volume filesystem type: {payload['fs_type']} -> {payload['verdict']}")


def test_exactly_one_container_wins_the_claim(shared_volume, src_dir):
    """Two containers, one shared volume, one claim key."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    job = uuid.uuid4().hex
    start = f"{time.time() + 6.0:.3f}"

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(
                _run_in_container, shared_volume, _RACE, job, start, src=src_dir
            )
            for _ in range(4)
        ]
        outcomes = [f.result().strip() for f in futures]

    assert sorted(outcomes).count("WON") == 1, (
        f"expected exactly one winner across containers, got {outcomes}"
    )
