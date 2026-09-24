"""External file_path inputs from third-party producers on the shared mount.

Layout under tmp_path mirrors the deployment shape:

    <root>/files/wav/123.wav   owned by another service — read only
    <root>/whisperx/           owned by this service
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest

from fake_backends import fake_transcription
from whisperx_api_server import file_fetch
from whisperx_api_server.config import (
    Config,
    FsStorageConfig,
    InputFsConfig,
    StorageConfig,
)
from whisperx_api_server.dependencies import get_config
from whisperx_api_server.transcriber import InvalidAudioError, UploadTooLargeError
from whisperx_worker import processor

pytestmark = pytest.mark.anyio


@pytest.fixture
def mount(tmp_path):
    theirs = tmp_path / "files" / "wav"
    theirs.mkdir(parents=True)
    source = theirs / "123.wav"
    source.write_bytes(b"third-party audio payload")

    owned = tmp_path / "whisperx"
    (owned / "results").mkdir(parents=True)
    (owned / "results" / "other-job").write_bytes(b"someone's envelope")

    return SimpleNamespace(
        root=str(tmp_path),
        owned=str(owned),
        source=source,
        their_dir=str(theirs),
    )


def _validate(mount, candidate, allowed_dirs=None):
    return file_fetch.validate_input_path(
        candidate,
        root=mount.root,
        owned_subtree=mount.owned,
        allowed_dirs=allowed_dirs,
    )


async def _copy(mount, candidate, *, allowed_dirs=None, max_bytes=0):
    return await file_fetch.copy_to_temp(
        candidate,
        "req-1",
        root=mount.root,
        owned_subtree=mount.owned,
        allowed_dirs=allowed_dirs,
        max_bytes=max_bytes,
    )


# --------------------------------------------------------------------------
# Confinement
# --------------------------------------------------------------------------


def test_a_file_under_the_root_is_accepted(mount):
    assert _validate(mount, str(mount.source)) == os.path.realpath(str(mount.source))


@pytest.mark.parametrize(
    "candidate",
    [
        "/etc/passwd",
        "../etc/passwd",
        "files/wav/123.wav",
        "",
        "   ",
        "/etc/pass\x00wd",
    ],
)
def test_paths_outside_the_root_or_not_absolute_are_rejected(mount, candidate):
    with pytest.raises(InvalidAudioError):
        _validate(mount, candidate)


def test_traversal_out_of_the_root_is_rejected(mount, tmp_path):
    outside = tmp_path.parent / "outside.wav"
    outside.write_bytes(b"x")
    try:
        with pytest.raises(InvalidAudioError):
            _validate(mount, os.path.join(mount.root, "..", outside.name))
    finally:
        outside.unlink(missing_ok=True)


def test_the_root_itself_is_rejected(mount):
    with pytest.raises(InvalidAudioError):
        _validate(mount, mount.root)


def test_our_own_subtree_is_rejected(mount):
    """An external producer must not hand us a result envelope or a lease."""
    with pytest.raises(InvalidAudioError):
        _validate(mount, os.path.join(mount.owned, "results", "other-job"))


@pytest.mark.skipif(
    os.name == "nt", reason="symlink creation needs elevation on Windows"
)
def test_symlink_inside_the_root_pointing_out_is_rejected(mount, tmp_path):
    secret = tmp_path.parent / "secret.txt"
    secret.write_bytes(b"secret")
    link = os.path.join(mount.their_dir, "sneaky.wav")
    try:
        os.symlink(str(secret), link)
        with pytest.raises(InvalidAudioError):
            _validate(mount, link)
    finally:
        secret.unlink(missing_ok=True)


def test_allowed_dirs_narrows_within_the_root(mount, tmp_path):
    other = tmp_path / "elsewhere"
    other.mkdir()
    stray = other / "a.wav"
    stray.write_bytes(b"x")

    assert _validate(mount, str(mount.source), allowed_dirs=[mount.their_dir])
    with pytest.raises(InvalidAudioError):
        _validate(mount, str(stray), allowed_dirs=[mount.their_dir])


def test_missing_root_configuration_is_refused(mount):
    with pytest.raises(InvalidAudioError):
        file_fetch.validate_input_path(
            str(mount.source), root="", owned_subtree="", allowed_dirs=None
        )


# --------------------------------------------------------------------------
# What may be opened
# --------------------------------------------------------------------------


async def test_a_directory_is_rejected(mount):
    with pytest.raises(InvalidAudioError):
        await _copy(mount, mount.their_dir)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs are POSIX-only")
async def test_a_fifo_is_rejected_without_blocking(mount):
    """Opening a FIFO waits for a writer; that would park the worker forever."""
    fifo = os.path.join(mount.their_dir, "pipe.wav")
    os.mkfifo(fifo)  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(InvalidAudioError):
        await _copy(mount, fifo)


async def test_a_missing_file_is_rejected(mount):
    with pytest.raises(InvalidAudioError):
        await _copy(mount, os.path.join(mount.their_dir, "nope.wav"))


async def test_an_empty_file_is_rejected(mount):
    empty = os.path.join(mount.their_dir, "empty.wav")
    open(empty, "wb").close()
    with pytest.raises(InvalidAudioError):
        await _copy(mount, empty)


async def test_a_file_over_the_size_ceiling_is_rejected(mount):
    with pytest.raises(UploadTooLargeError):
        await _copy(mount, str(mount.source), max_bytes=4)


# --------------------------------------------------------------------------
# Read-only
# --------------------------------------------------------------------------


async def test_copy_leaves_the_source_untouched(mount):
    temp = await _copy(mount, str(mount.source))
    try:
        with open(temp, "rb") as f:
            assert f.read() == b"third-party audio payload"
        assert mount.source.exists()
        assert mount.source.read_bytes() == b"third-party audio payload"
        assert os.stat(temp).st_ino != os.stat(mount.source).st_ino
    finally:
        os.remove(temp)


def test_the_public_surface_is_read_only():
    """Read-only by construction, not by convention: pin the exported API so a
    mutating operation cannot be added here without this failing."""
    exported = {
        name
        for name, value in vars(file_fetch).items()
        if not name.startswith("_")
        and callable(value)
        and getattr(value, "__module__", None) == file_fetch.__name__
    }
    assert exported == {"validate_input_path", "copy_to_temp"}


# --------------------------------------------------------------------------
# Wire mode
# --------------------------------------------------------------------------


def _event(**kwargs):
    base = {"job_id": "j1", "filename": "a.wav", "params": {}}
    base.update(kwargs)
    return base


@pytest.mark.parametrize(
    "event_kwargs",
    [
        {},
        {"s3_key": "audio/j1/a.wav", "audio_url": "http://x/a.wav"},
        {"s3_key": "audio/j1/a.wav", "file_path": "/mnt/a.wav"},
        {"audio_url": "http://x/a.wav", "file_path": "/mnt/a.wav"},
        {"s3_key": "k", "audio_url": "u", "file_path": "/p"},
    ],
)
def test_exactly_one_input_mode_is_required(event_kwargs):
    with pytest.raises(ValueError, match="exactly one"):
        processor._select_input_mode(_event(**event_kwargs), "j1")


@pytest.mark.parametrize(
    "event_kwargs,expected",
    [
        ({"s3_key": "audio/j1/a.wav"}, ("s3_key", "audio/j1/a.wav")),
        ({"audio_url": "http://x/a.wav"}, ("audio_url", "http://x/a.wav")),
        ({"file_path": "/mnt/a.wav"}, ("file_path", "/mnt/a.wav")),
    ],
)
def test_each_input_mode_resolves(event_kwargs, expected):
    assert processor._select_input_mode(_event(**event_kwargs), "j1") == expected


def _fs_config(mount, *, enabled: bool) -> Config:
    return Config(
        storage=StorageConfig(backend="fs", fs=FsStorageConfig(root=mount.root)),
        input_fs=InputFsConfig(enabled=enabled),
    )


async def test_file_path_events_are_refused_when_the_mode_is_disabled(mount):
    with pytest.raises(ValueError, match="INPUT_FS__ENABLED"):
        await processor._fetch_input(
            "file_path",
            str(mount.source),
            config=_fs_config(mount, enabled=False),
            job_id="j1",
            filename="a.wav",
        )


async def test_owned_subtree_is_derived_from_the_storage_config(mount):
    config = _fs_config(mount, enabled=True)
    assert processor._owned_subtree(config) == os.path.join(
        os.path.abspath(mount.root), "whisperx"
    )
    # Not applicable when state lives in S3 rather than on the mount.
    assert processor._owned_subtree(Config()) == ""


async def test_full_job_over_file_path_leaves_the_source_file_on_disk(
    mount, monkeypatch
):
    """The assertion that matters most: a completed job must not consume the
    producer's file, even with delete_after_download on."""
    for key, value in {
        "MODE": "kafka",
        "BACKENDS__TRANSCRIPTION": "fake",
        "BACKENDS__ALIGNMENT": "fake",
        "BACKENDS__DIARIZATION": "fake",
        "STORAGE__BACKEND": "fs",
        "STORAGE__FS__ROOT": mount.root,
        "STORAGE__DELETE_AFTER_DOWNLOAD": "true",
        "INPUT_FS__ENABLED": "true",
    }.items():
        monkeypatch.setenv(key, value)
    get_config.cache_clear()

    async def _fake_load(file_path, request_id, sample_rate=16000):
        # The pipeline deletes what it is handed right after this returns.
        assert os.path.exists(file_path)
        return np.zeros(sample_rate, dtype="float32")

    async def _fake_transcribe(**kwargs):
        return {"segments": [], "language": "en"}

    monkeypatch.setattr(processor, "load_audio_from_path", _fake_load)
    monkeypatch.setattr(fake_transcription, "transcribe", _fake_transcribe)

    try:
        result = await processor.process_job(
            {
                "job_id": "j-external",
                "file_path": str(mount.source),
                "filename": "123.wav",
                "params": {},
            }
        )
    finally:
        get_config.cache_clear()

    assert result is not None
    assert mount.source.exists(), "the producer's file must survive the job"
    assert mount.source.read_bytes() == b"third-party audio payload"
