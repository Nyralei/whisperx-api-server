"""Startup refuses a backend that cannot prove atomic create-if-absent."""

import pytest

from storage_fakes import MemoryObjectStore
from whisperx_api_server.config import Config, FsStorageConfig, StorageConfig
from whisperx_api_server.storage import service
from whisperx_api_server.storage.contracts import (
    StorageCapabilityError,
    StorageSelectionError,
)

pytestmark = pytest.mark.anyio


async def test_probe_passes_on_an_atomic_store():
    await service.probe_atomic_create(MemoryObjectStore())


async def test_probe_rejects_a_store_that_always_reports_created():
    store = MemoryObjectStore(atomic=False)
    with pytest.raises(StorageCapabilityError) as exc:
        await service.probe_atomic_create(store)
    message = str(exc.value)
    assert "atomic create-if-absent" in message
    assert "concurrently" in message


async def test_probe_cleans_up_after_itself():
    store = MemoryObjectStore()
    await service.probe_atomic_create(store)
    assert store.objects == {}


async def test_probe_leaves_no_stray_directory_on_the_filesystem(tmp_path):
    config = Config(
        storage=StorageConfig(backend="fs", fs=FsStorageConfig(root=str(tmp_path)))
    )
    try:
        await service.init_storage(config)
    finally:
        await service.close_storage()

    subtree = tmp_path / "whisperx"
    assert sorted(p.name for p in subtree.iterdir()) == ["audio", "claims", "results"]
    assert list((subtree / "claims").iterdir()) == []


async def test_init_storage_probes_and_leaves_no_singleton_on_failure(monkeypatch):
    store = MemoryObjectStore(atomic=False)
    monkeypatch.setattr(service, "create_store", lambda name, config: store)

    with pytest.raises(StorageCapabilityError):
        await service.init_storage(Config())

    assert service.active_store() is None
    assert store.closed is True


async def test_fs_backend_without_a_root_refuses_to_start():
    config = Config(storage=StorageConfig(backend="fs"))
    with pytest.raises(StorageSelectionError, match="STORAGE__FS__ROOT"):
        await service.init_storage(config)
    assert service.active_store() is None


async def test_fs_backend_with_a_missing_mount_refuses_to_start(tmp_path):
    """The mount must exist before the process starts; we never create the root."""
    config = Config(
        storage=StorageConfig(
            backend="fs", fs=FsStorageConfig(root=str(tmp_path / "not-mounted"))
        )
    )
    with pytest.raises(StorageCapabilityError, match="does not exist"):
        await service.init_storage(config)
    assert service.active_store() is None


async def test_unwritable_mount_refuses_with_an_actionable_message(
    tmp_path, monkeypatch
):
    """A fresh Docker named volume is root-owned; the operator needs to be told
    that, not handed a bare PermissionError traceback."""
    from whisperx_api_server.storage import fs_store

    def _denied(path, *args, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(fs_store.os, "makedirs", _denied)
    config = Config(
        storage=StorageConfig(backend="fs", fs=FsStorageConfig(root=str(tmp_path)))
    )
    with pytest.raises(StorageCapabilityError) as exc:
        await service.init_storage(config)

    message = str(exc.value)
    assert "Permission denied" in message
    assert "writable" in message
    assert "root:root" in message
    assert service.active_store() is None


async def test_fs_backend_passes_the_probe_end_to_end(tmp_path):
    config = Config(
        storage=StorageConfig(backend="fs", fs=FsStorageConfig(root=str(tmp_path)))
    )
    try:
        await service.init_storage(config)
        store = service.active_store()
        assert store is not None
        assert store.name == "fs"
    finally:
        await service.close_storage()
