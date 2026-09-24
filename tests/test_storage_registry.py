"""Storage backend name resolution."""

import pytest

from whisperx_api_server.config import Config, FsStorageConfig, StorageConfig
from whisperx_api_server.storage import registry
from whisperx_api_server.storage.contracts import StorageSelectionError


def test_builtin_backends_resolve(tmp_path):
    cfg = Config(
        storage=StorageConfig(backend="fs", fs=FsStorageConfig(root=str(tmp_path)))
    )
    assert registry.create_store("fs", cfg).name == "fs"
    assert registry.create_store("s3", cfg).name == "s3"


def test_name_is_normalized(tmp_path):
    cfg = Config(
        storage=StorageConfig(backend="fs", fs=FsStorageConfig(root=str(tmp_path)))
    )
    assert registry.create_store("  FS  ", cfg).name == "fs"


def test_unknown_backend_lists_what_is_available():
    with pytest.raises(StorageSelectionError) as exc:
        registry.create_store("azure", Config())
    message = str(exc.value)
    assert "azure" in message
    assert "fs" in message and "s3" in message
    assert "STORAGE__BACKEND" in message


@pytest.mark.parametrize("name", ["", "   ", "fs!", "My-Store", "fs/../x"])
def test_invalid_names_are_rejected(name):
    with pytest.raises(StorageSelectionError):
        registry.create_store(name, Config())
