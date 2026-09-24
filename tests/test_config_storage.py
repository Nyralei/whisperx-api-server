"""Storage configuration parsing and the delete_after_download back-fill."""

import pytest
from pydantic import ValidationError

from whisperx_api_server.config import Config, FsStorageConfig, S3Config, StorageConfig


@pytest.fixture
def clean_env(monkeypatch):
    for key in (
        "STORAGE__BACKEND",
        "STORAGE__DELETE_AFTER_DOWNLOAD",
        "STORAGE__FS__ROOT",
        "STORAGE__FS__PREFIX",
        "STORAGE__FS__DIR_MODE",
        "STORAGE__FS__RETENTION_DAYS",
        "S3__DELETE_AFTER_DOWNLOAD",
    ):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_defaults_keep_the_s3_backend(clean_env):
    config = Config()
    assert config.storage.backend == "s3"
    assert config.storage.delete_after_download is True


def test_two_level_env_nesting_lands(clean_env):
    clean_env.setenv("STORAGE__BACKEND", "fs")
    clean_env.setenv("STORAGE__FS__ROOT", "/mnt/shared")
    clean_env.setenv("STORAGE__FS__RETENTION_DAYS", "7")

    config = Config()

    assert config.storage.backend == "fs"
    assert config.storage.fs.root == "/mnt/shared"
    assert config.storage.fs.retention_days == 7


def test_legacy_s3_delete_flag_is_honoured(clean_env):
    clean_env.setenv("S3__DELETE_AFTER_DOWNLOAD", "false")
    assert Config().storage.delete_after_download is False


def test_explicit_storage_flag_beats_the_legacy_one(clean_env):
    clean_env.setenv("S3__DELETE_AFTER_DOWNLOAD", "false")
    clean_env.setenv("STORAGE__DELETE_AFTER_DOWNLOAD", "true")
    assert Config().storage.delete_after_download is True


def test_backfill_also_works_for_programmatic_construction():
    config = Config(s3=S3Config(delete_after_download=False))
    assert config.storage.delete_after_download is False


def test_octal_modes_are_parsed_as_octal(clean_env):
    clean_env.setenv("STORAGE__FS__DIR_MODE", "770")
    config = Config()
    assert config.storage.fs.dir_mode == "770"
    assert int(config.storage.fs.dir_mode, 8) == 0o770
    # The trap the string type exists to avoid.
    assert int("770") != 0o770


@pytest.mark.parametrize("mode", ["0o770", "778", "77", "rwx", "12345", ""])
def test_invalid_modes_are_rejected(mode):
    with pytest.raises(ValidationError):
        FsStorageConfig(root="/mnt", dir_mode=mode)


@pytest.mark.parametrize("prefix", ["", "   ", "/", "a/b", "..", "with space"])
def test_invalid_prefixes_are_rejected(prefix):
    with pytest.raises(ValidationError):
        FsStorageConfig(root="/mnt", prefix=prefix)


def test_prefix_is_stripped_of_slashes():
    assert FsStorageConfig(root="/mnt", prefix="/whisperx/").prefix == "whisperx"


def test_storage_config_defaults_are_independent_instances():
    a = StorageConfig()
    b = StorageConfig()
    a.fs.root = "/mnt/a"
    assert b.fs.root == ""
