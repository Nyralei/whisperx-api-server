"""Config loading: nested env delimiter, metrics-enabled alias, mode enum."""

import pytest
from pydantic import ValidationError

from whisperx_api_server.config import DistributedMode
from whisperx_api_server.dependencies import get_config


def _load(monkeypatch, **env):
    for key in ("METRICS_ENABLED", "METRICS__ENABLED", "MODE"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_config.cache_clear()
    return get_config()


def test_nested_env_delimiter(monkeypatch):
    cfg = _load(monkeypatch, WHISPER__MODEL="tiny", WHISPER__BATCH_SIZE="4")
    assert cfg.whisper.model == "tiny"
    assert cfg.whisper.batch_size == 4


def test_nested_kafka_field(monkeypatch):
    cfg = _load(monkeypatch, KAFKA__MAX_DELIVERY_ATTEMPTS="5")
    assert cfg.kafka.max_delivery_attempts == 5


def test_metrics_flat_alias_enables(monkeypatch):
    cfg = _load(monkeypatch, METRICS_ENABLED="true")
    assert cfg.metrics.enabled is True


def test_metrics_nested_takes_precedence_over_flat(monkeypatch):
    cfg = _load(monkeypatch, METRICS_ENABLED="true", METRICS__ENABLED="false")
    assert cfg.metrics.enabled is False


def test_metrics_default_disabled(monkeypatch):
    cfg = _load(monkeypatch)
    assert cfg.metrics.enabled is False


def test_mode_enum(monkeypatch):
    assert _load(monkeypatch, MODE="kafka").mode == DistributedMode.KAFKA
    assert _load(monkeypatch, MODE="direct").mode == DistributedMode.DIRECT


def test_mode_invalid_value_raises(monkeypatch):
    monkeypatch.setenv("MODE", "bogus")
    get_config.cache_clear()
    with pytest.raises(ValidationError):
        get_config()
