"""Name -> ObjectStore factory registry.

Factories rather than instances: a store holds a connection and needs
per-process config at construction time.
"""

from __future__ import annotations

import importlib
import re
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .contracts import ObjectStore, StorageSelectionError

if TYPE_CHECKING:
    from whisperx_api_server.config import Config

StoreFactory = Callable[["Config"], ObjectStore]

_VALID_STORE_NAME = re.compile(r"^[a-z0-9_]+$")

_stores: dict[str, StoreFactory] = {}
_registration_attempted: set[str] = set()
_registration_lock = threading.RLock()


def _normalize_store_name(store_name: str) -> str:
    normalized = store_name.strip().lower()
    if not normalized:
        raise StorageSelectionError("Storage backend name cannot be empty.")
    if not _VALID_STORE_NAME.match(normalized):
        raise StorageSelectionError(
            f"Invalid storage backend name {normalized!r}: only lowercase letters, "
            "digits, and underscores are allowed."
        )
    return normalized


def register_store(store_name: str, factory: StoreFactory) -> None:
    _stores[_normalize_store_name(store_name)] = factory


def _try_auto_register_store(store_name: str) -> None:
    normalized = _normalize_store_name(store_name)
    with _registration_lock:
        if normalized in _registration_attempted:
            return

        module_name = f"whisperx_api_server.storage.{normalized}_store"
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            if e.name == module_name:
                _registration_attempted.add(normalized)
                return
            raise StorageSelectionError(
                f"Failed importing storage module '{module_name}': {e}. "
                "If this is a missing dependency, install the kafka extras "
                "(whisperx-api-server[kafka])."
            ) from e
        except Exception as e:
            raise StorageSelectionError(
                f"Failed importing storage module '{module_name}': {e}"
            ) from e

        register_function_name = f"register_{normalized}_store"
        register_function: Any = getattr(module, register_function_name, None)
        if register_function is None or not callable(register_function):
            raise StorageSelectionError(
                f"Storage module '{module_name}' must expose a callable "
                f"'{register_function_name}()'."
            )

        try:
            register_function()
        except Exception as e:
            raise StorageSelectionError(
                f"Failed to register storage backend '{normalized}': {e}"
            ) from e

        _registration_attempted.add(normalized)


def list_stores() -> list[str]:
    return sorted(_stores.keys())


def create_store(store_name: str, config: Config) -> ObjectStore:
    normalized = _normalize_store_name(store_name)
    _try_auto_register_store(normalized)
    factory = _stores.get(normalized)
    if factory is None:
        available = ", ".join(list_stores()) or "none"
        raise StorageSelectionError(
            f"Unknown storage backend '{normalized}'. Available backends: {available}. "
            "Set STORAGE__BACKEND to one of these."
        )
    return factory(config)
