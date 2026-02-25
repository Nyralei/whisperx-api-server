import importlib
import threading
from typing import Any

from whisperx_api_server.dependencies import get_config

from .contracts import (
    AlignmentBackend,
    BackendSelectionError,
    DiarizationBackend,
    StageBackendSelection,
    TranscriptionBackend,
)

config = get_config()

_transcription_backends: dict[str, TranscriptionBackend] = {}
_alignment_backends: dict[str, AlignmentBackend] = {}
_diarization_backends: dict[str, DiarizationBackend] = {}
_backend_registration_attempted: set[str] = set()
_backend_registration_lock = threading.Lock()


def _normalize_backend_name(backend_name: str, stage: str) -> str:
    normalized = backend_name.strip().lower()
    if not normalized:
        raise BackendSelectionError(f"{stage} backend name cannot be empty.")
    return normalized


def _try_auto_register_backend(backend_name: str) -> None:
    normalized = _normalize_backend_name(backend_name, "backend")
    with _backend_registration_lock:
        if normalized in _backend_registration_attempted:
            return
        _backend_registration_attempted.add(normalized)

    module_name = f"whisperx_api_server.backends.{normalized}_backend"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        if e.name == module_name:
            return
        with _backend_registration_lock:
            _backend_registration_attempted.discard(normalized)
        raise BackendSelectionError(
            f"Failed importing backend module '{module_name}': {e}"
        ) from e
    except Exception as e:
        with _backend_registration_lock:
            _backend_registration_attempted.discard(normalized)
        raise BackendSelectionError(
            f"Failed importing backend module '{module_name}': {e}"
        ) from e

    register_function_name = f"register_{normalized}_backends"
    register_function: Any = getattr(module, register_function_name, None)
    if register_function is None:
        register_function = getattr(module, "register_backends", None)
    if register_function is None or not callable(register_function):
        with _backend_registration_lock:
            _backend_registration_attempted.discard(normalized)
        raise BackendSelectionError(
            f"Backend module '{module_name}' must expose a callable "
            f"'{register_function_name}()' or 'register_backends()'."
        )

    try:
        register_function()
    except Exception as e:
        with _backend_registration_lock:
            _backend_registration_attempted.discard(normalized)
        raise BackendSelectionError(
            f"Failed to register backend '{normalized}': {e}"
        ) from e


def register_transcription_backend(
    backend_name: str,
    backend: TranscriptionBackend,
) -> None:
    _transcription_backends[_normalize_backend_name(
        backend_name, "transcription")] = backend


def register_alignment_backend(
    backend_name: str,
    backend: AlignmentBackend,
) -> None:
    _alignment_backends[_normalize_backend_name(
        backend_name, "alignment")] = backend


def register_diarization_backend(
    backend_name: str,
    backend: DiarizationBackend,
) -> None:
    _diarization_backends[_normalize_backend_name(
        backend_name, "diarization")] = backend


def list_transcription_backends() -> list[str]:
    _try_auto_register_backend(config.backends.transcription)
    return sorted(_transcription_backends.keys())


def list_alignment_backends() -> list[str]:
    _try_auto_register_backend(config.backends.alignment)
    return sorted(_alignment_backends.keys())


def list_diarization_backends() -> list[str]:
    _try_auto_register_backend(config.backends.diarization)
    return sorted(_diarization_backends.keys())


def _build_unknown_backend_error(
    *,
    stage: str,
    backend_name: str,
    available_backends: list[str],
) -> BackendSelectionError:
    available_value = ", ".join(
        available_backends) if available_backends else "none"
    return BackendSelectionError(
        f"Unknown {stage} backend '{backend_name}'. Available backends: {available_value}."
    )


def get_transcription_backend(backend_name: str) -> TranscriptionBackend:
    normalized = _normalize_backend_name(backend_name, "transcription")
    _try_auto_register_backend(normalized)
    backend = _transcription_backends.get(normalized)
    if backend is None:
        raise _build_unknown_backend_error(
            stage="transcription",
            backend_name=normalized,
            available_backends=list_transcription_backends(),
        )
    return backend


def get_alignment_backend(backend_name: str) -> AlignmentBackend:
    normalized = _normalize_backend_name(backend_name, "alignment")
    _try_auto_register_backend(normalized)
    backend = _alignment_backends.get(normalized)
    if backend is None:
        raise _build_unknown_backend_error(
            stage="alignment",
            backend_name=normalized,
            available_backends=list_alignment_backends(),
        )
    return backend


def get_diarization_backend(backend_name: str) -> DiarizationBackend:
    normalized = _normalize_backend_name(backend_name, "diarization")
    _try_auto_register_backend(normalized)
    backend = _diarization_backends.get(normalized)
    if backend is None:
        raise _build_unknown_backend_error(
            stage="diarization",
            backend_name=normalized,
            available_backends=list_diarization_backends(),
        )
    return backend


def resolve_stage_backends() -> StageBackendSelection:
    return StageBackendSelection(
        transcription=_normalize_backend_name(
            config.backends.transcription,
            "transcription",
        ),
        alignment=_normalize_backend_name(
            config.backends.alignment,
            "alignment",
        ),
        diarization=_normalize_backend_name(
            config.backends.diarization,
            "diarization",
        ),
    )


def get_default_transcription_model_name() -> str:
    selected_backends = resolve_stage_backends()
    backend = get_transcription_backend(selected_backends.transcription)
    return backend.get_default_model_name()
