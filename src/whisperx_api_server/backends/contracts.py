from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from whisperx_api_server.config import Language


class BackendSelectionError(ValueError):
    """Raised when a requested stage backend is unknown or invalid."""


@dataclass(frozen=True)
class StageBackendSelection:
    transcription: str
    alignment: str
    diarization: str


class TranscriptionBackend(Protocol):
    def get_default_model_name(self) -> str:
        ...

    async def preload_default(self) -> None:
        ...

    def list_loaded_models(self) -> list[str]:
        ...

    async def load_model(self, model_name: str) -> None:
        ...

    def unload_model(self, model_name: str) -> bool:
        ...

    async def transcribe(
        self,
        *,
        model_name: str,
        audio: np.ndarray,
        batch_size: int,
        chunk_size: int,
        language: Language | None,
        task: str,
        asr_options: dict[str, Any] | None,
        request_id: str,
    ) -> dict[str, Any]:
        ...


class AlignmentBackend(Protocol):
    async def preload_default(self) -> None:
        ...

    def list_loaded_models(self) -> list[str]:
        ...

    async def load_model(self, model_name: str) -> None:
        ...

    def unload_model(self, model_name: str) -> bool:
        ...

    async def align(
        self,
        *,
        result: dict[str, Any],
        audio: np.ndarray,
        request_id: str,
    ) -> dict[str, Any]:
        ...


class DiarizationBackend(Protocol):
    async def preload_default(self) -> None:
        ...

    def list_loaded_models(self) -> list[str]:
        ...

    async def load_model(self, model_name: str) -> None:
        ...

    def unload_model(self, model_name: str) -> bool:
        ...

    async def diarize(
        self,
        *,
        result: dict[str, Any],
        audio: np.ndarray,
        speaker_embeddings: bool,
        request_id: str,
    ) -> dict[str, Any]:
        ...
