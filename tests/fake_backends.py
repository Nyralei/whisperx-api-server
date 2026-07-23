"""Fake stage backends registered through the real registry so unit and
integration tests run without torch/whisperx."""

from typing import Any

from whisperx_api_server.backends.registry import (
    register_alignment_backend,
    register_diarization_backend,
    register_transcription_backend,
)

FAKE_SEGMENTS = [{"start": 0.0, "end": 1.0, "text": "hello world"}]


class FakeTranscriptionBackend:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []
        self.raise_exc: BaseException | None = None

    def get_default_model_name(self) -> str:
        return "fake-tiny"

    async def preload_default(self) -> None:
        pass

    def list_loaded_models(self) -> list[str]:
        return ["fake-tiny"]

    async def load_model(self, model_name: str) -> None:
        pass

    async def unload_model(self, model_name: str) -> bool:
        return False

    async def transcribe(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.raise_exc is not None:
            raise self.raise_exc
        return {"segments": [dict(s) for s in FAKE_SEGMENTS], "language": "en"}


class FakeAlignmentBackend:
    async def preload_default(self) -> None:
        pass

    def list_loaded_models(self) -> list[str]:
        return []

    async def load_model(self, model_name: str) -> None:
        pass

    async def unload_model(self, model_name: str) -> bool:
        return False

    async def align(self, *, result, audio, request_id) -> dict[str, Any]:
        return result


class FakeDiarizationBackend:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    async def preload_default(self) -> None:
        pass

    def list_loaded_models(self) -> list[str]:
        return []

    async def load_model(self, model_name: str) -> None:
        pass

    async def unload_model(self, model_name: str) -> bool:
        return False

    async def diarize(self, **kwargs: Any):
        self.calls.append(kwargs)
        return kwargs["result"]


fake_transcription = FakeTranscriptionBackend()
fake_alignment = FakeAlignmentBackend()
fake_diarization = FakeDiarizationBackend()


def register_fake_backends() -> None:
    register_transcription_backend("fake", fake_transcription)
    register_alignment_backend("fake", fake_alignment)
    register_diarization_backend("fake", fake_diarization)
