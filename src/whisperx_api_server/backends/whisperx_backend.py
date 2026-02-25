import asyncio
import logging
import time
from dataclasses import replace
from typing import Any

import numpy as np
import torch
from whisperx import alignment as whisperx_alignment
from whisperx import asr as whisperx_asr
from whisperx import diarize as whisperx_diarize

from whisperx_api_server.config import Language
from whisperx_api_server.dependencies import get_config
from .whisperx_runtime import (
    align_model_instances,
    determine_inference_device,
    diarize_pipeline_instances,
    load_align_model,
    load_diarize_pipeline,
    load_transcribe_pipeline,
    transcribe_pipeline_instances,
    unload_model_object,
)

from .registry import (
    register_alignment_backend,
    register_diarization_backend,
    register_transcription_backend,
)

logger = logging.getLogger(__name__)
config = get_config()

_DEFAULT_ASR_OPTIONS = {
    "suppress_numerals": True,
    "temperatures": 0.0,
    "word_timestamps": False,
    "initial_prompt": None,
    "hotwords": None,
}
_registered = False


def _apply_request_options(
    model: whisperx_asr.FasterWhisperPipeline,
    asr_options: dict[str, Any] | None,
) -> None:
    opts = dict(_DEFAULT_ASR_OPTIONS)
    if asr_options:
        opts.update(asr_options)

    temperatures = opts["temperatures"]
    if not isinstance(temperatures, (list, tuple)):
        temperatures = [temperatures]
    else:
        temperatures = list(temperatures)

    overrides = {
        "temperatures": temperatures,
        "word_timestamps": opts["word_timestamps"],
        "initial_prompt": opts["initial_prompt"],
        "hotwords": opts["hotwords"],
    }

    if hasattr(model.options, "__dataclass_fields__"):
        model.options = replace(model.options, **overrides)
    model.suppress_numerals = opts["suppress_numerals"]


class WhisperXTranscriptionBackend:
    def get_default_model_name(self) -> str:
        return config.whisper.model

    async def preload_default(self) -> None:
        if not config.whisper.preload_model:
            return
        await load_transcribe_pipeline(model_name=self.get_default_model_name())

    def list_loaded_models(self) -> list[str]:
        return sorted({cache_key[0] for cache_key in transcribe_pipeline_instances})

    async def load_model(self, model_name: str) -> None:
        await load_transcribe_pipeline(model_name=model_name)

    def unload_model(self, model_name: str) -> bool:
        to_remove = [k for k in transcribe_pipeline_instances if k[0] == model_name]
        for cache_key in to_remove:
            pipeline = transcribe_pipeline_instances.pop(cache_key, None)
            if pipeline is not None:
                unload_model_object(pipeline)
        return bool(to_remove)

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
        loop = asyncio.get_running_loop()
        lang = getattr(language, "value", language) if language else None
        num_workers = 0 if determine_inference_device(
        ) == "cuda" else config.whisper.num_workers

        model_load_start = time.perf_counter()
        model = await load_transcribe_pipeline(model_name=model_name)
        logger.info(
            f"Request ID: {request_id} - Loaded transcription pipeline backend=whisperx model={model_name} in {time.perf_counter() - model_load_start:.2f} seconds"
        )

        lock = getattr(model, "_transcribe_lock", None)

        def _run_transcription() -> dict[str, Any]:
            with torch.inference_mode():
                return model.transcribe(
                    audio=audio,
                    batch_size=batch_size,
                    chunk_size=chunk_size,
                    num_workers=num_workers,
                    language=lang,
                    task=task,
                )

        if lock is not None:
            async with lock:
                _apply_request_options(model, asr_options)
                result = await loop.run_in_executor(None, _run_transcription)
        else:
            _apply_request_options(model, asr_options)
            result = await loop.run_in_executor(None, _run_transcription)

        logger.info(
            f"Request ID: {request_id} - Transcription completed using backend=whisperx")
        return result


class WhisperXAlignmentBackend:
    async def preload_default(self) -> None:
        if not config.alignment.preload_model:
            return
        if config.alignment.preload_model_name:
            logger.info(
                f"Preloading alignment model for: {config.alignment.preload_model_name}")
            await load_align_model(config.alignment.preload_model_name)
            return

        for lang in config.alignment.whitelist:
            logger.info(f"Preloading alignment model for: {lang}")
            await load_align_model(lang)

    def list_loaded_models(self) -> list[str]:
        return sorted(str(key) for key in align_model_instances.keys())

    async def load_model(self, model_name: str) -> None:
        await load_align_model(model_name)

    def unload_model(self, model_name: str) -> bool:
        align_model_data = align_model_instances.pop(model_name, None)
        if align_model_data is None:
            return False
        unload_model_object(align_model_data.get("model"))
        return True

    async def align(
        self,
        *,
        result: dict[str, Any],
        audio: np.ndarray,
        request_id: str,
    ) -> dict[str, Any]:
        language_code = result.get("language")
        if not language_code:
            raise ValueError(
                "Alignment backend requires 'language' in transcription result."
            )

        loop = asyncio.get_running_loop()
        device = determine_inference_device()

        alignment_model_start = time.perf_counter()
        logger.info(
            f"Request ID: {request_id} - Loading alignment model backend=whisperx language={language_code}"
        )
        model_a, metadata = await load_align_model(language_code=language_code)
        logger.info(
            f"Request ID: {request_id} - Alignment model loaded in {time.perf_counter() - alignment_model_start:.2f} seconds"
        )

        def _run_alignment() -> Any:
            with torch.inference_mode():
                return whisperx_alignment.align(
                    transcript=result["segments"],
                    model=model_a,
                    align_model_metadata=metadata,
                    audio=audio,
                    device=device,
                    return_char_alignments=False,
                )

        alignment_start = time.perf_counter()
        result["segments"] = await loop.run_in_executor(None, _run_alignment)
        logger.info(
            f"Request ID: {request_id} - Alignment completed using backend=whisperx in {time.perf_counter() - alignment_start:.2f} seconds"
        )
        return result


class WhisperXDiarizationBackend:
    async def preload_default(self) -> None:
        if not config.diarization.preload_model:
            return
        await load_diarize_pipeline(model_name=config.diarization.model)

    def list_loaded_models(self) -> list[str]:
        return sorted(str(key) for key in diarize_pipeline_instances.keys())

    async def load_model(self, model_name: str) -> None:
        await load_diarize_pipeline(model_name=model_name)

    def unload_model(self, model_name: str) -> bool:
        model = diarize_pipeline_instances.pop(model_name, None)
        if model is None:
            return False
        unload_model_object(model)
        return True

    async def diarize(
        self,
        *,
        result: dict[str, Any],
        audio: np.ndarray,
        speaker_embeddings: bool,
        request_id: str,
    ) -> dict[str, Any]:
        if "segments" not in result:
            raise ValueError(
                "Diarization backend requires 'segments' in transcription result."
            )

        loop = asyncio.get_running_loop()
        diarization_model_start = time.perf_counter()
        logger.info(
            f"Request ID: {request_id} - Loading diarization model backend=whisperx model={config.diarization.model}"
        )
        diarize_model = await load_diarize_pipeline(model_name=config.diarization.model)
        logger.info(
            f"Request ID: {request_id} - Diarization model loaded in {time.perf_counter() - diarization_model_start:.2f} seconds"
        )

        def _run_diarization() -> tuple[Any, Any]:
            with torch.inference_mode():
                if speaker_embeddings:
                    return diarize_model(audio=audio, return_embeddings=True)
                return diarize_model(audio=audio), None

        diarize_start = time.perf_counter()
        diarize_segments, embeddings = await loop.run_in_executor(None, _run_diarization)
        result["segments"] = whisperx_diarize.assign_word_speakers(
            diarize_segments,
            result["segments"],
            embeddings,
        )
        logger.info(
            f"Request ID: {request_id} - Diarization completed using backend=whisperx in {time.perf_counter() - diarize_start:.2f} seconds"
        )
        return result


def register_whisperx_backends() -> None:
    global _registered
    if _registered:
        return
    register_transcription_backend("whisperx", WhisperXTranscriptionBackend())
    register_alignment_backend("whisperx", WhisperXAlignmentBackend())
    register_diarization_backend("whisperx", WhisperXDiarizationBackend())
    _registered = True
