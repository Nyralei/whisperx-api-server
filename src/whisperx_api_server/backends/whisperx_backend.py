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
from whisperx_api_server.executors import get_model_executor, get_io_executor
from .whisperx_runtime import (
    align_model_instances,
    alignment_cache_mod_lock,
    alignment_locks,
    acquire_align_model,
    acquire_diarize_pipeline,
    acquire_transcribe_pipeline,
    align_cache_key_for,
    determine_inference_device,
    diarize_locks,
    diarize_pipeline_instances,
    load_align_model,
    load_diarize_pipeline,
    load_transcribe_pipeline,
    release_align_model,
    release_diarize_pipeline,
    release_transcribe_pipeline,
    transcribe_pipeline_instances,
    transcribe_pipeline_locks,
    unload_model_object_async,
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

    async def unload_model(self, model_name: str) -> bool:
        # Pop first so no new request can grab this instance from the cache.
        # New requests calling load_transcribe_pipeline() will then get a fresh
        # lock from the defaultdict, isolated from our teardown.
        #
        # Old in-flight requests already hold a reference to the popped lock
        # (via getattr(model, "_transcribe_lock", ...)), so we hold the lock
        # for the entire teardown — both pre-existing critical sections and
        # post-pop callers serialize behind us, and once we release the lock
        # they would still see a torn-down model. Acceptable: the request
        # would either succeed on CPU or surface a backend error mapped to 5xx.
        to_remove = [k for k in transcribe_pipeline_instances if k[0] == model_name]
        if not to_remove:
            return False
        for cache_key in to_remove:
            pipeline = transcribe_pipeline_instances.pop(cache_key, None)
            lock = transcribe_pipeline_locks.pop(cache_key, None)
            if pipeline is None:
                continue
            if lock is not None:
                async with lock:
                    await unload_model_object_async(pipeline)
            else:
                await unload_model_object_async(pipeline)
        return True

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
        cache_key = getattr(model, "_cache_key", None)

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

        # Refcount so concurrent /models/unload can see this pipeline is in use.
        if cache_key is not None:
            acquire_transcribe_pipeline(cache_key)
        try:
            if lock is not None:
                async with lock:
                    _apply_request_options(model, asr_options)
                    result = await loop.run_in_executor(get_model_executor(), _run_transcription)
            else:
                _apply_request_options(model, asr_options)
                result = await loop.run_in_executor(get_model_executor(), _run_transcription)
        finally:
            if cache_key is not None:
                release_transcribe_pipeline(cache_key)

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

    async def unload_model(self, model_name: str) -> bool:
        # Pop atomically against the cache-eviction lock so concurrent
        # whitelist cleanup can't race the same key. Hold the per-key lock
        # for the entire teardown — both in-flight `align()` callers and the
        # whitelist cleanup serialize behind it.
        async with alignment_cache_mod_lock:
            align_model_data = align_model_instances.pop(model_name, None)
            lock = alignment_locks.pop(model_name, None)
        if align_model_data is None:
            return False
        if lock is not None:
            async with lock:
                await unload_model_object_async(align_model_data.get("model"))
        else:
            await unload_model_object_async(align_model_data.get("model"))
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
        cache_key = align_cache_key_for(language_code)

        # Increment refcount before load so eviction skips this key while we hold it.
        acquire_align_model(language_code)
        try:
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
            async with alignment_locks[cache_key]:
                result["segments"] = await loop.run_in_executor(get_model_executor(), _run_alignment)
        finally:
            release_align_model(language_code)

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

    async def unload_model(self, model_name: str) -> bool:
        # Pop first so new requests get a fresh cache miss path. Hold the
        # popped lock during teardown so any in-flight diarize that already
        # has a reference to the old pipeline finishes before we tear it down.
        model = diarize_pipeline_instances.pop(model_name, None)
        lock = diarize_locks.pop(model_name, None)
        if model is None:
            return False
        if lock is not None:
            async with lock:
                await unload_model_object_async(model)
        else:
            await unload_model_object_async(model)
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

        # Refcount so concurrent /diarize_models/unload sees this is in use.
        acquire_diarize_pipeline(config.diarization.model)
        diarize_start = time.perf_counter()
        try:
            async with diarize_locks[config.diarization.model]:
                diarize_segments, embeddings = await loop.run_in_executor(get_model_executor(), _run_diarization)
        finally:
            release_diarize_pipeline(config.diarization.model)
        # Lock released before CPU-only word assignment so the diarize slot is free sooner.
        result["segments"] = await loop.run_in_executor(
            get_io_executor(),
            lambda: whisperx_diarize.assign_word_speakers(
                diarize_segments,
                result["segments"],
                embeddings,
            ),
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
