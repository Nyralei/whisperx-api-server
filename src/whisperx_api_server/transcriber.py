import contextlib
import os
from dataclasses import replace

import numpy as np
import torch
import gc
import logging
import time
import tempfile
import asyncio
from fastapi import UploadFile

from whisperx import asr as whisperx_asr
from whisperx import audio as whisperx_audio
from whisperx import alignment as whisperx_alignment
from whisperx import diarize as whisperx_diarize
from whisperx import schema as whisperx_schema
from faster_whisper.transcribe import TranscriptionOptions

from whisperx_api_server.config import (
    Language,
)
from whisperx_api_server.dependencies import get_config
from whisperx_api_server.models import (
    load_align_model,
    load_diarize_pipeline,
    load_transcribe_pipeline,
    determine_inference_device
)

logger = logging.getLogger(__name__)

config = get_config()

_concurrency_semaphore = None


def _get_concurrency_semaphore() -> asyncio.Semaphore | None:
    global _concurrency_semaphore
    if not torch.cuda.is_available():
        return None
    if _concurrency_semaphore is None:
        max_concurrent = int(os.getenv("MAX_CONCURRENT_TRANSCRIPTIONS", "1"))
        _concurrency_semaphore = asyncio.Semaphore(max_concurrent)
    return _concurrency_semaphore


def _cleanup_cache_only():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


async def _save_upload_to_temp(audio_file: UploadFile, request_id: str) -> str:
    loop = asyncio.get_running_loop()
    try:
        file_bytes = await audio_file.read()
    except Exception as e:
        logger.error(
            f"Request ID: {request_id} - Failed to read uploaded file: {e}")
        raise

    def _write_temp_file(data: bytes) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{audio_file.filename}") as temp_file:
            temp_file.write(data)
            return temp_file.name

    try:
        file_path = await loop.run_in_executor(None, _write_temp_file, file_bytes)
    except Exception as e:
        logger.error(
            f"Request ID: {request_id} - Failed to write temp file: {e}")
        raise

    return file_path


async def _load_audio(file_path: str, request_id: str):
    loop = asyncio.get_running_loop()
    try:
        audio = await loop.run_in_executor(None, whisperx_audio.load_audio, file_path)
        logger.info(
            f"Request ID: {request_id} - Audio loaded from {file_path}")
        return audio
    except Exception as e:
        logger.error(f"Request ID: {request_id} - Failed to load audio: {e}")
        raise


def _apply_request_options(model: whisperx_asr.FasterWhisperPipeline, asr_options: dict | None) -> None:
    """Apply per-request asr_options to the pipeline so each request can set language, hotwords, etc."""
    opts = asr_options or {}
    overrides = {}
    if opts.get("temperatures") is not None:
        t = opts["temperatures"]
        overrides["temperatures"] = [t] if not isinstance(
            t, (list, tuple)) else list(t)
    if opts.get("word_timestamps") is not None:
        overrides["word_timestamps"] = opts["word_timestamps"]
    if opts.get("initial_prompt") is not None:
        overrides["initial_prompt"] = opts["initial_prompt"]
    if opts.get("hotwords") is not None:
        overrides["hotwords"] = opts["hotwords"]
    if overrides and hasattr(model.options, "__dataclass_fields__"):
        model.options = replace(model.options, **overrides)
    if "suppress_numerals" in opts:
        model.suppress_numerals = opts["suppress_numerals"]


async def _transcribe_audio(
    model: whisperx_asr.FasterWhisperPipeline,
    audio: str,
    batch_size: int,
    chunk_size: int,
    language: Language,
    task: str,
    asr_options: dict | None,
    request_id: str,
):
    loop = asyncio.get_running_loop()
    lang = getattr(language, "value", language) if language else None
    num_workers = 0 if determine_inference_device(
    ) == "cuda" else config.whisper.num_workers
    lock = getattr(model, "_transcribe_lock", None)

    def _run_transcription():
        with torch.inference_mode():
            return model.transcribe(
                audio=audio,
                batch_size=batch_size,
                chunk_size=chunk_size,
                num_workers=num_workers,
                language=lang,
                task=task,
            )

    async def _do_transcribe():
        if lock is not None:
            async with lock:
                _apply_request_options(model, asr_options)
                return await loop.run_in_executor(None, _run_transcription)
        _apply_request_options(model, asr_options)
        return await loop.run_in_executor(None, _run_transcription)

    result = await _do_transcribe()
    logger.info(f"Request ID: {request_id} - Transcription completed")
    return result


async def _align_audio(result: whisperx_schema.TranscriptionResult, audio: np.ndarray, request_id: str) -> whisperx_schema.TranscriptionResult:
    loop = asyncio.get_running_loop()
    device = determine_inference_device()
    try:
        alignment_model_start = time.time()
        logger.info(f"Request ID: {request_id} - Loading alignment model")
        model_a, metadata = await load_align_model(language_code=result["language"])
        logger.info(f"Request ID: {request_id} - Alignment model loaded")
        logger.info(
            f"Request ID: {request_id} - Loading alignment model took {time.time() - alignment_model_start:.2f} seconds")

        def _run_alignment():
            with torch.inference_mode():
                return whisperx_alignment.align(
                    transcript=result["segments"],
                    model=model_a,
                    align_model_metadata=metadata,
                    audio=audio,
                    device=device,
                    return_char_alignments=False
                )
        alignment_start = time.time()
        result["segments"] = await loop.run_in_executor(None, _run_alignment)
        logger.info(
            f"Request ID: {request_id} - Alignment took {time.time() - alignment_start:.2f} seconds")
        return result
    except Exception as e:
        logger.error(f"Request ID: {request_id} - Alignment failed: {e}")
        raise


async def _diarize_audio(result: whisperx_schema.TranscriptionResult, audio: np.ndarray, speaker_embeddings: bool, request_id: str) -> whisperx_schema.TranscriptionResult:
    loop = asyncio.get_running_loop()
    try:
        diarization_model_start = time.time()
        logger.info(f"Request ID: {request_id} - Loading diarization model")
        diarize_model = await load_diarize_pipeline(model_name=config.diarization.model)
        logger.info(
            f"Request ID: {request_id} - Diarization model loaded. Loading took {time.time() - diarization_model_start:.2f} seconds. Starting diarization")

        def _run_diarization():
            with torch.inference_mode():
                if speaker_embeddings:
                    return diarize_model(audio=audio, return_embeddings=True)
                else:
                    return diarize_model(audio=audio), None
        diarize_start = time.time()
        diarize_segments, embeddings = await loop.run_in_executor(None, _run_diarization)
        result["segments"] = whisperx_diarize.assign_word_speakers(
            diarize_segments, result["segments"])

        if embeddings is not None:
            result["embeddings"] = embeddings

        logger.info(
            f"Request ID: {request_id} - Diarization took {time.time() - diarize_start:.2f} seconds")
        return result
    except Exception as e:
        logger.error(f"Request ID: {request_id} - Diarization failed: {e}")
        raise


def _finalize_text(result, align_or_diarize: bool):
    segments = result.get("segments", [])
    if align_or_diarize and isinstance(segments, dict):
        segments = segments.get("segments", [])

    result["text"] = '\n'.join([s.get("text", "").strip()
                               for s in segments if s.get("text")])
    return result


async def transcribe(
    audio_file: UploadFile,
    batch_size: int = config.batch_size,
    chunk_size: int = config.chunk_size,
    asr_options: dict = {},
    language: Language = config.default_language,
    model_name: str = config.whisper.model,
    align: bool = False,
    diarize: bool = False,
    speaker_embeddings: bool = False,
    request_id: str = "",
    task: str = "transcribe"
) -> whisperx_schema.TranscriptionResult:
    start_time = time.time()
    file_path = None
    audio = None
    concurrency_sem = _get_concurrency_semaphore()

    try:
        file_path = await _save_upload_to_temp(audio_file, request_id)
        logger.info(
            f"Request ID: {request_id} - Saving uploaded file took {time.time() - start_time:.2f} seconds")

        if concurrency_sem:
            await concurrency_sem.acquire()
            logger.debug(
                f"Request ID: {request_id} - Acquired GPU concurrency semaphore")

        logger.info(
            f"Request ID: {request_id} - Transcribing {audio_file.filename} with model: {model_name} and options: {asr_options}, language: {language}, task: {task}")

        model_loading_start = time.time()

        model = await load_transcribe_pipeline(model_name=model_name)

        logger.info(
            f"Request ID: {request_id} - Loading model pipeline took {time.time() - model_loading_start:.2f} seconds")

        audio_loading_start = time.time()

        audio = await _load_audio(file_path, request_id)

        logger.info(
            f"Request ID: {request_id} - Loading audio took {time.time() - audio_loading_start:.2f} seconds")

        transcription_start = time.time()

        result = await _transcribe_audio(
            model, audio, batch_size, chunk_size, language, task, asr_options, request_id
        )

        logger.info(
            f"Request ID: {request_id} - Transcription took {time.time() - transcription_start:.2f} seconds")

        if align or diarize:
            result = await _align_audio(result, audio, request_id)

        if diarize:
            result = await _diarize_audio(result, audio, speaker_embeddings, request_id)

        result = _finalize_text(result, align or diarize)

        logger.info(
            f"Request ID: {request_id} - Transcription completed for {audio_file.filename}")

        return result
    except Exception as e:
        logger.error(
            f"Request ID: {request_id} - Transcription failed for {audio_file.filename} with error: {e}")
        raise
    finally:
        with contextlib.suppress(Exception):
            if concurrency_sem:
                concurrency_sem.release()
        with contextlib.suppress(Exception):
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
        if config.audio_cleanup and audio is not None:
            del audio
            logger.info(f"Request ID: {request_id} - Audio data cleaned up")
        if config.cache_cleanup:
            _cleanup_cache_only()
            logger.info(f"Request ID: {request_id} - Cache cleanup completed")
