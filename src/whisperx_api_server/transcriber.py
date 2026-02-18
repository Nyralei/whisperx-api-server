import contextlib
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
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
_io_executor: ThreadPoolExecutor | None = None
_UPLOAD_STREAM_CHUNK_SIZE = 1024 * 1024  # 1 MiB
_UPLOAD_WRITE_BUFFER_SIZE = 1024 * 1024  # 1 MiB


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


def _get_io_executor() -> ThreadPoolExecutor:
    global _io_executor
    if _io_executor is None:
        workers = int(os.getenv("IO_EXECUTOR_WORKERS", "4"))
        _io_executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="whisperx_io")
    return _io_executor


async def _save_upload_to_temp(audio_file: UploadFile, request_id: str) -> str:
    """Stream upload to temp file in chunks"""
    upload_start = time.perf_counter()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{audio_file.filename}") as tmp:
        file_path = tmp.name

    chunk_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=2)
    write_error: list[BaseException] = []
    chunks_written: list[int] = []

    def _writer() -> None:
        try:
            with open(file_path, "wb", buffering=_UPLOAD_WRITE_BUFFER_SIZE) as f:
                chunk_idx = 0
                while True:
                    chunk = chunk_queue.get()
                    if chunk is None:
                        break
                    f.write(chunk)
                    chunks_written.append(len(chunk))
                    logger.debug(
                        f"Request ID: {request_id} - writer saved chunk {chunk_idx} "
                        f"size={len(chunk)} bytes, total_so_far={sum(chunks_written)}")
                    chunk_idx += 1
                logger.debug(
                    f"Request ID: {request_id} - writer finished: {chunk_idx} chunks, "
                    f"total={sum(chunks_written)} bytes")
        except BaseException as e:
            write_error.append(e)

    writer_thread = threading.Thread(target=_writer, daemon=True)
    writer_thread.start()
    logger.debug(
        f"Request ID: {request_id} - upload stream started, chunk_size={_UPLOAD_STREAM_CHUNK_SIZE}, "
        f"target={file_path}")

    try:
        chunk_idx = 0
        total_read = 0
        while True:
            read_start = time.perf_counter()
            chunk = await audio_file.read(_UPLOAD_STREAM_CHUNK_SIZE)
            read_elapsed = time.perf_counter() - read_start
            if len(chunk) == 0:
                break
            total_read += len(chunk)
            logger.debug(
                f"Request ID: {request_id} - read chunk {chunk_idx} size={len(chunk)} bytes "
                f"(read_ms={read_elapsed*1000:.1f}, total_read={total_read})")
            chunk_queue.put(chunk)
            chunk_idx += 1
            if len(chunk) < _UPLOAD_STREAM_CHUNK_SIZE:
                break
        chunk_queue.put(None)
        upload_elapsed = time.perf_counter() - upload_start
        logger.debug(
            f"Request ID: {request_id} - upload stream done: {chunk_idx} chunks, {total_read} bytes "
            f"in {upload_elapsed:.3f}s (~{total_read / (1024 * 1024) / upload_elapsed:.2f} MiB/s)")
    except Exception as e:
        logger.error(
            f"Request ID: {request_id} - Failed to read uploaded file: {e}")
        chunk_queue.put(None)
        writer_thread.join()
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
        raise
    finally:
        writer_thread.join(timeout=30)

    if write_error:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
        logger.error(
            f"Request ID: {request_id} - Failed to write temp file: {write_error[0]}")
        raise write_error[0]

    if logger.isEnabledFor(logging.DEBUG) and os.path.exists(file_path):
        logger.debug(
            f"Request ID: {request_id} - temp file ready size={os.path.getsize(file_path)} bytes")
    return file_path


async def _load_audio(file_path: str, request_id: str) -> np.ndarray:
    loop = asyncio.get_running_loop()
    executor = _get_io_executor()
    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
    logger.debug(
        f"Request ID: {request_id} - load_audio starting path={file_path} file_size={file_size}")
    load_start = time.perf_counter()
    try:
        audio = await loop.run_in_executor(executor, whisperx_audio.load_audio, file_path)
        load_elapsed = time.perf_counter() - load_start
        logger.info(
            f"Request ID: {request_id} - Audio loaded from {file_path}")
        logger.debug(
            f"Request ID: {request_id} - load_audio done shape={audio.shape} dtype={audio.dtype} "
            f"duration_s={load_elapsed:.3f} (~{file_size / (1024 * 1024) / load_elapsed:.2f} MiB/s decode)")
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
    audio: np.ndarray,
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


def _finalize_text(result: whisperx_schema.TranscriptionResult, align_or_diarize: bool) -> whisperx_schema.TranscriptionResult:
    segments = result.get("segments", [])
    if align_or_diarize and isinstance(segments, dict):
        segments = segments.get("segments", [])

    result["text"] = '\n'.join([s.get("text", "").strip()
                               for s in segments if s.get("text")])
    return result


async def transcribe(
    audio_file: UploadFile,
    batch_size: int = config.whisper.batch_size,
    chunk_size: int = config.whisper.chunk_size,
    asr_options: dict = {},
    language: Language = config.default_language,
    model_name: str = config.whisper.model,
    align: bool = False,
    diarize: bool = False,
    speaker_embeddings: bool = False,
    request_id: str = "",
    task: str = "transcribe"
) -> whisperx_schema.TranscriptionResult:
    start_time = time.perf_counter()
    file_path = None
    audio = None
    concurrency_sem = _get_concurrency_semaphore()
    profile: dict[str, float] = {}

    try:
        t0 = time.perf_counter()
        file_path = await _save_upload_to_temp(audio_file, request_id)
        profile["upload_save"] = time.perf_counter() - t0
        logger.info(
            f"Request ID: {request_id} - Saving uploaded file took {profile['upload_save']:.2f} seconds")

        if concurrency_sem:
            await concurrency_sem.acquire()
            logger.debug(
                f"Request ID: {request_id} - Acquired GPU concurrency semaphore")

        logger.info(
            f"Request ID: {request_id} - Transcribing {audio_file.filename} with model: {model_name} and options: {asr_options}, language: {language}, task: {task}")

        t0 = time.perf_counter()
        model = await load_transcribe_pipeline(model_name=model_name)
        profile["model_load"] = time.perf_counter() - t0
        logger.info(
            f"Request ID: {request_id} - Loading model pipeline took {profile['model_load']:.2f} seconds")

        t0 = time.perf_counter()
        audio = await _load_audio(file_path, request_id)
        profile["audio_load"] = time.perf_counter() - t0
        logger.info(
            f"Request ID: {request_id} - Loading audio took {profile['audio_load']:.2f} seconds")

        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
            file_path = None

        t0 = time.perf_counter()
        result = await _transcribe_audio(
            model, audio, batch_size, chunk_size, language, task, asr_options, request_id
        )
        profile["transcribe"] = time.perf_counter() - t0
        logger.info(
            f"Request ID: {request_id} - Transcription took {profile['transcribe']:.2f} seconds")

        if align or diarize:
            t0 = time.perf_counter()
            result = await _align_audio(result, audio, request_id)
            profile["align"] = time.perf_counter() - t0
            logger.debug(
                f"Request ID: {request_id} - Alignment took {profile['align']:.2f} seconds")

        if diarize:
            t0 = time.perf_counter()
            result = await _diarize_audio(result, audio, speaker_embeddings, request_id)
            profile["diarize"] = time.perf_counter() - t0
            logger.debug(
                f"Request ID: {request_id} - Diarization took {profile['diarize']:.2f} seconds")

        t0 = time.perf_counter()
        result = _finalize_text(result, align or diarize)
        profile["finalize"] = time.perf_counter() - t0

        total = time.perf_counter() - start_time
        logger.info(
            f"Request ID: {request_id} - Transcription completed for {audio_file.filename}")
        logger.debug(
            f"Request ID: {request_id} - profile: total={total:.2f}s | "
            + " | ".join(f"{k}={v:.2f}s" for k, v in profile.items())
            + f" | (other={total - sum(profile.values()):.2f}s)")

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
            if file_path is not None and os.path.exists(file_path):
                os.remove(file_path)
        if config.audio_cleanup and audio is not None:
            del audio
            logger.info(f"Request ID: {request_id} - Audio data cleaned up")
        if config.cache_cleanup:
            _cleanup_cache_only()
            logger.info(f"Request ID: {request_id} - Cache cleanup completed")
