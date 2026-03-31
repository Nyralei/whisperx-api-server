import contextlib
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import torch
import gc
import logging
import time
import tempfile
import asyncio
from fastapi import UploadFile

from whisperx import audio as whisperx_audio

from whisperx_api_server.config import (
    Language,
)
from whisperx_api_server.dependencies import get_config
import whisperx_api_server.s3_client as s3_client
import whisperx_api_server.kafka_client as kafka_client
from whisperx_api_server.backends.registry import (
    get_default_transcription_model_name,
    get_alignment_backend,
    get_diarization_backend,
    get_transcription_backend,
    resolve_stage_backends,
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
        _concurrency_semaphore = asyncio.Semaphore(
            config.max_concurrent_transcriptions)
    return _concurrency_semaphore


def _cleanup_cache_only():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_io_executor() -> ThreadPoolExecutor:
    global _io_executor
    if _io_executor is None:
        _io_executor = ThreadPoolExecutor(
            max_workers=config.io_executor_workers, thread_name_prefix="whisperx_io")
    return _io_executor


async def _save_upload_to_temp(audio_file: UploadFile, request_id: str) -> str:
    """Stream upload to temp file in chunks"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{audio_file.filename}") as tmp:
        file_path = tmp.name

    chunk_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=2)
    write_error: list[BaseException] = []

    def _writer() -> None:
        try:
            with open(file_path, "wb", buffering=_UPLOAD_WRITE_BUFFER_SIZE) as f:
                while True:
                    chunk = chunk_queue.get()
                    if chunk is None:
                        break
                    f.write(chunk)
        except BaseException as e:
            write_error.append(e)

    writer_thread = threading.Thread(target=_writer, daemon=True)
    writer_thread.start()

    try:
        while True:
            chunk = await audio_file.read(_UPLOAD_STREAM_CHUNK_SIZE)
            if len(chunk) == 0:
                break
            try:
                chunk_queue.put_nowait(chunk)
            except queue.Full:
                await asyncio.to_thread(chunk_queue.put, chunk)
            if len(chunk) < _UPLOAD_STREAM_CHUNK_SIZE:
                break
        try:
            chunk_queue.put_nowait(None)
        except queue.Full:
            await asyncio.to_thread(chunk_queue.put, None)
    except Exception as e:
        logger.error(
            f"Request ID: {request_id} - Failed to read uploaded file: {e}")
        with contextlib.suppress(queue.Full):
            chunk_queue.put_nowait(None)
        await asyncio.to_thread(writer_thread.join, 5)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
        raise
    finally:
        await asyncio.to_thread(writer_thread.join, 30)

    if write_error:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
        logger.error(
            f"Request ID: {request_id} - Failed to write temp file: {write_error[0]}")
        raise write_error[0]

    return file_path


async def _load_audio(file_path: str, request_id: str) -> np.ndarray:
    loop = asyncio.get_running_loop()
    executor = _get_io_executor()
    try:
        audio = await loop.run_in_executor(executor, whisperx_audio.load_audio, file_path)
        logger.info(
            f"Request ID: {request_id} - Audio loaded from {file_path}")
        return audio
    except Exception as e:
        logger.error(f"Request ID: {request_id} - Failed to load audio: {e}")
        raise


def _finalize_text(result: dict[str, Any], align_or_diarize: bool) -> dict[str, Any]:
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
    asr_options: dict | None = None,
    language: Language = config.default_language,
    model_name: str | None = None,
    align: bool = False,
    diarize: bool = False,
    speaker_embeddings: bool = False,
    request_id: str = "",
    task: str = "transcribe",
) -> dict[str, Any]:
    start_time = time.perf_counter()
    file_path = None
    audio = None
    concurrency_sem = _get_concurrency_semaphore()
    semaphore_acquired = False
    profile: dict[str, float] = {}

    try:
        t0 = time.perf_counter()
        file_path = await _save_upload_to_temp(audio_file, request_id)
        profile["upload_save"] = time.perf_counter() - t0
        logger.info(
            f"Request ID: {request_id} - Saving uploaded file took {profile['upload_save']:.2f} seconds")

        if concurrency_sem:
            await concurrency_sem.acquire()
            semaphore_acquired = True
            logger.debug(
                f"Request ID: {request_id} - Acquired GPU concurrency semaphore")

        selected_backends = resolve_stage_backends()
        transcription_stage_backend = get_transcription_backend(
            selected_backends.transcription)
        alignment_stage_backend = (
            get_alignment_backend(selected_backends.alignment)
            if (align or diarize)
            else None
        )
        diarization_stage_backend = (
            get_diarization_backend(selected_backends.diarization)
            if diarize
            else None
        )

        if not model_name:
            model_name = get_default_transcription_model_name()

        logger.info(
            f"Request ID: {request_id} - Transcribing {audio_file.filename} with model: {model_name}, options: {asr_options}, language: {language}, task: {task}, stage_backends: {selected_backends}")

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
        result = await transcription_stage_backend.transcribe(
            model_name=model_name,
            audio=audio,
            batch_size=batch_size,
            chunk_size=chunk_size,
            language=language,
            task=task,
            asr_options=asr_options,
            request_id=request_id,
        )
        profile["transcribe"] = time.perf_counter() - t0
        logger.info(
            f"Request ID: {request_id} - Transcription took {profile['transcribe']:.2f} seconds")

        if align or diarize:
            if alignment_stage_backend is None:
                raise RuntimeError("Alignment backend is not initialized.")
            t0 = time.perf_counter()
            result = await alignment_stage_backend.align(
                result=result,
                audio=audio,
                request_id=request_id,
            )
            profile["align"] = time.perf_counter() - t0
            logger.debug(
                f"Request ID: {request_id} - Alignment took {profile['align']:.2f} seconds")

        if diarize:
            if diarization_stage_backend is None:
                raise RuntimeError("Diarization backend is not initialized.")
            t0 = time.perf_counter()
            result = await diarization_stage_backend.diarize(
                result=result,
                audio=audio,
                speaker_embeddings=speaker_embeddings,
                request_id=request_id,
            )
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
            if concurrency_sem and semaphore_acquired:
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


class QueueFullError(Exception):
    pass


async def transcribe_via_kafka(
    audio_file: UploadFile,
    *,
    params: dict[str, Any],
    request_id: str,
) -> dict[str, Any]:
    max_pending = config.kafka.max_pending_jobs
    if max_pending > 0 and len(kafka_client._pending_jobs) >= max_pending:
        raise QueueFullError(
            f"Too many pending jobs ({len(kafka_client._pending_jobs)}/{max_pending})"
        )

    logger.info(f"Request ID: {request_id} - Uploading audio to S3")
    audio_bytes = await audio_file.read()
    s3_key = await s3_client.upload_audio(audio_bytes, request_id, audio_file.filename or "audio")
    del audio_bytes

    logger.info(
        f"Request ID: {request_id} - Submitting job to Kafka (key: {s3_key})")
    future = await kafka_client.submit_job(request_id, s3_key, audio_file.filename or "audio", params)
    try:
        result = await asyncio.wait_for(future, timeout=config.kafka.reply_timeout_seconds)
        logger.info(f"Request ID: {request_id} - Received result from worker")
        return result
    except asyncio.TimeoutError:
        kafka_client._pending_jobs.pop(request_id, None)
        logger.error(
            f"Request ID: {request_id} - Timed out waiting for worker reply")
        raise TimeoutError(
            f"Job {request_id} timed out after {config.kafka.reply_timeout_seconds}s"
        )
