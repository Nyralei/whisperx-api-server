import contextlib
import os
import queue
import re
import threading
from typing import Any

import numpy as np
import gc
import logging
import time
import tempfile
import asyncio
import torch
from fastapi import UploadFile

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
from whisperx_api_server.observability import pipeline as _pipe
from whisperx_api_server.observability import kafka as _kafka
from whisperx_api_server import request_status

logger = logging.getLogger(__name__)

config = get_config()

_concurrency_semaphore: asyncio.Semaphore | None = None
_decode_semaphore: asyncio.Semaphore | None = None
_UPLOAD_STREAM_CHUNK_SIZE = 1024 * 1024  # 1 MiB
_UPLOAD_WRITE_BUFFER_SIZE = 1024 * 1024  # 1 MiB
_UPLOAD_WRITER_JOIN_TIMEOUT_SECS = 10  # writer thread should drain in <1s on healthy disk
_FILENAME_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]")
_FILENAME_MAX_SUFFIX_LEN = 64


class InvalidAudioError(ValueError):
    """The upload could not be decoded as audio — treat as HTTP 422 (client error)."""


class UploadTooLargeError(ValueError):
    """The upload exceeded max_upload_size_bytes — treat as HTTP 413 (client error)."""


def _safe_filename(filename: str | None, default: str = "audio") -> str:
    """Return a sanitized filename. Strips path components and non-word chars
    so a malicious or unusual filename cannot break tempfile paths or S3 keys.
    Returns `default` if nothing usable is left."""
    if not filename:
        return default
    base = os.path.basename(filename)
    safe = _FILENAME_SAFE_CHARS.sub("_", base)[:_FILENAME_MAX_SUFFIX_LEN]
    return safe or default


def _safe_filename_suffix(filename: str | None) -> str:
    """Return a sanitized suffix for tempfile naming."""
    safe = _safe_filename(filename, default="")
    return f"_{safe}" if safe else ""


def init_concurrency() -> None:
    """Eagerly create the inference and decode-admission semaphores. Called once from lifespan."""
    global _concurrency_semaphore, _decode_semaphore
    n = config.max_concurrent_transcriptions
    _concurrency_semaphore = asyncio.Semaphore(n) if n > 0 else None
    # Decode admission: allows one more concurrent decode than inference slots so a new
    # request can decode while the previous one is still on the model executor.
    d = n + 1 if n > 0 else 0
    _decode_semaphore = asyncio.Semaphore(d) if d > 0 else None


def _get_concurrency_semaphore() -> asyncio.Semaphore | None:
    return _concurrency_semaphore


def _get_decode_semaphore() -> asyncio.Semaphore | None:
    return _decode_semaphore


def _cleanup_cache_only():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


_SAMPLE_RATE = 16000  # matches whisperx.audio.SAMPLE_RATE


def _ffmpeg_decode_cmd(input_arg: str, sample_rate: int) -> list[str]:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "0"]
    if input_arg != "pipe:0":
        cmd.append("-nostdin")
    cmd += ["-i", input_arg, "-f", "f32le", "-ac", "1", "-ar", str(sample_rate), "pipe:1"]
    return cmd


async def _run_ffmpeg_decode(
    input_arg: str,
    feed_stdin,
    request_id: str,
    sample_rate: int,
) -> np.ndarray:
    proc = await asyncio.create_subprocess_exec(
        *_ffmpeg_decode_cmd(input_arg, sample_rate),
        stdin=asyncio.subprocess.PIPE if feed_stdin else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    async def _pump_stdin():
        if feed_stdin is None:
            return
        try:
            await feed_stdin(proc.stdin)
        except (BrokenPipeError, ConnectionResetError):
            pass  # ffmpeg died early; stderr reader will surface the real reason
        finally:
            try:
                proc.stdin.close()
            except (BrokenPipeError, ConnectionResetError):
                pass

    try:
        _, pcm, err = await asyncio.gather(
            _pump_stdin(),
            proc.stdout.read(),
            proc.stderr.read(),
        )
    except BaseException:
        proc.kill()
        await proc.wait()
        raise

    rc = await proc.wait()
    if rc != 0:
        msg = err.decode(errors="replace").strip()
        logger.warning(f"Request ID: {request_id} - ffmpeg exited {rc}: {msg}")
        raise InvalidAudioError(f"Could not decode audio (ffmpeg exit {rc}): {msg}")

    audio = np.frombuffer(pcm, dtype=np.float32).copy()
    if audio.size == 0:
        raise InvalidAudioError(
            "Decoded audio is empty (no audio stream or zero-length input)."
        )
    logger.info(
        f"Request ID: {request_id} - Audio decoded "
        f"({audio.size} samples, {audio.size / sample_rate:.2f}s)"
    )
    return audio


async def load_audio_from_path(
    file_path: str,
    request_id: str,
    sample_rate: int = _SAMPLE_RATE,
) -> np.ndarray:
    return await _run_ffmpeg_decode(file_path, None, request_id, sample_rate)


async def load_audio_from_bytes(
    audio_bytes: bytes,
    request_id: str,
    sample_rate: int = _SAMPLE_RATE,
) -> np.ndarray:
    async def feed(stdin):
        stdin.write(audio_bytes)
        await stdin.drain()
    return await _run_ffmpeg_decode("pipe:0", feed, request_id, sample_rate)


async def _save_upload_to_temp(audio_file: UploadFile, request_id: str) -> str:
    """Stream upload to temp file in chunks. Enforces max_upload_size_bytes if set."""
    max_bytes = config.max_upload_size_bytes
    suffix = _safe_filename_suffix(audio_file.filename)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
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

    total_bytes = 0
    try:
        while True:
            chunk = await audio_file.read(_UPLOAD_STREAM_CHUNK_SIZE)
            if len(chunk) == 0:
                break
            total_bytes += len(chunk)
            if max_bytes > 0 and total_bytes > max_bytes:
                raise UploadTooLargeError(
                    f"Upload exceeds max_upload_size_bytes ({max_bytes} bytes)."
                )
            try:
                chunk_queue.put_nowait(chunk)
            except queue.Full:
                await asyncio.to_thread(chunk_queue.put, chunk)
        try:
            chunk_queue.put_nowait(None)
        except queue.Full:
            await asyncio.to_thread(chunk_queue.put, None)
    except Exception as e:
        if not isinstance(e, UploadTooLargeError):
            logger.error(
                f"Request ID: {request_id} - Failed to read uploaded file: {e}")
        with contextlib.suppress(queue.Full):
            chunk_queue.put_nowait(None)
        await asyncio.to_thread(writer_thread.join, _UPLOAD_WRITER_JOIN_TIMEOUT_SECS)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
        raise
    finally:
        await asyncio.to_thread(writer_thread.join, _UPLOAD_WRITER_JOIN_TIMEOUT_SECS)

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
    decode_sem = _get_decode_semaphore()
    profile: dict[str, float] = {}
    audio_duration_seconds = 0.0

    try:
        request_status.set_stage(request_id, "upload_save")
        t0 = time.perf_counter()
        file_path = await _save_upload_to_temp(audio_file, request_id)
        profile["upload_save"] = time.perf_counter() - t0
        logger.info(
            f"Request ID: {request_id} - Saving uploaded file took {profile['upload_save']:.2f} seconds")
        _pipe.stage_duration.labels(
            stage="upload_save").observe(profile["upload_save"])

        # Decode outside the inference semaphore, bounded by the decode-admission semaphore.
        async with contextlib.AsyncExitStack() as _decode_stack:
            if decode_sem is not None:
                await _decode_stack.enter_async_context(decode_sem)
            request_status.set_stage(request_id, "audio_load")
            t0 = time.perf_counter()
            audio = await load_audio_from_path(file_path, request_id)
            profile["audio_load"] = time.perf_counter() - t0
            logger.info(
                f"Request ID: {request_id} - Loading audio took {profile['audio_load']:.2f} seconds")
            _pipe.stage_duration.labels(
                stage="audio_load").observe(profile["audio_load"])
            audio_duration_seconds = len(audio) / 16000.0

            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass
                file_path = None
        # decode_sem released; temp file deleted; float32 array in memory

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

        _sem_t0 = time.perf_counter()
        if concurrency_sem:
            request_status.set_stage(request_id, "awaiting_concurrency")
            await concurrency_sem.acquire()
        _sem_elapsed = time.perf_counter() - _sem_t0
        logger.debug(
            f"Request ID: {request_id} - Acquired inference concurrency semaphore")
        _pipe.semaphore_wait.observe(_sem_elapsed)

        try:
            request_status.set_stage(request_id, "transcribe")
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
            _pipe.stage_duration.labels(
                stage="transcribe").observe(profile["transcribe"])
            if audio_duration_seconds > 0:
                _pipe.realtime_factor.labels(model=model_name, stage="transcribe").observe(
                    profile["transcribe"] / audio_duration_seconds
                )

            if align or diarize:
                if alignment_stage_backend is None:
                    raise RuntimeError("Alignment backend is not initialized.")
                request_status.set_stage(request_id, "align")
                t0 = time.perf_counter()
                result = await alignment_stage_backend.align(
                    result=result,
                    audio=audio,
                    request_id=request_id,
                )
                profile["align"] = time.perf_counter() - t0
                logger.debug(
                    f"Request ID: {request_id} - Alignment took {profile['align']:.2f} seconds")
                _pipe.stage_duration.labels(
                    stage="align").observe(profile["align"])
                if audio_duration_seconds > 0:
                    _pipe.realtime_factor.labels(model=model_name, stage="align").observe(
                        profile["align"] / audio_duration_seconds
                    )

            if diarize:
                if diarization_stage_backend is None:
                    raise RuntimeError("Diarization backend is not initialized.")
                request_status.set_stage(request_id, "diarize")
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
                _pipe.stage_duration.labels(
                    stage="diarize").observe(profile["diarize"])
                if audio_duration_seconds > 0:
                    _pipe.realtime_factor.labels(model=model_name, stage="diarize").observe(
                        profile["diarize"] / audio_duration_seconds
                    )

            request_status.set_stage(request_id, "finalize")
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

            request_status.mark_completed(request_id)
            return result
        finally:
            if concurrency_sem:
                concurrency_sem.release()
    except Exception as e:
        logger.error(
            f"Request ID: {request_id} - Transcription failed for {audio_file.filename} with error: {e}")
        request_status.mark_failed(request_id, str(e), type(e).__name__)
        raise
    finally:
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
        _kafka.queue_rejected_total.inc()
        err = QueueFullError(
            f"Too many pending jobs ({len(kafka_client._pending_jobs)}/{max_pending})"
        )
        request_status.mark_failed(request_id, str(err), "QueueFullError")
        raise err

    safe_name = _safe_filename(audio_file.filename)
    request_status.set_stage(request_id, "uploading_to_s3")
    logger.info(f"Request ID: {request_id} - Uploading audio to S3")
    try:
        s3_key = await s3_client.upload_audio_stream(audio_file, request_id, safe_name)
    except Exception as e:
        request_status.mark_failed(request_id, str(e), type(e).__name__)
        raise

    request_status.set_stage(request_id, "submitted_to_kafka")
    logger.info(
        f"Request ID: {request_id} - Submitting job to Kafka (key: {s3_key})")
    try:
        future = await kafka_client.submit_job(request_id, s3_key, safe_name, params)
    except Exception as e:
        request_status.mark_failed(request_id, str(e), type(e).__name__)
        raise

    request_status.set_stage(request_id, "awaiting_worker")
    try:
        result = await asyncio.wait_for(future, timeout=config.kafka.reply_timeout_seconds)
        logger.info(f"Request ID: {request_id} - Received result from worker")
        request_status.mark_completed(request_id)
        return result
    except asyncio.TimeoutError:
        kafka_client._pending_jobs.pop(request_id, None)
        _kafka.job_timeout_total.inc()
        logger.error(
            f"Request ID: {request_id} - Timed out waiting for worker reply")
        err = TimeoutError(
            f"Job {request_id} timed out after {config.kafka.reply_timeout_seconds}s"
        )
        request_status.mark_failed(request_id, str(err), "TimeoutError")
        raise err
    except BaseException as e:
        kafka_client._pending_jobs.pop(request_id, None)
        request_status.mark_failed(request_id, str(e), type(e).__name__)
        raise
