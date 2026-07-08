from __future__ import annotations

import asyncio
import contextlib
import gc
import logging
import os
import queue
import re
import tempfile
import threading
import time
from typing import TYPE_CHECKING, Any

from fastapi import UploadFile

if TYPE_CHECKING:
    import numpy as np

import whisperx_api_server.kafka_client as kafka_client
import whisperx_api_server.s3_client as s3_client
from whisperx_api_server import request_status
from whisperx_api_server.backends.registry import (
    get_alignment_backend,
    get_default_transcription_model_name,
    get_diarization_backend,
    get_transcription_backend,
    resolve_stage_backends,
)
from whisperx_api_server.config import (
    Language,
)
from whisperx_api_server.dependencies import get_config
from whisperx_api_server.observability import kafka as _kafka
from whisperx_api_server.observability import pipeline as _pipe

logger = logging.getLogger(__name__)

_concurrency_semaphore: asyncio.Semaphore | None = None
_decode_semaphore: asyncio.Semaphore | None = None
_UPLOAD_STREAM_CHUNK_SIZE = 1024 * 1024  # 1 MiB
_UPLOAD_WRITE_BUFFER_SIZE = 1024 * 1024  # 1 MiB
# writer thread should drain in <1s on healthy disk
_UPLOAD_WRITER_JOIN_TIMEOUT_SECS = 10
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
    n = get_config().max_concurrent_transcriptions
    _concurrency_semaphore = asyncio.Semaphore(n) if n > 0 else None
    # Decode admission: allows one more concurrent decode than inference slots so a new
    # request can decode while the previous one is still on the model executor.
    d = n + 1 if n > 0 else 0
    _decode_semaphore = asyncio.Semaphore(d) if d > 0 else None


def log_concurrency_notes() -> None:
    """Warn that raising the inference limit doesn't parallelize one model."""
    n = get_config().max_concurrent_transcriptions
    if n > 1:
        logger.warning(
            "max_concurrent_transcriptions=%d, but each transcription pipeline "
            "holds a per-model lock during inference: requests for the SAME model "
            "still run one at a time. The limit parallelizes across DISTINCT "
            "models (and the align/diarize stages); raise same-model throughput "
            "with more GPUs or replicas/workers.",
            n,
        )


def _get_concurrency_semaphore() -> asyncio.Semaphore | None:
    return _concurrency_semaphore


def _get_decode_semaphore() -> asyncio.Semaphore | None:
    return _decode_semaphore


def _cleanup_cache_only():
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


_SAMPLE_RATE = 16000  # matches whisperx.audio.SAMPLE_RATE
_DECODE_READ_CHUNK_SIZE = 1024 * 1024  # 1 MiB


def _ffmpeg_decode_cmd(input_path: str, sample_rate: int) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        "0",
        "-nostdin",
        "-i",
        input_path,
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]


async def load_audio_from_path(
    file_path: str,
    request_id: str,
    sample_rate: int = _SAMPLE_RATE,
) -> np.ndarray:
    import numpy as np

    proc = await asyncio.create_subprocess_exec(
        *_ffmpeg_decode_cmd(file_path, sample_rate),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stdout is not None and proc.stderr is not None  # both opened as PIPE
    stdout, stderr = proc.stdout, proc.stderr

    async def _read_pcm() -> bytearray:
        buf = bytearray()
        while True:
            chunk = await stdout.read(_DECODE_READ_CHUNK_SIZE)
            if not chunk:
                return buf
            buf += chunk

    try:
        pcm, err = await asyncio.gather(_read_pcm(), stderr.read())
    except BaseException:
        proc.kill()
        await proc.wait()
        raise

    rc = await proc.wait()
    if rc != 0:
        msg = err.decode(errors="replace").strip()
        logger.warning("Request ID: %s - ffmpeg exited %s: %s", request_id, rc, msg)
        raise InvalidAudioError(f"Could not decode audio (ffmpeg exit {rc}): {msg}")

    if len(pcm) % 4:  # partial trailing f32 frame
        del pcm[len(pcm) - len(pcm) % 4 :]
    # writable zero-copy view; avoids holding PCM in memory twice
    audio = np.frombuffer(pcm, dtype=np.float32)
    if audio.size == 0:
        raise InvalidAudioError(
            "Decoded audio is empty (no audio stream or zero-length input)."
        )
    logger.info(
        "Request ID: %s - Audio decoded (%s samples, %.2fs)",
        request_id,
        audio.size,
        audio.size / sample_rate,
    )
    return audio


async def _save_upload_to_temp(audio_file: UploadFile, request_id: str) -> str:
    """Stream upload to temp file in chunks. Enforces max_upload_size_bytes if set."""
    max_bytes = get_config().max_upload_size_bytes
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
                "Request ID: %s - Failed to read uploaded file: %s", request_id, e
            )
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
            "Request ID: %s - Failed to write temp file: %s", request_id, write_error[0]
        )
        raise write_error[0]

    return file_path


def _finalize_text(result: dict[str, Any], align_or_diarize: bool) -> dict[str, Any]:
    segments = result.get("segments", [])
    if align_or_diarize and isinstance(segments, dict):
        segments = segments.get("segments", [])

    result["text"] = "\n".join(
        [s.get("text", "").strip() for s in segments if s.get("text")]
    )
    return result


async def transcribe(
    audio_file: UploadFile | None = None,
    batch_size: int | None = None,
    chunk_size: int | None = None,
    asr_options: dict | None = None,
    language: Language | None = None,
    model_name: str | None = None,
    align: bool = False,
    diarize: bool = False,
    speaker_embeddings: bool = False,
    request_id: str = "",
    task: str = "transcribe",
    source_url: str | None = None,
) -> dict[str, Any]:
    if bool(audio_file) == bool(source_url):
        raise ValueError("transcribe requires exactly one of audio_file or source_url")
    config = get_config()
    if batch_size is None:
        batch_size = config.whisper.batch_size
    if chunk_size is None:
        chunk_size = config.whisper.chunk_size
    if language is None:
        language = config.default_language
    start_time = time.perf_counter()
    file_path = None
    audio = None
    concurrency_sem = _get_concurrency_semaphore()
    decode_sem = _get_decode_semaphore()
    profile: dict[str, float] = {}
    audio_duration_seconds = 0.0
    if audio_file is not None:
        source_label = audio_file.filename
    else:
        from whisperx_api_server import url_fetch

        assert source_url is not None  # exactly one of audio_file/source_url
        source_label = url_fetch.filename_from_url(source_url)

    stage_name = "url_download" if source_url is not None else "upload_save"
    try:
        request_status.set_stage(request_id, stage_name)
        t0 = time.perf_counter()
        if source_url is not None:
            from whisperx_api_server import url_fetch

            file_path = await url_fetch.download_url_to_temp(
                source_url,
                request_id,
                max_bytes=config.max_upload_size_bytes,
                connect_timeout=config.url_fetch_connect_timeout_seconds,
                total_timeout=config.url_fetch_timeout_seconds,
                allow_private_hosts=config.url_fetch_allow_private_hosts,
                allowed_hosts=config.url_fetch_allowed_hosts,
            )
        else:
            assert audio_file is not None  # exactly one of audio_file/source_url
            file_path = await _save_upload_to_temp(audio_file, request_id)
        profile[stage_name] = time.perf_counter() - t0
        logger.info(
            "Request ID: %s - Saving source took %.2f seconds",
            request_id,
            profile[stage_name],
        )
        _pipe.stage_duration.labels(stage=stage_name).observe(profile[stage_name])

        # Decode outside the inference semaphore, bounded by the decode-admission semaphore.
        async with contextlib.AsyncExitStack() as _decode_stack:
            if decode_sem is not None:
                await _decode_stack.enter_async_context(decode_sem)
            request_status.set_stage(request_id, "audio_load")
            t0 = time.perf_counter()
            audio = await load_audio_from_path(file_path, request_id)
            profile["audio_load"] = time.perf_counter() - t0
            logger.info(
                "Request ID: %s - Loading audio took %.2f seconds",
                request_id,
                profile["audio_load"],
            )
            _pipe.stage_duration.labels(stage="audio_load").observe(
                profile["audio_load"]
            )
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
            selected_backends.transcription
        )
        alignment_stage_backend = (
            get_alignment_backend(selected_backends.alignment)
            if (align or diarize)
            else None
        )
        diarization_stage_backend = (
            get_diarization_backend(selected_backends.diarization) if diarize else None
        )

        if not model_name:
            model_name = get_default_transcription_model_name()

        logger.info(
            "Request ID: %s - Transcribing %s with model: %s, options: %s, language: %s, task: %s, stage_backends: %s",
            request_id,
            source_label,
            model_name,
            asr_options,
            language,
            task,
            selected_backends,
        )

        _sem_t0 = time.perf_counter()
        if concurrency_sem:
            request_status.set_stage(request_id, "awaiting_concurrency")
            await concurrency_sem.acquire()
        _sem_elapsed = time.perf_counter() - _sem_t0
        logger.debug(
            "Request ID: %s - Acquired inference concurrency semaphore", request_id
        )
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
                "Request ID: %s - Transcription took %.2f seconds",
                request_id,
                profile["transcribe"],
            )
            _pipe.stage_duration.labels(stage="transcribe").observe(
                profile["transcribe"]
            )
            if audio_duration_seconds > 0:
                _pipe.realtime_factor.labels(
                    model=model_name, stage="transcribe"
                ).observe(profile["transcribe"] / audio_duration_seconds)

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
                    "Request ID: %s - Alignment took %.2f seconds",
                    request_id,
                    profile["align"],
                )
                _pipe.stage_duration.labels(stage="align").observe(profile["align"])
                if audio_duration_seconds > 0:
                    _pipe.realtime_factor.labels(
                        model=model_name, stage="align"
                    ).observe(profile["align"] / audio_duration_seconds)

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
                    "Request ID: %s - Diarization took %.2f seconds",
                    request_id,
                    profile["diarize"],
                )
                _pipe.stage_duration.labels(stage="diarize").observe(profile["diarize"])
                if audio_duration_seconds > 0:
                    _pipe.realtime_factor.labels(
                        model=model_name, stage="diarize"
                    ).observe(profile["diarize"] / audio_duration_seconds)

            request_status.set_stage(request_id, "finalize")
            t0 = time.perf_counter()
            result = _finalize_text(result, align or diarize)
            profile["finalize"] = time.perf_counter() - t0

            total = time.perf_counter() - start_time
            logger.info(
                "Request ID: %s - Transcription completed for %s",
                request_id,
                source_label,
            )
            logger.debug(
                "Request ID: %s - profile: total=%.2fs | %s | (other=%.2fs)",
                request_id,
                total,
                " | ".join(f"{k}={v:.2f}s" for k, v in profile.items()),
                total - sum(profile.values()),
            )

            request_status.mark_completed(request_id)
            return result
        finally:
            if concurrency_sem:
                concurrency_sem.release()
    except Exception as e:
        logger.error(
            "Request ID: %s - Transcription failed for %s with error: %s",
            request_id,
            source_label,
            e,
        )
        request_status.mark_failed(request_id, str(e), type(e).__name__)
        raise
    finally:
        with contextlib.suppress(Exception):
            if file_path is not None and os.path.exists(file_path):
                os.remove(file_path)
        if config.audio_cleanup and audio is not None:
            del audio
            logger.info("Request ID: %s - Audio data cleaned up", request_id)
        if config.cache_cleanup:
            _cleanup_cache_only()
            logger.info("Request ID: %s - Cache cleanup completed", request_id)


class QueueFullError(Exception):
    pass


async def _submit_kafka_job(
    audio_file: UploadFile | None,
    source_url: str | None,
    params: dict[str, Any],
    request_id: str,
    *,
    track_future: bool,
    callback_url: str | None = None,
) -> asyncio.Future | None:
    """Upload (or forward the URL), publish the request, and announce the job.

    Shared by the synchronous (track_future=True) and async (False) Kafka paths.
    Returns the reply future when tracked, else None.
    """
    if bool(audio_file) == bool(source_url):
        raise ValueError(
            "Kafka submission requires exactly one of audio_file or source_url"
        )
    config = get_config()
    max_pending = config.kafka.max_pending_jobs
    if max_pending > 0 and len(kafka_client._pending_jobs) >= max_pending:
        _kafka.queue_rejected_total.inc()
        err = QueueFullError(
            f"Too many pending jobs ({len(kafka_client._pending_jobs)}/{max_pending})"
        )
        request_status.mark_failed(request_id, str(err), "QueueFullError")
        raise err

    if source_url is not None:
        from whisperx_api_server import url_fetch

        safe_name = url_fetch.filename_from_url(source_url)
        s3_key: str | None = None
        logger.info(
            "Request ID: %s - Forwarding source URL to worker (skipping S3 upload)",
            request_id,
        )
    else:
        assert audio_file is not None  # exactly one of audio_file/source_url
        safe_name = _safe_filename(audio_file.filename)
        request_status.set_stage(request_id, "uploading_to_s3")
        logger.info("Request ID: %s - Uploading audio to S3", request_id)
        try:
            s3_key = await s3_client.upload_audio_stream(
                audio_file, request_id, safe_name
            )
        except Exception as e:
            request_status.mark_failed(request_id, str(e), type(e).__name__)
            raise

    request_status.set_stage(request_id, "submitted_to_kafka")
    logger.info(
        "Request ID: %s - Submitting job to Kafka (s3_key=%s, audio_url=%s)",
        request_id,
        s3_key,
        "set" if source_url else None,
    )
    try:
        future = await kafka_client.submit_job(
            request_id,
            s3_key,
            source_url,
            safe_name,
            params,
            track_future=track_future,
            callback_url=callback_url,
        )
    except Exception as e:
        request_status.mark_failed(request_id, str(e), type(e).__name__)
        raise

    # Announce the job so replicas that didn't handle this POST can serve /status.
    local_state = request_status.get(request_id)
    await kafka_client.publish_submitted(
        config.kafka,
        request_id,
        filename=(local_state or {}).get("filename", safe_name),
        params=(local_state or {}).get("params"),
    )
    request_status.set_stage(request_id, "awaiting_worker")
    return future


async def submit_kafka_job(
    audio_file: UploadFile | None = None,
    *,
    params: dict[str, Any],
    request_id: str,
    source_url: str | None = None,
    callback_url: str | None = None,
) -> None:
    """Submit a job and return without waiting for the reply (async API). The
    terminal envelope is durable in S3 (results/{job_id}); the caller polls
    /status and fetches the outcome from /result."""
    await _submit_kafka_job(
        audio_file,
        source_url,
        params,
        request_id,
        track_future=False,
        callback_url=callback_url,
    )


async def transcribe_via_kafka(
    audio_file: UploadFile | None = None,
    *,
    params: dict[str, Any],
    request_id: str,
    source_url: str | None = None,
    callback_url: str | None = None,
) -> dict[str, Any]:
    config = get_config()
    future = await _submit_kafka_job(
        audio_file,
        source_url,
        params,
        request_id,
        track_future=True,
        callback_url=callback_url,
    )
    assert future is not None  # track_future=True always registers a future
    timeout = config.kafka.reply_timeout_seconds
    try:
        result = await kafka_client.wait_for_reply(request_id, future, timeout)
        logger.info("Request ID: %s - Received result from worker", request_id)
        request_status.mark_completed(request_id)
        return result
    except TimeoutError as timeout_exc:
        kafka_client.discard_pending(request_id)
        _kafka.job_timeout_total.inc()
        logger.error(
            "Request ID: %s - No worker reply or progress within %.0fs",
            request_id,
            timeout,
        )
        err = TimeoutError(
            f"Job {request_id} timed out: no worker reply or progress within {timeout}s"
        )
        request_status.mark_failed(request_id, str(err), "TimeoutError")
        raise err from timeout_exc
    except asyncio.CancelledError:
        # Client disconnect / shutdown. The worker keeps processing and the reply
        # consumer settles status + pops the entry when the result lands, so the
        # job must not be marked failed here. Cancelling the future just tells
        # the reply consumer there is no awaiter left.
        future.cancel()
        raise
    except Exception as e:
        kafka_client.discard_pending(request_id)
        request_status.mark_failed(request_id, str(e), type(e).__name__)
        raise
