import os
from whisperx import transcribe as whisperx_transcribe
from whisperx import audio as whisperx_audio
from whisperx import alignment as whisperx_alignment
from whisperx import diarize as whisperx_diarize
from whisperx import types as whisperx_types
from fastapi import UploadFile
import logging
import time
import tempfile
import torch
import gc

from whisperx_api_server.config import (
    Language,
)
from whisperx_api_server.dependencies import get_config
from whisperx_api_server.models import (
    CustomWhisperModel,
    load_align_model_cached,
    load_diarize_model_cached,
)

logger = logging.getLogger(__name__)

config = get_config()

def cleanup_cache_only():
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

async def transcribe(
    audio_file: UploadFile,
    batch_size: int = config.batch_size,
    chunk_size: int = 30,
    asr_options: dict = {},
    language: Language = config.default_language,
    whispermodel: CustomWhisperModel = config.whisper.model,
    align: bool = False,
    diarize: bool = False,
    request_id: str = "",
    task: str = "transcribe",
) -> whisperx_types.TranscriptionResult:
    start_time = time.time()  # Start timing
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{audio_file.filename}") as temp_file:
        temp_file.write(audio_file.file.read())
        file_path = temp_file.name

    logger.info(f"Request ID: {request_id} - Saving uploaded file took {time.time() - start_time:.2f} seconds")

    audio = None
    
    try:
        logger.info(f"Request ID: {request_id} - Transcribing {audio_file.filename} with model: {whispermodel.model_size_or_path} and options: {asr_options}")
        model_loading_start = time.time()
        model = whisperx_transcribe.load_model(
            whisper_arch=whispermodel.model_size_or_path,
            device=whispermodel.device,
            compute_type=whispermodel.compute_type,
            language=language,
            asr_options=asr_options,
            vad_model=config.whisper.vad_model,
            vad_method=config.whisper.vad_method,
            vad_options=config.whisper.vad_options,
            model=whispermodel,
            task=task,
        )
        logger.info(f"Request ID: {request_id} - Loading model took {time.time() - model_loading_start:.2f} seconds")

        audio_loading_start = time.time()
        audio = whisperx_audio.load_audio(file_path)
        logger.info(f"Request ID: {request_id} - Loading audio took {time.time() - audio_loading_start:.2f} seconds")

        transcription_start = time.time()
        result = model.transcribe(
            audio=audio,
            batch_size=batch_size,
            chunk_size=chunk_size,
            num_workers=config.whisper.num_workers,
            language=language,
            task=task,
        )
        logger.info(f"Request ID: {request_id} - Transcription took {time.time() - transcription_start:.2f} seconds")

        if align or diarize:
            alignment_model_start = time.time()
            logger.info(f"Request ID: {request_id} - Loading alignment model")
            model_a, metadata = await load_align_model_cached(
                language_code=result["language"],
            )
            logger.info(f"Request ID: {request_id} - Alignment model loaded")
            logger.info(f"Request ID: {request_id} - Loading alignment model took {time.time() - alignment_model_start:.2f} seconds")

            alignment_start = time.time()
            result["segments"] = whisperx_alignment.align(
                transcript=result["segments"],
                model=model_a,
                align_model_metadata=metadata,
                audio=audio,
                device=whispermodel.device,
                return_char_alignments=False
            )
            logger.info(f"Request ID: {request_id} - Alignment took {time.time() - alignment_start:.2f} seconds")

        if diarize:
            diarization_model_start = time.time()

            logger.info(f"Request ID: {request_id} - Loading diarization model")

            diarize_model = await load_diarize_model_cached(model_name="tensorlake/speaker-diarization-3.1")

            logger.info(f"Request ID: {request_id} - Diarization model loaded. Loading took {time.time() - diarization_model_start:.2f} seconds. Starting diarization")

            diarize_start = time.time()

            diarize_segments = diarize_model(audio)

            result["segments"] = whisperx_diarize.assign_word_speakers(diarize_segments, result["segments"])

            logger.info(f"Request ID: {request_id} - Diarization took {time.time() - diarize_start:.2f} seconds")

        if align or diarize:
            result["text"] = '\n'.join([segment["text"].strip() for segment in result["segments"]["segments"] if segment["text"].strip()])
        else:
            result["text"] = '\n'.join([segment["text"].strip() for segment in result["segments"] if segment["text"].strip()])

        logger.info(f"Request ID: {request_id} - Transcription completed for {audio_file.filename}")
    except Exception as e:
        logger.error(f"Request ID: {request_id} - Transcription failed for {audio_file.filename} with error: {e}")
        raise
    finally:
        try:
            os.remove(file_path)
        except Exception:
            logger.error(f"Request ID: {request_id} - Could not remove temporary file: {file_path}")
        
        if audio is not None:
            del audio
            logger.debug(f"Request ID: {request_id} - Audio data cleaned up")
        
        cleanup_cache_only()
        logger.debug(f"Request ID: {request_id} - Cache cleanup completed")

    return result