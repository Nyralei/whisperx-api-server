import os
import shutil
import torch
from whisperx import asr as whisperx_asr
from whisperx import transcribe as whisperx_transcribe
from whisperx import audio as whisperx_audio
from whisperx import alignment as whisperx_alignment
from whisperx import diarize as whisperx_diarize
from typing import Union, List, Optional
from fastapi import UploadFile
import logging
import time

from whisperx_api_server.config import config
from whisperx_api_server.formatters import format_transcription
from whisperx_api_server.models import (
    load_align_model_cached,
    load_diarize_model_cached,
)

logger = logging.getLogger("transcriber_logger")
logger.setLevel(config.log_level)

class CustomWhisperModel(whisperx_asr.WhisperModel):
    def __init__(
        self,
        model_size_or_path: str,
        device: str = "auto",
        device_index: Union[int, List[int]] = 0,
        compute_type: str = "default",
        cpu_threads: int = 0,
        num_workers: int = 1,
        download_root: Optional[str] = None,
        local_files_only: bool = False,
    ):
        # Call the parent class's __init__ method
        super().__init__(
            model_size_or_path=model_size_or_path,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
            download_root=download_root,
            local_files_only=local_files_only,
        )
        # Explicitly store the parameters as instance attributes
        self.model_size_or_path = model_size_or_path
        self.device = device
        self.device_index = device_index
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.num_workers = num_workers
        self.download_root = download_root
        self.local_files_only = local_files_only

def check_device():
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        logger.error("Could not determine device. Using 'cpu' instead.")
        return "cpu"

async def initialize_model(model_name) -> CustomWhisperModel:
    inference_device = config.whisper.inference_device.value
    if inference_device == "auto":
        inference_device = check_device()

    return CustomWhisperModel(
        model_size_or_path=model_name,
        device=inference_device,
        device_index=config.whisper.device_index,
        compute_type=config.whisper.compute_type.value,
        cpu_threads=config.whisper.cpu_threads,
        num_workers=config.whisper.num_workers,
    )

async def transcribe(
    audio_file: UploadFile,
    batch_size,
    asr_options,
    language,
    response_format,
    whispermodel,
    highlight_words,
    diarize
):
    start_time = time.time()  # Start timing
    file_path = f"/tmp/{audio_file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)

    logger.info(f"Saving uploaded file took {time.time() - start_time:.2f} seconds")

    try:
        logger.info(f"Transcribing {audio_file.filename} with model: {whispermodel.model_size_or_path} and options: {asr_options}")
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
        )
        logger.info(f"Loading model took {time.time() - model_loading_start:.2f} seconds")

        audio_loading_start = time.time()
        audio = whisperx_audio.load_audio(file_path)
        logger.info(f"Loading audio took {time.time() - audio_loading_start:.2f} seconds")

        transcription_start = time.time()
        result = model.transcribe(audio=audio, batch_size=batch_size)
        logger.info(f"Transcription took {time.time() - transcription_start:.2f} seconds")

        alignment_model_start = time.time()
        model_a, metadata = load_align_model_cached(
            language_code=result["language"],
            device=whispermodel.device
        )
        logger.info("Alignment model loaded")
        logger.info(f"Loading alignment model took {time.time() - alignment_model_start:.2f} seconds")

        alignment_start = time.time()
        result["segments"] = whisperx_alignment.align(
            transcript=result["segments"],
            model=model_a,
            align_model_metadata=metadata,
            audio=audio,
            device=whispermodel.device,
            return_char_alignments=False
        )
        logger.info(f"Alignment took {time.time() - alignment_start:.2f} seconds")

        if diarize:
            diarize_start = time.time()

            logger.info("Loading diarization model")

            diarize_model = load_diarize_model_cached(model_name="tensorlake/speaker-diarization-3.1", device=whispermodel.device)

            logger.info("Diarization model loaded. Starting diarization...")

            diarize_segments = diarize_model(audio)

            result["segments"] = whisperx_diarize.assign_word_speakers(diarize_segments, result["segments"])

            logger.info(f"Diarization took {time.time() - diarize_start:.2f} seconds")

        result["text"] = '\n'.join([segment["text"].strip() for segment in result["segments"]["segments"] if segment["text"].strip()])

        logger.info(f"Transcription completed for {audio_file.filename}")

    finally:
        try:
            os.remove(file_path)
        except Exception:
            logger.error(f"Could not remove temporary file: {file_path}")

    return format_transcription(result, response_format, highlight_words=highlight_words)