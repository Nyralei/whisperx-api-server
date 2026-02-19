## Overview

WhisperX API Server is a FastAPI-based server designed to transcribe audio files using the Whisper ASR (Automatic Speech Recognition) model based on WhisperX (https://github.com/m-bain/WhisperX) Python library. The API offers an OpenAI-like interface that allows users to upload audio files and receive transcription results in various formats. It supports customizable options such as different models, languages, temperature settings, and more.

Features
1. Audio Transcription: Transcribe audio files using the Whisper ASR model.
2. Model Caching: Load and cache models for reusability and faster performance.
3. OpenAI-like API, based on https://platform.openai.com/docs/api-reference/audio/createTranscription and https://platform.openai.com/docs/api-reference/audio/createTranslation

## API Endpoints

### `POST /v1/audio/transcriptions`
https://platform.openai.com/docs/api-reference/audio/createTranscription

**Parameters**:
- `file`: The audio file to transcribe.
- `model (str)`: Whisper model name. Default is `config.whisper.model`. If `whisper-1` is provided, it is replaced with the configured default model.
- `language (str | null)`: Language code for transcription. Default is `config.default_language`.
- `prompt (str | null)`: Optional transcription prompt. Default is `null`.
- `response_format (str)`: One of `text`, `json`, `verbose_json`, `vtt_json`, `srt`, `vtt`, `aud`. Default is `config.default_response_format`.
- `temperature (float)`: Temperature setting for transcription. Default is `0.0`.
- `timestamp_granularities[] (list[str])`: Timestamp granularity values (`segment`, `word`). Default is `["segment"]`.
- `stream (bool)`: OpenAI-compatible streaming flag. Currently accepted but not used by the server. Default is `False`.
- `hotwords (str | null)`: Optional hotwords for transcription. Default is `null`.
- `suppress_numerals (bool)`: Suppress numerals in transcription. Default is `True`.
- `highlight_words (bool)`: Highlight words in subtitle-style outputs (`vtt`, `srt`). Default is `False`.
- `align (bool)`: Enable transcription timing alignment. Default is `True`.
- `diarize (bool)`: Enable speaker diarization. Default is `False`.
- `speaker_embeddings (bool)`: Include speaker embeddings during diarization flow. Default is `False`.
- `chunk_size (int)`: Chunk size (seconds) for VAD segment merging. Default is `config.whisper.chunk_size`.
- `batch_size (int)`: Batch size used during inference. Default is `config.whisper.batch_size`.

**Returns**: Transcription output in the requested `response_format`:
- `json`: JSON object with `text`.
- `verbose_json`: Full transcript JSON object.
- `vtt_json`: Full transcript JSON object plus `vtt_text`.
- `text`, `srt`, `vtt`, `aud`: Plain text response body.

### `POST /v1/audio/translations`
https://platform.openai.com/docs/api-reference/audio/createTranslation

**Parameters**:
- `file`: The audio file to translate.
- `model (str)`: Whisper model name. Default is `config.whisper.model`. If `whisper-1` is provided, it is replaced with the configured default model.
- `prompt (str)`: Optional translation prompt. Default is an empty string.
- `response_format (str)`: One of `text`, `json`, `verbose_json`, `vtt_json`, `srt`, `vtt`, `aud`. Default is `config.default_response_format`.
- `temperature (float)`: Temperature setting for translation. Default is `0.0`.
- `chunk_size (int)`: Chunk size (seconds) for VAD segment merging. Default is `config.whisper.chunk_size`.
- `batch_size (int)`: Batch size used during inference. Default is `config.whisper.batch_size`.

**Returns**: Translation output in the requested `response_format` (same response behavior as `/v1/audio/transcriptions`).

### `GET /healthcheck`
Returns current API health status as JSON: `{"status": "healthy"}`.

### `GET /models/list`
Lists loaded transcription models.

### `POST /models/unload`
Unloads a transcription model from cache.

**Parameters**:
- `model (str)`: Model name to unload.

### `POST /models/load`
Loads a transcription model into cache.

**Parameters**:
- `model (str)`: Model name to load.

### `GET /align_models/list`
Lists loaded alignment models.

### `POST /align_models/unload`
Unloads an alignment model.

**Parameters**:
- `language (str)`: Language code of the alignment model to unload.

### `POST /align_models/load`
Loads an alignment model.

**Parameters**:
- `language (str)`: Language code of the alignment model to load.

### `GET /diarize_models/list`
Lists loaded diarization models.

### `POST /diarize_models/unload`
Unloads a diarization model.

**Parameters**:
- `model (str)`: Diarization model name to unload.

### `POST /diarize_models/load`
Loads a diarization model.

**Parameters**:
- `model (str)`: Diarization model name to load.

### Running the API

**With Docker**:

For CPU:
```bash
    docker compose build whisperx-api-server-cpu

    docker compose up whisperx-api-server-cpu
```

For CUDA (GPU):
```bash
    docker compose build whisperx-api-server-cuda

    docker compose up whisperx-api-server-cuda

```

## Contributing

Feel free to submit issues, fork the repository, and send pull requests to contribute to the project.

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE Version 3. See the `LICENSE` file for details.