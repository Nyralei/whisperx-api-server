## Overview

WhisperX API Server is a FastAPI-based server designed to transcribe audio files using the Whisper ASR (Automatic Speech Recognition) model based on WhisperX (https://github.com/m-bain/WhisperX) Python library. The API offers an OpenAI-like interface that allows users to upload audio files and receive transcription results in various formats. It supports customizable options such as different models, languages, temperature settings, and more.

Features
1. Audio Transcription: Transcribe audio files using the Whisper ASR model.
2. Model Caching: Load and cache models for reusability and faster performance.
3. OpenAI-like API, based on https://platform.openai.com/docs/api-reference/audio/createTranscription

## API Endpoints

### `POST /v1/audio/transcriptions`
This is the main endpoint for uploading audio files and receiving transcriptions.

**Parameters**:
- `file`: The audio file to transcribe.
- `model (str)`: The Whisper model to use. Default is `config.whisper.model`.
- `language (str)`: The language for transcription. Default is `config.default_language`.
- `prompt (str)`: Optional transcription prompt.
- `response_format (str)`: The format of the transcription output. Defaults to `json`.
- `temperature (float)`: Temperature setting for transcription. Default is `0.0`.
- `timestamp_granularities (list)`: Granularity of timestamps, either `segment` or `word`. Default is `["segment"]`. Currently doesn't work with OpenAI client libraries.
- `stream (bool)`: Enable streaming mode for real-time transcription. WIP.
- `hotwords (str)`: Optional hotwords for transcription.
- `suppress_numerals (bool)`: Option to suppress numerals in the transcription. Default is `True`.
- `highlight_words (bool)`: Highlight words in the transcription output for formats like VTT and SRT.

**Returns**: Transcription results in the specified format.

### `GET /healthcheck`
Returns the current health status of the API server.

### `GET /models/list`
Lists all loaded models currently available on the server.

### `POST /models/unload`
Unloads a specific model from memory cache.

### `POST /models/load`
Loads a specified model into memory.

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