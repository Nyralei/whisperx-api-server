# WhisperX API Server

A FastAPI server that exposes [WhisperX](https://github.com/m-bain/WhisperX) as an OpenAI-compatible audio transcription API. Supports both a simple single-server mode and a horizontally scalable distributed mode backed by Kafka and S3.

## Features

- **OpenAI-compatible** — drop-in replacement for `/v1/audio/transcriptions` and `/v1/audio/translations`
- **Alignment & diarization** — word-level timestamps and speaker labels out of the box
- **Multiple output formats** — `json`, `verbose_json`, `vtt_json`, `srt`, `vtt`, `aud`, `text`
- **Distributed mode** — offload GPU work to dedicated workers via Kafka + S3 (MinIO)
- **Pluggable backends** — swap transcription, alignment, and diarization implementations per stage
- **API key auth** — single key or a JSON key-map for multi-client setups

## Quick Start

Each profile selects exactly one combination — service set, image tags, and `--extra` build args are all driven by the profile so images stay minimal.

### Standalone (single server)

```bash
# 1. Normal CUDA              → image: whisperx-api:cuda
docker compose --profile cuda up

# 2. Normal CPU               → image: whisperx-api:cpu
docker compose --profile cpu up

# 5. CUDA + Observability     → image: whisperx-api:cuda-metrics  (+ Prometheus)
docker compose --profile cuda-observe up

# 6. CPU + Observability      → image: whisperx-api:cpu-metrics   (+ Prometheus)
docker compose --profile cpu-observe up
```

The API is available at `http://localhost:8000`. With an `*-observe` profile, Prometheus is on `:9090` (point your own Grafana / dashboard at it).

### Distributed mode (Kafka + workers)

```bash
# Copy and edit credentials before first run
cp .env.example .env

# 3. CUDA + Kafka                       → api: whisperx-api:kafka,         worker: whisperx-worker:cuda-kafka
docker compose -f compose-kafka.yaml --profile cuda up

# 4. CPU + Kafka                        → api: whisperx-api:kafka,         worker: whisperx-worker:cpu-kafka
docker compose -f compose-kafka.yaml --profile cpu up

# 7. CUDA + Kafka + Observability       → api: whisperx-api:kafka-metrics, worker: whisperx-worker:cuda-kafka-metrics
docker compose -f compose-kafka.yaml --profile cuda-observe up

# 8. CPU + Kafka + Observability        → api: whisperx-api:kafka-metrics, worker: whisperx-worker:cpu-kafka-metrics
docker compose -f compose-kafka.yaml --profile cpu-observe up
```

> Workers process one job at a time per container. Scale horizontally by running multiple worker replicas.

### Profile matrix

| # | Mode | Profile | Compose file |
|---|---|---|---|
| 1 | Normal CUDA | `cuda` | `compose.yaml` |
| 2 | Normal CPU | `cpu` | `compose.yaml` |
| 3 | CUDA + Kafka | `cuda` | `compose-kafka.yaml` |
| 4 | CPU + Kafka | `cpu` | `compose-kafka.yaml` |
| 5 | CUDA + Observability | `cuda-observe` | `compose.yaml` |
| 6 | CPU + Observability | `cpu-observe` | `compose.yaml` |
| 7 | CUDA + Kafka + Observability | `cuda-observe` | `compose-kafka.yaml` |
| 8 | CPU + Kafka + Observability | `cpu-observe` | `compose-kafka.yaml` |

## Configuration

All settings are environment variables. Nested fields use `__` as a delimiter (e.g. `WHISPER__MODEL=large-v3`).

All available settings are defined in [`config.py`](src/whisperx_api_server/config.py). Variables you'll most likely need to set:

| Variable | Default | Description |
|---|---|---|
| `WHISPER__MODEL` | `large-v3` | Transcription model name |
| `WHISPER__COMPUTE_TYPE` | `default` | Quantization — `float16` for GPU, `float32` for CPU |
| `WHISPER__INFERENCE_DEVICE` | `auto` | `cpu`, `cuda`, or `auto` |
| `HF_TOKEN` | — | Hugging Face token (required for pyannote diarization) |
| `API_KEY` | — | Single API key for all requests |
| `API_KEYS_FILE` | — | Path to JSON file mapping key → client name |
| `MODE` | `direct` | `direct` or `kafka` |

**Additional variables for Kafka mode:**

| Variable | Default | Description |
|---|---|---|
| `KAFKA__BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka broker address |
| `S3__ENDPOINT_URL` | `http://localhost:9000` | S3 / MinIO endpoint |
| `S3__BUCKET` | `whisperx-audio` | Bucket for audio uploads |
| `MINIO_ROOT_USER` | `minioadmin` | MinIO root user — **change before deploying** |
| `MINIO_ROOT_PASSWORD` | `minioadmin` | MinIO root password — **change before deploying** |

## API Reference

### `POST /v1/audio/transcriptions`

Transcribe an audio file. Compatible with the [OpenAI transcription API](https://platform.openai.com/docs/api-reference/audio/createTranscription).

**Form parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | — | Audio file (required) |
| `model` | string | config default | Model name. `whisper-1` is aliased to the configured default. |
| `language` | string | config default | ISO-639-1 language code. Auto-detected if omitted. |
| `prompt` | string | — | Optional context/hotwords hint |
| `response_format` | string | `json` | `text`, `json`, `verbose_json`, `vtt_json`, `srt`, `vtt`, `aud` |
| `temperature` | float | `0.0` | Sampling temperature |
| `timestamp_granularities[]` | list | `["segment"]` | `segment`, `word` |
| `align` | bool | `true` | Enable word-level alignment (required for subtitle formats) |
| `diarize` | bool | `false` | Enable speaker diarization (requires `align=true`) |
| `speaker_embeddings` | bool | `false` | Include speaker embeddings in diarization output |
| `highlight_words` | bool | `false` | Highlight words in `vtt`/`srt` output |
| `suppress_numerals` | bool | `true` | Spell out numbers |
| `hotwords` | string | — | Comma-separated hotwords to bias toward |
| `batch_size` | int | config default | Inference batch size |
| `chunk_size` | int | config default | VAD chunk size in seconds |

**Response formats**

| Format | Content-Type | Body |
|---|---|---|
| `json` | `application/json` | `{"text": "..."}` |
| `verbose_json` | `application/json` | Full transcript with segments and timestamps |
| `vtt_json` | `application/json` | `verbose_json` + `"vtt_text"` field |
| `text` / `srt` / `aud` | `text/plain` | Raw text / subtitle file |
| `vtt` | `text/vtt` | WebVTT subtitle file |

---

### `POST /v1/audio/translations`

Translate audio to English. Same parameters as `/v1/audio/transcriptions`, minus `language`, `align`, `diarize`, and diarization-related fields.

---

### `GET /healthcheck`

Returns `{"status": "healthy"}`. Not protected by API key auth.

---

### Model management

| Endpoint | Description |
|---|---|
| `GET /models/list` | List loaded transcription models |
| `POST /models/load` | Load a model (`model` param) |
| `POST /models/unload` | Unload a model (`model` param) |
| `GET /align_models/list` | List loaded alignment models |
| `POST /align_models/load` | Load an alignment model (`language` param) |
| `POST /align_models/unload` | Unload an alignment model (`language` param) |
| `GET /diarize_models/list` | List loaded diarization models |
| `POST /diarize_models/load` | Load a diarization model (`model` param) |
| `POST /diarize_models/unload` | Unload a diarization model (`model` param) |

## Pluggable Backends

Each pipeline stage (transcription, alignment, diarization) can use a different backend. Set the active backend via environment variables:

```bash
BACKENDS__TRANSCRIPTION=whisperx
BACKENDS__ALIGNMENT=whisperx
BACKENDS__DIARIZATION=whisperx
```

Only the `whisperx` backend ships by default. Custom backends can be registered via the backend registry at `src/whisperx_api_server/backends/`.

## Compose Files

| File | Purpose |
|---|---|
| `compose.yaml` | Standalone server — profiles: `cuda`, `cpu`, `cuda-observe`, `cpu-observe` |
| `compose-kafka.yaml` | Distributed stack (API + Kafka + MinIO + workers) — same four profiles |

Every runtime variant is gated by exactly one profile, so `docker compose up` never accidentally starts a GPU process on a machine that doesn't have one and observability stacks never spawn duplicate API servers.

## Contributing

Issues, forks, and pull requests are welcome.

## License

GNU General Public License v3.0 — see [`LICENSE`](LICENSE) for details.
