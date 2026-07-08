# WhisperX API Server

A FastAPI server that exposes [WhisperX](https://github.com/m-bain/WhisperX) as an OpenAI-compatible audio transcription API. Supports both a simple single-server mode and a horizontally scalable distributed mode backed by Kafka and S3.

## Features

- **OpenAI-compatible** — drop-in replacement for `/v1/audio/transcriptions` and `/v1/audio/translations`
- **Alignment & diarization** — word-level timestamps and speaker labels out of the box
- **Multiple output formats** — `json`, `verbose_json`, `vtt_json`, `srt`, `vtt`, `aud`, `text`
- **Distributed mode** — offload GPU work to dedicated workers via Kafka + S3 (MinIO)
- **Live request status** — poll an in-flight transcription's current pipeline stage by request id, in both direct and Kafka modes
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

> Workers process one job at a time per container. Scale horizontally by running multiple worker replicas. On `SIGTERM` (stop / rolling update) a worker finishes its in-flight job — reply, then offset commit — before exiting, and refuses to start a new one. Set the container `stop_grace_period` to your worst-case job duration so a long job isn't killed mid-flight; a job killed before commit is safely redelivered (idempotent resend) at the cost of one reprocess.

> Delivery is at-least-once. A worker writes each job's result envelope to S3 (`results/{job_id}`) before replying, so a redelivered job resends the stored reply instead of re-running. Each job is guarded by a processing lease (`claims/{job_id}`, TTL `KAFKA__JOB_LEASE_TTL_SECONDS`, default 300s): a copy redelivered mid-run (e.g. after a consumer-group rebalance) defers instead of starting a concurrent duplicate, and takes over only once the lease expires. The lease also counts delivery attempts — a job that repeatedly kills its worker is, past `KAFKA__MAX_DELIVERY_ATTEMPTS` (default 3), routed to the `transcription-dlq` topic so the submitter fails fast instead of every worker dying on it. Both `results/` and `claims/` objects expire via the bucket lifecycle (`S3__OBJECT_EXPIRY_DAYS`, default 1 day).

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
| `AUTH_REQUIRED` | `false` | When `true`, refuse to start unless `API_KEY` or `API_KEYS_FILE` is set |
| `MODE` | `direct` | `direct` or `kafka` |
| `MAX_CONCURRENT_TRANSCRIPTIONS` | `1` | Max concurrent ML inferences (transcribe / align / diarize). `0` = unlimited. See the concurrency note below. |

> **`MAX_CONCURRENT_TRANSCRIPTIONS` parallelizes across *distinct* models, not within one.** Each transcription pipeline holds a per-model lock during the transcribe step, so two requests for the **same** model still run one at a time even with the limit raised — the setting lets a request for a *different* model (and the align / diarize stages) proceed concurrently. To raise same-model throughput, add GPUs or run more replicas / workers. Whichever process runs inference (the API in direct mode, the worker in Kafka mode) logs this caveat at startup when the limit is >1.

> **Auth is off by default.** With neither `API_KEY` nor `API_KEYS_FILE` set, the server accepts every request without credentials and logs a startup warning. Set either to enforce auth (missing header → 401, invalid key → 403), or set `AUTH_REQUIRED=true` to turn an unconfigured deployment into a startup failure rather than an open server.

**Additional variables for Kafka mode:**

| Variable | Default | Description |
|---|---|---|
| `KAFKA__BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka broker address |
| `KAFKA__PROGRESS_TOPIC` | `transcription-progress` | Best-effort topic for per-stage worker progress events consumed by the status endpoint |
| `S3__ENDPOINT_URL` | `http://localhost:9000` | S3 / MinIO endpoint |
| `S3__BUCKET` | `whisperx-audio` | Bucket for audio uploads |
| `MINIO_ROOT_USER` | `minioadmin` | MinIO root user — **change before deploying** |
| `MINIO_ROOT_PASSWORD` | `minioadmin` | MinIO root password — **change before deploying** |

**Status-endpoint tuning (both modes):**

| Variable | Default | Description |
|---|---|---|
| `REQUEST_STATUS__TTL_SECONDS` | `300` | How long terminal states (completed / failed) are retained for polling |
| `REQUEST_STATUS__MAX_ENTRIES` | `4096` | Hard cap on tracked requests; terminal entries are evicted first when over capacity |
| `REQUEST_STATUS__CLEANUP_INTERVAL_SECONDS` | `30` | How often the background sweep evicts expired entries |

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
| `async` | bool | `false` | **Kafka mode only.** Return `202 Accepted` immediately instead of blocking; fetch the outcome later. See [Async job submission](#async-job-submission-kafka-mode). |
| `callback_url` | string | — | Optional URL the result envelope is POSTed to when the job finishes. Validated against the same SSRF policy as `audio_url`. See [Completion webhook](#completion-webhook). |

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

### `GET /v1/audio/transcriptions/{request_id}/status`

Return the live processing stage for a transcription request. Useful for surfacing a "still working — currently transcribing" indicator in long-running UIs.

Because the transcription POST is synchronous (the response only arrives when the whole pipeline finishes), the client must set its own id on the POST so it can poll status in parallel:

```bash
# Submit with a known id
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H 'X-Request-ID: my-request-1' \
  -F file=@audio.mp3 -F model=large-v3 -F align=true &

# Poll status from another shell
curl http://localhost:8000/v1/audio/transcriptions/my-request-1/status
```

The middleware accepts client-supplied `X-Request-ID` values matching `[A-Za-z0-9._-]{1,128}`; anything else is rejected and replaced with a server-generated UUID (returned via the response header, by which point it is too late to poll).

**Response (200)**

```jsonc
{
  "request_id": "my-request-1",
  "status": "in_progress",            // queued | in_progress | completed | failed
  "mode": "direct",                   // or "kafka"
  "stage": "transcribe",              // name of the active stage
  "submitted_at": 1779198758.12,
  "updated_at":   1779198764.16,
  "completed_at": null,               // set when status is terminal
  "filename": "audio.mp3",
  "stages": [
    {"name": "upload_save",           "duration_seconds": 0.003, "started_at": 1779198758.12, "completed_at": 1779198758.12, "in_progress": false},
    {"name": "audio_load",            "duration_seconds": 0.374, "started_at": 1779198758.12, "completed_at": 1779198758.49, "in_progress": false},
    {"name": "awaiting_concurrency",  "duration_seconds": 0.0,   "started_at": 1779198758.49, "completed_at": 1779198758.49, "in_progress": false},
    {"name": "transcribe",                                       "started_at": 1779198758.49,                                "in_progress": true}
  ],
  "error": null,
  "error_type": null
}
```

**Stages**

| Mode | Stage names |
|---|---|
| direct | `upload_save`, `audio_load`, `awaiting_concurrency`, `transcribe`, `align`, `diarize`, `finalize` |
| kafka (API-side) | `uploading_to_s3`, `submitted_to_kafka`, `awaiting_worker` |
| kafka (worker-side, via `transcription-progress` topic) | `worker.s3_download`, `worker.audio_load`, `worker.awaiting_gpu`, `worker.transcribe`, `worker.align`, `worker.diarize`, `worker.finalize` |

Failures (invalid audio, queue full, timeout, worker error, …) end the lifecycle with `status="failed"` and populate `error` / `error_type`. Stages completed before the failure are preserved.

Terminal states (`completed` / `failed`) are retained for `REQUEST_STATUS__TTL_SECONDS` (default 300s) so polling clients that arrive just after the POST returns can still confirm the outcome. After that, the id 404s.

**Other responses**

- `400 Bad Request` — malformed `request_id` (must match `[A-Za-z0-9._-]{1,128}`)
- `404 Not Found` — id is unknown, expired, or not yet seen by this replica

> Both the request/reply path and `/status` scale across API replicas in Kafka mode. Each reply is delivered to every replica and only the one holding the job resolves it, so any replica can serve the POST. Status converges too: the submitting replica announces the job on the progress topic and every replica consumes the progress stream, so `GET /status` works on any replica behind a load balancer — no sticky sessions required. Each replica's tracker therefore holds entries for jobs across all replicas (bounded by `REQUEST_STATUS__MAX_ENTRIES`, default 4096, well above `KAFKA__MAX_PENDING_JOBS`). In-flight status is still in-memory: a replica restart loses the live history for jobs it learned about (they would expire within minutes anyway), while completed/failed outcomes stay readable for `REQUEST_STATUS__TTL_SECONDS`. Direct mode is single-process and unaffected.

---

### Async job submission (Kafka mode)

By default the transcription POST is synchronous — the HTTP response arrives only when the whole pipeline finishes. In **Kafka mode** you can instead submit the job and return immediately by setting the `async` form field, then fetch the result later. This decouples slow transcriptions from the request connection (no client read-timeout to tune) and survives an API-replica restart, because the result is read from durable storage rather than an in-memory future.

```bash
# Submit — returns 202 without waiting for transcription
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H 'X-Request-ID: my-async-1' \
  -F file=@audio.mp3 -F model=large-v3 -F align=true \
  -F async=true
```

**Response (202 Accepted)**

```jsonc
{
  "request_id": "my-async-1",
  "status": "accepted",
  "status_url": "/v1/audio/transcriptions/my-async-1/status",
  "result_url": "/v1/audio/transcriptions/my-async-1/result"
}
```

Poll `status_url` (see above) to follow progress, then fetch `result_url` once `status` is `completed`. Unlike the synchronous path, you don't need to set `X-Request-ID` up front — the `202` body returns the resolved `request_id` and both URLs immediately, whether the id was client-supplied or server-generated (a `uuid4`). Supplying your own `X-Request-ID` (matching `[A-Za-z0-9._-]{1,128}`) is optional: a predictable id for log correlation. Async is rejected in direct mode with `400 Bad Request`.

---

### `GET /v1/audio/transcriptions/{request_id}/result`

Fetch the final result of an async job (Kafka mode only). It reads the stored `results/{job_id}` envelope from S3, so it works from any replica and after restarts, with no in-memory state required.

**Query parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `response_format` | string | config default | Same formats as the POST: `text`, `json`, `verbose_json`, `vtt_json`, `srt`, `vtt`, `aud` |
| `highlight_words` | bool | `false` | Highlight words in `vtt`/`srt` output |

Formatting is applied at fetch time from these query parameters — the `response_format` set on the async POST is not stored, so you choose the format (and may request several) when you fetch.

**Responses**

- `200 OK` — formatted transcription (Content-Type per `response_format`), identical to what the synchronous POST would have returned
- `400 Bad Request` — malformed `request_id`
- `404 Not Found` — result not available (still pending, unknown, or expired); also returned in direct mode
- worker failures are replayed with the same status code the synchronous endpoint returns (e.g. invalid audio → `422`, timeout → `504`), not `404`

Results are retained for `S3__OBJECT_EXPIRY_DAYS` (default 1 day) via the bucket lifecycle policy; after that the id 404s.

---

### Completion webhook

Add a `callback_url` form field to any transcription POST to have the result **pushed** to you when the job finishes, instead of (or alongside) polling. The URL is validated up front against the same SSRF policy as `audio_url` (`URL_FETCH_ALLOW_PRIVATE_HOSTS` / `URL_FETCH_ALLOWED_HOSTS`); a rejected host fails the request with `422` before any work starts.

On completion the server sends a single `POST` to `callback_url` with the terminal envelope as the JSON body:

```jsonc
{
  "job_id": "my-async-1",
  "status": "ok",      // or "error"
  "result": { },       // raw transcript (segments, language); absent on error
  "error": "...",      // present only when status is "error"
  "error_type": "..."
}
```

Format the `result` yourself, or ignore the body and fetch `result_url` for a formatted response.

- **Who delivers:** in **Kafka mode** the worker delivers — it is the single point that runs the job and holds the envelope, and it survives an API-replica restart. In **direct mode** the API delivers in-process after returning the synchronous response.
- **Both outcomes notify (Kafka mode):** success, handled failures, and jobs retired to the dead-letter queue all fire the webhook with the matching envelope. Direct mode fires on success only — a direct-mode failure is already returned to the (synchronously waiting) caller as an HTTP error.
- **Delivery guarantee:** best-effort with one retry, bounded by `WEBHOOK_TIMEOUT_SECONDS` (default 15s). It fires at most once per fresh completion and is **never** re-sent on the redelivery/marker-resend path, so a worker restart between the reply and the callback can drop it. Treat the durable `result_url` as the source of truth and de-duplicate on `job_id`. A non-2xx response or connection error is logged and dropped, not retried indefinitely.

---

### `GET /info`

Returns the running version, mode, uptime, concurrency / queue state, and (in Kafka mode) discovered worker membership. Add `?detail=full` for extended Kafka topology.

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

## Local Installation (pip / uv)

```bash
# API server for the distributed (Kafka) setup — no ML dependencies required:
pip install ".[kafka]"

# API server with local inference (direct mode):
pip install ".[cpu]"        # or ".[cuda]" for GPU

# Kafka worker:
pip install ".[cpu,kafka]"  # or ".[cuda,kafka]"
```

Two console scripts are installed:

```bash
whisperx-api      # start the API server (UVICORN_HOST:UVICORN_PORT, default 0.0.0.0:8000)
whisperx-worker   # start a Kafka worker
```

Without the `cpu`/`cuda` extras, PyTorch and WhisperX are not installed. Such a server can still take requests and hand them to workers in Kafka mode, but direct-mode inference and the subtitle response formats (`srt`, `vtt`, `vtt_json`, `aud`) need the ML extras and return a clear error otherwise.

## Contributing

Issues, forks, and pull requests are welcome.

## License

GNU General Public License v3.0 — see [`LICENSE`](LICENSE) for details.
