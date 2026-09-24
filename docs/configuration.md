# Configuration

All settings are environment variables. Nested fields use `__` as a delimiter (e.g. `WHISPER__MODEL=large-v3`).

All available settings are defined in [`config.py`](../src/whisperx_api_server/config.py). Variables you'll most likely need to set:

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
| `WEBUI__ENABLED` | `false` | Serve the bundled [web UI](web-ui.md) at `/webui` |

> **`MAX_CONCURRENT_TRANSCRIPTIONS` parallelizes across *distinct* models, not within one.** Each transcription pipeline holds a per-model lock during the transcribe step, so two requests for the **same** model still run one at a time even with the limit raised — the setting lets a request for a *different* model (and the align / diarize stages) proceed concurrently. To raise same-model throughput, add GPUs or run more replicas / workers. Whichever process runs inference (the API in direct mode, the worker in Kafka mode) logs this caveat at startup when the limit is >1.

> **Auth is off by default.** With neither `API_KEY` nor `API_KEYS_FILE` set, the server accepts every request without credentials and logs a startup warning. Set either to enforce auth (missing header → 401, invalid key → 403), or set `AUTH_REQUIRED=true` to turn an unconfigured deployment into a startup failure rather than an open server.

## Kafka mode

| Variable | Default | Description |
|---|---|---|
| `KAFKA__BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka broker address |
| `KAFKA__PROGRESS_TOPIC` | `transcription-progress` | Best-effort topic for per-stage worker progress events consumed by the status endpoint |
| `KAFKA__JOB_LEASE_TTL_SECONDS` | `300` | Processing-lease TTL; bounds crash-recovery takeover latency |
| `KAFKA__MAX_DELIVERY_ATTEMPTS` | `3` | Attempts before a job is routed to `transcription-dlq` |
| `STORAGE__BACKEND` | `s3` | Object storage backend: `s3` or `fs`. See [Storage Backends](storage.md) |
| `STORAGE__DELETE_AFTER_DOWNLOAD` | `true` | Delete the stored input once the job has a result. Never applies to `file_path` inputs |
| `MINIO_ROOT_USER` | `minioadmin` | S3 server root user — **change before deploying** |
| `MINIO_ROOT_PASSWORD` | `minioadmin` | S3 server root password — **change before deploying** |

> **`STORAGE` alone is not a variable.** A bare `STORAGE=fs` crashes startup — pydantic tries to JSON-decode it as the whole sub-model. Use `STORAGE__BACKEND=fs`. The same trap applies to bare `KAFKA`, `S3`, `WHISPER`, and `WEBUI`.

### S3 (storage backend `s3`)

| Variable | Default | Description |
|---|---|---|
| `S3__ENDPOINT_URL` | `http://localhost:9000` | S3 endpoint |
| `S3__BUCKET` | `whisperx-audio` | Bucket for audio uploads |
| `S3__REGION` | `us-east-1` | Region |
| `S3__ACCESS_KEY_ID` | `minioadmin` | Access key. The shipped compose stack sets this from `MINIO_ROOT_USER` |
| `S3__SECRET_ACCESS_KEY` | `minioadmin` | Secret key. The shipped compose stack sets this from `MINIO_ROOT_PASSWORD` |
| `S3__OBJECT_EXPIRY_DAYS` | `1` | Bucket lifecycle expiry, in days. `0` disables |
| `S3__MANAGE_LIFECYCLE` | `false` | Apply the lifecycle rule at startup |
| `S3__MULTIPART_PART_SIZE` | `8388608` | Part size (bytes) for multipart audio uploads. Clamped up to the 5 MiB S3 minimum, and scaled up for files that would exceed 10000 parts |
| `S3__MULTIPART_CONCURRENCY` | `4` | Parts uploaded in parallel. Peak buffer per upload ≈ part size × this |
| `S3__DELETE_AFTER_DOWNLOAD` | `true` | Deprecated alias for `STORAGE__DELETE_AFTER_DOWNLOAD`; still honoured |

### Shared filesystem (storage backend `fs`)

| Variable | Default | Description |
|---|---|---|
| `STORAGE__FS__ROOT` | *(required)* | Shared mount, visible at the **same path** in every API and worker container. The confinement boundary — nothing above it is ever read or written |
| `STORAGE__FS__PREFIX` | `whisperx` | Private subtree under the root that whisperx owns. Everything else under the root belongs to other services and is never written, deleted, or swept |
| `STORAGE__FS__DIR_MODE` | `770` | Octal directory mode, no `0o` prefix. Lets API and worker containers on different UIDs share a GID |
| `STORAGE__FS__FILE_MODE` | `660` | Octal file mode, no `0o` prefix |
| `STORAGE__FS__FSYNC` | `true` | fsync written objects before rename |
| `STORAGE__FS__RETENTION_DAYS` | `1` | Age after which the API's sweep removes audio and result envelopes. `0` disables |
| `STORAGE__FS__SWEEP_INTERVAL_SECONDS` | `3600` | How often the sweep runs. `0` disables |
| `STORAGE__FS__ALLOW_UNSAFE_MOUNT` | `false` | Override the startup refusal on filesystem types known to break atomic create-if-absent |

### Inputs written by other services (`file_path` jobs, Kafka only)

| Variable | Default | Description |
|---|---|---|
| `INPUT_FS__ENABLED` | `false` | Accept `file_path` on request-topic events. Off by default: this is an inbound trust surface |
| `INPUT_FS__ROOT` | *(inherits `STORAGE__FS__ROOT`)* | Confinement boundary for `file_path` inputs |
| `INPUT_FS__ALLOWED_DIRS` | *(empty = anywhere under the root)* | JSON list narrowing which directories under the root are accepted |

See [Sharing the mount with other services](storage.md#sharing-the-mount-with-other-services) for what these actually permit.

## Fetching audio by URL (both modes)

| Variable | Default | Description |
|---|---|---|
| `MAX_UPLOAD_SIZE_BYTES` | `0` | Hard upload/fetch size cap. `0` = unlimited |
| `URL_FETCH_TIMEOUT_SECONDS` | `300` | Total ceiling for an `audio_url` download |
| `URL_FETCH_CONNECT_TIMEOUT_SECONDS` | `15` | Connect-phase timeout |
| `URL_FETCH_ALLOW_PRIVATE_HOSTS` | `false` | Allow targets resolving to private / loopback / link-local IPs |
| `URL_FETCH_ALLOWED_HOSTS` | *(empty)* | JSON list of explicitly allowed hostnames |
| `WEBHOOK_TIMEOUT_SECONDS` | `15` | Total ceiling for a `callback_url` POST |

## Result store (direct mode)

Backs `GET /v1/audio/transcriptions/{id}/result` in direct mode by persisting finished results to disk, so they can be re-formatted without re-running the pipeline. Kafka mode uses the storage backend's `results/` prefix instead and ignores these.

| Variable | Default | Description |
|---|---|---|
| `RESULT_STORE__ENABLED` | `true` | Persist finished results for later fetch |
| `RESULT_STORE__DIR` | *(`<tempdir>/whisperx-results`)* | Where to write them. Best-effort: a temp dir, not shared across replicas |
| `RESULT_STORE__TTL_SECONDS` | `3600` | Age after which a stored result is dropped. `0` = no time-based expiry |
| `RESULT_STORE__MAX_ENTRIES` | `64` | Hard cap on retained results |

## Metrics (both modes)

Off by default. Requires the optional `metrics` extra (`pip install "whisperx-api-server[metrics]"`); the compose `*-observe` profiles set this up for you.

| Variable | Default | Description |
|---|---|---|
| `METRICS_ENABLED` | `false` | Master switch. When false no observability code loads and `/metrics` is not registered. `METRICS__ENABLED` also works |
| `METRICS__GPU_POLL_INTERVAL` | `15` | Seconds between GPU polls |
| `METRICS__WORKER_PORT` | `9091` | Port for the Kafka worker's own `/metrics` server. Give each scraped replica a unique value. Ignored in direct mode |

> In direct mode `METRICS_ENABLED=true` assumes `--workers 1` — the per-app collector registry is not shared across worker processes.

## Status endpoint (both modes)

| Variable | Default | Description |
|---|---|---|
| `REQUEST_STATUS__TTL_SECONDS` | `300` | How long terminal states (completed / failed) are retained for polling |
| `REQUEST_STATUS__MAX_ENTRIES` | `4096` | Hard cap on tracked requests; terminal entries are evicted first when over capacity |
| `REQUEST_STATUS__CLEANUP_INTERVAL_SECONDS` | `30` | How often the background sweep evicts expired entries |
| `REQUEST_STATUS__SSE_MAX_DURATION_SECONDS` | `3600` | Max lifetime of an `/events` stream |
