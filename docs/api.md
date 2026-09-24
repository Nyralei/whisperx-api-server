# API Reference

## `POST /v1/audio/transcriptions`

Transcribe an audio file. Compatible with the [OpenAI transcription API](https://platform.openai.com/docs/api-reference/audio/createTranscription).

**Form parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | — | Audio file. Provide exactly one of `file` or `audio_url` — neither or both is a `422`. |
| `audio_url` | string | — | Fetch the audio from this URL instead of uploading it. Validated against the SSRF policy (`URL_FETCH_ALLOW_PRIVATE_HOSTS` / `URL_FETCH_ALLOWED_HOSTS`) and subject to `MAX_UPLOAD_SIZE_BYTES`. |
| `model` | string | config default | Model name. `whisper-1` is aliased to the configured default. |
| `language` | string | config default | ISO-639-1 language code. Auto-detected if omitted. |
| `prompt` | string | — | Optional context/hotwords hint |
| `response_format` | string | `json` | `text`, `json`, `verbose_json`, `vtt_json`, `srt`, `vtt`, `aud` |
| `temperature` | float | `0.0` | Sampling temperature |
| `timestamp_granularities[]` | list | `["segment"]` | `segment`, `word` |
| `align` | bool | `true` | Enable word-level alignment (required for subtitle formats) |
| `diarize` | bool | `false` | Enable speaker diarization (requires `align=true`) |
| `speaker_embeddings` | bool | `false` | Include speaker embeddings in diarization output |
| `min_speakers` | int | — | Lower bound on the speaker count for diarization (≥ 1) |
| `max_speakers` | int | — | Upper bound on the speaker count for diarization (≥ `min_speakers`) |
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

## `POST /v1/audio/translations`

Translate audio to English. Same parameters as `/v1/audio/transcriptions`, minus `language`, `align`, `diarize`, and diarization-related fields.

---

## `GET /v1/audio/transcriptions/{request_id}/status`

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
| kafka (API-side) | `uploading_audio`, `submitted_to_kafka`, `awaiting_worker` |
| kafka (worker-side, via `transcription-progress` topic) | `worker.audio_download` (or `worker.url_download` / `worker.file_copy`, depending on the input mode), `worker.audio_load`, `worker.awaiting_gpu`, `worker.transcribe`, `worker.align`, `worker.diarize`, `worker.finalize` |

> **Renamed:** `uploading_to_s3` → `uploading_audio` and `worker.s3_download` → `worker.audio_download`, since neither is S3-specific any more. The `s3_download` key in the reply's `profile` block is now `audio_download`. Update any dashboard keyed on the old names.

Failures (invalid audio, queue full, timeout, worker error, …) end the lifecycle with `status="failed"` and populate `error` / `error_type`. Stages completed before the failure are preserved.

Terminal states (`completed` / `failed`) are retained for `REQUEST_STATUS__TTL_SECONDS` (default 300s) so polling clients that arrive just after the POST returns can still confirm the outcome. After that, the id 404s.

**Other responses**

- `400 Bad Request` — malformed `request_id` (must match `[A-Za-z0-9._-]{1,128}`)
- `404 Not Found` — id is unknown, expired, or not yet seen by this replica

> Both the request/reply path and `/status` scale across API replicas in Kafka mode. Each reply is delivered to every replica and only the one holding the job resolves it, so any replica can serve the POST. Status converges too: the submitting replica announces the job on the progress topic and every replica consumes the progress stream, so `GET /status` works on any replica behind a load balancer — no sticky sessions required. Each replica's tracker therefore holds entries for jobs across all replicas (bounded by `REQUEST_STATUS__MAX_ENTRIES`, default 4096, well above `KAFKA__MAX_PENDING_JOBS`). In-flight status is still in-memory: a replica restart loses the live history for jobs it learned about (they would expire within minutes anyway), while completed/failed outcomes stay readable for `REQUEST_STATUS__TTL_SECONDS`. Direct mode is single-process and unaffected.

---

## `GET /v1/audio/transcriptions/{request_id}/events`

Streams the processing status of a transcription request as Server-Sent Events. Each state change is emitted as a `data:` line carrying the status JSON — `status` (`queued` / `in_progress` / `completed` / `failed`), the active `stage`, the per-stage `stages[]` timeline, and `error` / `error_type` on failure. `: ping` comment lines are sent while idle to hold the connection open. The stream ends when the request reaches a terminal state, the client disconnects, or `REQUEST_STATUS__SSE_MAX_DURATION_SECONDS` elapses.

The client sets a known `X-Request-ID` (matching `[A-Za-z0-9._-]{1,128}`) on the transcription POST and opens the stream alongside it:

```bash
curl -N http://localhost:8000/v1/audio/transcriptions/my-request-1/events
```

When auth is configured the endpoint requires the API key like every other route, so consume it with a header-capable client rather than a browser `EventSource`. A malformed id returns `400 Bad Request`; an unknown, expired, or not-yet-tracked id returns `404 Not Found`.

---

## Async job submission (Kafka mode)

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

## `GET /v1/audio/transcriptions/{request_id}/result`

Fetch and (re-)format the final result of a finished transcription without re-running the pipeline. In **Kafka mode** it reads the stored `results/{job_id}` envelope from object storage, so it works from any replica and after restarts. In **direct mode** it reads an on-disk result store written when the synchronous POST completes — enabled by default (`RESULT_STORE__ENABLED`), bounded by count and age, and best-effort (a temp dir, not shared across replicas or guaranteed across restarts).

**Query parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `response_format` | string | config default | Same formats as the POST: `text`, `json`, `verbose_json`, `vtt_json`, `srt`, `vtt`, `aud` |
| `highlight_words` | bool | `false` | Highlight words in `vtt`/`srt` output |

Formatting is applied at fetch time from these query parameters — the `response_format` set on the async POST is not stored, so you choose the format (and may request several) when you fetch.

**Responses**

- `200 OK` — formatted transcription (Content-Type per `response_format`), identical to what the synchronous POST would have returned
- `400 Bad Request` — malformed `request_id`
- `404 Not Found` — result not available (still pending, unknown, expired, or — in direct mode — never stored / result store disabled)
- worker failures are replayed with the same status code the synchronous endpoint returns (e.g. invalid audio → `422`, timeout → `504`), not `404`

Kafka-mode result retention depends on the storage backend — see [Retention](storage.md#retention). Direct-mode results are kept for `RESULT_STORE__TTL_SECONDS` (default 1h), up to `RESULT_STORE__MAX_ENTRIES` files. After that the id 404s.

---

## Completion webhook

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

## `GET /info`

Returns the running version, mode, uptime, concurrency / queue state, and (in Kafka mode) discovered worker membership. Also reports `max_upload_size_bytes` (`null` = unlimited) and `subtitle_formats_available` (whether this process can render `vtt`/`srt`/`aud`/`vtt_json`), so a client can pre-validate uploads and format support. Add `?detail=full` for extended Kafka topology.

---

## `GET /healthcheck`

Returns `{"status": "healthy"}`. Not protected by API key auth.

---

## Model management

**Direct mode only, apart from the catalog.** In Kafka mode the API loads no models — model lifecycle lives in the worker processes — so every endpoint below except `/models/catalog` is not registered and returns `404`. The API logs which prefixes it disabled at startup.

| Endpoint | Description |
|---|---|
| `GET /models/catalog` | List known transcription model names + configured default (both modes) |
| `GET /models/list` | List loaded transcription models |
| `POST /models/load` | Load a model (`model` param) |
| `POST /models/unload` | Unload a model (`model` param) |
| `GET /align_models/list` | List loaded alignment models |
| `POST /align_models/load` | Load an alignment model (`language` param) |
| `POST /align_models/unload` | Unload an alignment model (`language` param) |
| `GET /diarize_models/list` | List loaded diarization models |
| `POST /diarize_models/load` | Load a diarization model (`model` param) |
| `POST /diarize_models/unload` | Unload a diarization model (`model` param) |
