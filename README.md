# WhisperX API Server

A FastAPI server that exposes [WhisperX](https://github.com/m-bain/WhisperX) as an OpenAI-compatible audio transcription API. Supports both a simple single-server mode and a horizontally scalable distributed mode backed by Kafka and object storage.

## Features

- **OpenAI-compatible** — drop-in replacement for `/v1/audio/transcriptions` and `/v1/audio/translations`
- **Alignment & diarization** — word-level timestamps and speaker labels out of the box
- **Multiple output formats** — `json`, `verbose_json`, `vtt_json`, `srt`, `vtt`, `aud`, `text`
- **Distributed mode** — offload GPU work to dedicated workers via Kafka, backed by S3 **or** a shared filesystem
- **Live request status** — poll an in-flight transcription's current pipeline stage by request id, in both direct and Kafka modes
- **Pluggable backends** — swap transcription, alignment, and diarization implementations per stage
- **API key auth** — single key or a JSON key-map for multi-client setups
- **Optional web UI** — a built-in browser client (upload, live stage progress, interactive transcript, exports), off by default

## Documentation

| Page | Contents |
|---|---|
| [Configuration](docs/configuration.md) | The environment variables you're likely to set, grouped by concern |
| [API Reference](docs/api.md) | Endpoints, form parameters, status/result/events, webhooks |
| [Storage Backends](docs/storage.md) | `s3` vs `fs`, the job lease, sharing a mount with other services |
| [Deployment](docs/deployment.md) | Compose profiles, worker lifecycle, delivery semantics, pip install |
| [Pluggable Backends](docs/backends.md) | Per-stage transcription / alignment / diarization backends |
| [Web UI](docs/web-ui.md) | The optional bundled browser client |

## Quick Start

Each profile selects exactly one combination — service set, image tags, and `--extra` build args are all driven by the profile so images stay minimal.

### Standalone (single server)

```bash
docker compose --profile cuda up           # → whisperx-api:cuda
docker compose --profile cpu up            # → whisperx-api:cpu
docker compose --profile cuda-observe up   # → whisperx-api:cuda-metrics  (+ Prometheus)
docker compose --profile cpu-observe up    # → whisperx-api:cpu-metrics   (+ Prometheus)
```

The API is available at `http://localhost:8000`. With an `*-observe` profile, Prometheus is on `:9090` (point your own Grafana / dashboard at it).

### Distributed mode (Kafka + workers)

```bash
cp .env.example .env    # edit credentials before first run
docker compose -f compose-kafka.yaml --profile cuda up
```

See [Deployment](docs/deployment.md) for the full profile matrix, worker lifecycle, and delivery guarantees.

### Transcribe something

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.mp3 -F model=large-v3 -F align=true
```

## Contributing

Issues, forks, and pull requests are welcome.

## License

GNU General Public License v3.0 — see [`LICENSE`](LICENSE) for details.
