# Deployment

## Profile matrix

Every runtime variant is gated by exactly one profile, so `docker compose up` never accidentally starts a GPU process on a machine that doesn't have one, and observability stacks never spawn duplicate API servers.

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

| File | Purpose |
|---|---|
| `compose.yaml` | Standalone server — profiles: `cuda`, `cpu`, `cuda-observe`, `cpu-observe` |
| `compose-kafka.yaml` | Distributed stack (API + Kafka + S3 + workers) — same four profiles |

## Running the distributed stack

```bash
# Copy and edit credentials before first run
cp .env.example .env

docker compose -f compose-kafka.yaml --profile cuda up          # CUDA worker
docker compose -f compose-kafka.yaml --profile cpu up           # CPU worker
docker compose -f compose-kafka.yaml --profile cuda-observe up  # + Prometheus
docker compose -f compose-kafka.yaml --profile cpu-observe up   # + Prometheus
```

In Kafka mode the API is a CPU-only router (no model load) regardless of which worker compute backend is selected.

### Trying the filesystem storage backend

`compose-kafka.yaml` mounts a named volume at `/home/ubuntu/shared` in both the API and worker containers, so no extra profile is needed:

```bash
STORAGE__BACKEND=fs docker compose -f compose-kafka.yaml --profile cpu up
```

A named volume *is* a shared mount across containers on one host, which is enough for a demo but is not multi-host — for that, mount the same NFS/SMB export at the same path in every container. `depends_on` cannot be made conditional, so the fs demo still starts an idle S3 service.

## Worker lifecycle

Workers process one job at a time per container. Scale horizontally by running multiple worker replicas.

On `SIGTERM` (stop / rolling update) a worker finishes its in-flight job — reply, then offset commit — before exiting, and refuses to start a new one. Set the container `stop_grace_period` to your worst-case job duration so a long job isn't killed mid-flight; a job killed before commit is safely redelivered (idempotent resend) at the cost of one reprocess.

## Delivery semantics

Delivery is at-least-once. A worker writes each job's result envelope to storage (`results/{job_id}`) before replying, so a redelivered job resends the stored reply instead of re-running.

Each job is guarded by a processing lease (`claims/{job_id}`, TTL `KAFKA__JOB_LEASE_TTL_SECONDS`, default 300s): a copy redelivered mid-run — for example after a consumer-group rebalance — defers instead of starting a concurrent duplicate, and takes over only once the lease expires. The lease also counts delivery attempts: a job that repeatedly kills its worker is, past `KAFKA__MAX_DELIVERY_ATTEMPTS` (default 3), routed to the `transcription-dlq` topic so the submitter fails fast instead of every worker dying on it.

The lease depends on the storage backend providing an atomic create-if-absent, which is verified at startup — see [Storage Backends](storage.md#atomic-create-if-absent-is-a-hard-requirement).

Retention of `results/` and `claims/` depends on the backend: bucket lifecycle on S3, a sweep in the API process on a filesystem. See [Retention](storage.md#retention).

## Local installation (pip / uv)

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
