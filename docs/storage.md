# Storage Backends

Kafka mode stores three things: the input audio, the terminal result envelope (so a redelivered job resends instead of re-running), and a `claims/<job_id>` processing lease. Direct mode uses none of this.

```bash
STORAGE__BACKEND=s3   # default — S3-compatible object storage
STORAGE__BACKEND=fs   # a shared POSIX mount, no object store required
```

Custom backends can be registered via the storage registry at [`src/whisperx_api_server/storage/`](../src/whisperx_api_server/storage/). See [Configuration](configuration.md#kafka-mode) for the full variable list.

## Atomic create-if-absent is a hard requirement

The lease is what stops two workers transcribing the same job after a rebalance redelivers an uncommitted message. It needs an atomic create-if-absent — conditional `PUT` with `If-None-Match` on S3, `link()` on a filesystem.

**A backend that cannot prove that primitive refuses to start in Kafka mode.** Both the API and the worker exit non-zero before readiness is set, and the worker's `/ready` reports `storage_initialized` as unmet on the way down. There is no degraded mode.

For `STORAGE__BACKEND=s3` that means Silo, MinIO `RELEASE.2024-08-*` or later, or an S3 provider with conditional writes.

For `STORAGE__BACKEND=fs`, startup enforces the mount is suitable in two steps: it reads the filesystem type from `/proc/self/mountinfo` before any I/O, then runs a real `link()` probe.

| Mount | Result |
|---|---|
| NFSv3+/NFSv4, SMB3, GlusterFS, CephFS, Lustre, GFS2, OCFS2 | Supported |
| Any local filesystem (ext4, XFS, Btrfs, ZFS, overlay, tmpfs, a Docker named volume) shared between containers on **one** host | Supported |
| FUSE object-storage gateways — s3fs, gcsfuse, rclone mount, blobfuse, JuiceFS without a metadata engine | **Refused at startup**, by name and by probe |
| NFSv2 | **Refused at startup** |
| Anything else | Warns, then relies on the `link()` probe |

`STORAGE__FS__ALLOW_UNSAFE_MOUNT=true` downgrades a refusal to a warning. The one residual gap is a filesystem that implements `link()` without cross-host coherency; no such filesystem is known.

The mount-type check is Linux-only and is skipped elsewhere (it logs that it did), leaving the probe as the sole gate.

## The mount must be writable by the container's user

**This is on you, whatever root you pick.** The shipped images run as uid/gid `1000:1000` (`ubuntu`), and startup refuses with an actionable message if it cannot create `<root>/<prefix>`.

The trap is that an empty mount usually arrives owned by `root:root`:

| Mount | What you have to do |
|---|---|
| Docker **named volume** | Docker seeds a fresh volume from the image directory at that path, *including its ownership*. The provided images pre-create `/home/ubuntu/shared` as `ubuntu`, which is why the shipped compose demo works. **Mount a named volume anywhere else — `/mnt`, `/data` — and it is created `root:root` and startup fails.** Pre-create that path in your own image layer, or use one of the rows below. |
| **Bind mount** | The host directory's ownership applies as-is. `chown 1000:1000 /host/path` before starting, or run the containers with a matching `user:`. |
| **NFS** | Export it so uid 1000 can write — no `root_squash` surprises. `STORAGE__FS__DIR_MODE`/`FILE_MODE` (default `770`/`660`) let API and worker interoperate on a shared GID with different UIDs. |
| **SMB/CIFS** | Mount with `uid=1000,gid=1000`. |

Ownership fixes must be applied to the *mount*, not to `<root>/<prefix>` — whisperx creates the prefix subtree itself on first start.

## Sharing the mount with other services

`STORAGE__FS__ROOT` is the **security boundary** — no verb can read or write above it. `STORAGE__FS__PREFIX` is the **private subtree** whisperx owns:

```
/mnt                    STORAGE__FS__ROOT   — nothing above this is reachable
├── files/wav/123.wav   another service owns this; whisperx reads it, never writes or deletes
└── whisperx/           STORAGE__FS__PREFIX — whisperx's own subtree
    ├── audio/  results/  claims/
```

whisperx never writes, deletes, or sweeps anything outside `<root>/<prefix>`, so a producer's files are safe on the same mount. The retention sweep runs in the API process, rooted at the prefix.

To have another service drive a job from a file it wrote itself, set `INPUT_FS__ENABLED=true` and produce an event carrying `file_path` instead of `s3_key`:

```json
{"job_id": "abc123", "file_path": "/mnt/files/wav/123.wav", "filename": "123.wav", "params": {}}
```

Events must carry **exactly one** of `s3_key`, `audio_url`, or `file_path`.

Rules for `file_path`, all enforced:

- The resolved path (after following symlinks) must sit under the root. `/etc/passwd`, `../..` traversal, and a symlink inside the root pointing out of it are all rejected.
- Paths under `<root>/<prefix>/` are rejected — an external producer cannot hand the worker a lease or a result envelope and have it treated as audio.
- Only regular files. A FIFO would block the worker forever on open; devices, sockets, and directories are not audio.
- `MAX_UPLOAD_SIZE_BYTES` applies.
- **The source file is never deleted**, regardless of `STORAGE__DELETE_AFTER_DOWNLOAD`. That setting only ever applies to inputs whisperx stored itself. The worker copies the file to a temp path and works on the copy.

`file_path` is **Kafka-only by design** — there is no HTTP equivalent, because a local-path parameter on a public endpoint is a local-file-inclusion primitive. Semi-trusted internal producers on a shared broker are a different threat model from arbitrary HTTP clients.

## Untrusted keys on the wire

`s3_key` arrives on the request topic and is fed to a read and a delete. It is confined to the `audio/<job_id>/<filename>` shape before it reaches any backend, so a producer cannot name `results/<other-job>` or `claims/<other-job>` and delete another job's envelope or release another worker's live lease. On the filesystem backend a second, independent layer rejects absolute paths, traversal segments, and symlinks that escape the root.

## Upload memory

Audio larger than one part goes to S3 as a multipart upload rather than a single `PutObject`, which would hold the whole body in RAM and — on a plain-HTTP endpoint, where SigV4 cannot skip payload signing — hash all of it inline on the event loop, stalling health probes and expiring Kafka consumer sessions on a large file. Peak buffered bytes per upload are roughly `S3__MULTIPART_PART_SIZE × S3__MULTIPART_CONCURRENCY` (default 8 MiB × 4), independent of file size. A failed part aborts the upload, so partial parts are not left behind as billable storage.

The filesystem backend streams to a temp file and renames, so its peak is one chunk regardless.

## Retention

S3 has bucket lifecycle rules (`S3__OBJECT_EXPIRY_DAYS` with `S3__MANAGE_LIFECYCLE=true`).

The filesystem backend has none, so the API runs a sweep every `STORAGE__FS__SWEEP_INTERVAL_SECONDS`, removing audio and result envelopes older than `STORAGE__FS__RETENTION_DAYS`. Claims are swept by reading each lease's `expires_at`, never by file age — deleting a live lease would reintroduce the concurrent-duplicate run it exists to prevent.

Flat `results/` and `claims/` directories are fine into the tens of thousands of entries; past roughly 100k retained entries a `readdir` on a network filesystem starts to drag, so lower `STORAGE__FS__RETENTION_DAYS`.
