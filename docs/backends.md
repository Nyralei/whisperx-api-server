# Pluggable Backends

Each pipeline stage (transcription, alignment, diarization) can use a different backend. Set the active backend via environment variables:

```bash
BACKENDS__TRANSCRIPTION=whisperx
BACKENDS__ALIGNMENT=whisperx
BACKENDS__DIARIZATION=whisperx
```

Only the `whisperx` backend ships by default. Custom backends can be registered via the backend registry at [`src/whisperx_api_server/backends/`](../src/whisperx_api_server/backends/).
