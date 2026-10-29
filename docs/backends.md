# Pluggable Backends

Each pipeline stage (transcription, alignment, diarization) can use a different backend. Set the active backend via environment variables:

```bash
BACKENDS__TRANSCRIPTION=whisperx
BACKENDS__ALIGNMENT=whisperx
BACKENDS__DIARIZATION=whisperx
```

Only the `whisperx` backend ships by default. Custom backends can be registered via the backend registry at [`src/whisperx_api_server/backends/`](../src/whisperx_api_server/backends/).

When the transcript language has no default WhisperX alignment model and no configured override, word alignment is skipped with a warning. The response retains the original text and segment timestamps; diarization still assigns speakers to segments. This applies to explicit and automatically detected languages, in both direct and Kafka modes, including requests that enable only diarization. Word timestamps are unavailable for these segments.

Set `ALIGNMENT__MODELS` to a JSON mapping of language codes to compatible alignment models to enable word alignment for additional languages. The `multilingual` key continues to override the alignment model for every language. Model loading or inference failures for supported languages or configured overrides still fail the request.
