import { useEffect, useState } from "react";

import { getModelCatalog } from "./api/client";
import { ApiKeyGate } from "./components/ApiKeyGate";
import { DocsLinks } from "./components/DocsLinks";
import { ErrorPanel } from "./components/ErrorPanel";
import { FileDrop } from "./components/FileDrop";
import { OptionsPanel } from "./components/OptionsPanel";
import { ResultView } from "./components/ResultView";
import { StagesPanel } from "./components/StagesPanel";
import { ThemeToggle } from "./components/ThemeToggle";
import { useToasts } from "./components/Toast";
import { Button, Spinner } from "./components/ui";
import { useServerConnection } from "./hooks/useServerConnection";
import { useStoredOptions } from "./hooks/useStoredOptions";
import { useTranscriptionJob } from "./hooks/useTranscriptionJob";
import { useUnloadWarning } from "./hooks/useUnloadWarning";
import { useWindowDrop } from "./hooks/useWindowDrop";
import { validateFile } from "./lib/files";

function Logo() {
  return (
    <svg aria-hidden="true" viewBox="0 0 32 32" className="h-6 w-6">
      <rect width="32" height="32" rx="3" fill="#22d3ee" />
      <g stroke="#06232b" strokeWidth="2.5" strokeLinecap="round">
        <path d="M8 13v6" />
        <path d="M13 9v14" />
        <path d="M18 12v8" />
        <path d="M24 14v4" />
      </g>
    </svg>
  );
}

export default function App() {
  const connection = useServerConnection();
  const mode = connection.info?.mode ?? null;
  const job = useTranscriptionJob(connection.apiKey, mode);

  const [options, setOptions] = useStoredOptions();
  const [file, setFile] = useState<File | null>(null);
  const [modelSuggestions, setModelSuggestions] = useState<string[]>([]);
  const maxUploadBytes = connection.info?.max_upload_size_bytes ?? null;

  useEffect(() => {
    if (connection.phase !== "ready") return;
    let alive = true;
    void getModelCatalog(connection.apiKey).then((catalog) => {
      if (alive) setModelSuggestions([...(catalog.loaded ?? []), ...catalog.models]);
    });
    return () => {
      alive = false;
    };
  }, [connection.phase, connection.apiKey]);

  const running = job.state.phase === "uploading" || job.state.phase === "processing";
  const ready = connection.phase === "ready";
  const canStart = ready && file !== null && !running;

  const startJob = () => {
    if (file) job.start(file, options);
  };
  const retryJob = () => {
    if (job.state.file) job.start(job.state.file, options);
  };

  const { toast } = useToasts();
  const acceptFile = (f: File) => {
    const verdict = validateFile(f, maxUploadBytes);
    if (!verdict.ok) {
      toast(verdict.reason, "error");
      return;
    }
    if (job.state.phase === "completed" || job.state.phase === "failed") job.reset();
    setFile(f);
    if (verdict.warning) toast(verdict.warning, "info");
  };
  const externalDragging = useWindowDrop(ready && !running, acceptFile);

  // A reload/close can't restore the picked audio file, so warn before losing a
  // running or finished transcription.
  useUnloadWarning(running || job.state.phase === "completed");

  return (
    <div className="min-h-screen bg-canvas text-fg">
      <header className="border-b border-border bg-surface">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center gap-x-3 gap-y-2 px-4 py-3">
          <Logo />
          <h1 className="flex items-center text-lg font-semibold lowercase tracking-tight">
            whisperx
            <span className="term-cursor ml-1" aria-hidden="true" />
          </h1>
          {connection.info ? (
            <span className="rounded-sm bg-surface-2 px-2 py-0.5 text-[11px] tabular-nums text-fg-muted">
              v{connection.info.version} · {connection.info.mode} mode
            </span>
          ) : null}
          <div className="ml-auto flex items-center gap-2 text-xs text-fg-muted">
            {connection.phase === "connecting" ? (
              <>
                <Spinner className="text-fg-subtle" /> connecting
              </>
            ) : connection.phase === "ready" ? (
              <>
                <span className="h-2 w-2 bg-success" aria-hidden="true" />
                connected
                {connection.authRequired ? (
                  <button
                    type="button"
                    onClick={connection.clearKey}
                    className="text-accent-text underline-offset-2 hover:underline"
                  >
                    change key
                  </button>
                ) : null}
              </>
            ) : (
              <>
                <span className="h-2 w-2 bg-danger-text" aria-hidden="true" />
                {connection.phase === "need-key" ? "key required" : "offline"}
              </>
            )}
          </div>
          <DocsLinks />
          <ThemeToggle />
        </div>
      </header>

      {connection.phase === "unreachable" ? (
        <main className="mx-auto max-w-5xl px-4 py-8">
          <section
            role="alert"
            className="rounded-sm border border-danger-border bg-danger-bg p-5"
          >
            <h2 className="text-base font-semibold text-danger-strong">
              server unreachable
            </h2>
            <p className="mt-1 text-sm text-danger-text">
              Could not reach the API on this origin. Is the server running?
            </p>
            <Button variant="secondary" className="mt-3" onClick={connection.retry}>
              Retry
            </Button>
          </section>
        </main>
      ) : connection.phase === "need-key" ? (
        <main className="mx-auto max-w-5xl px-4 py-8">
          <ApiKeyGate connection={connection} />
        </main>
      ) : (
        <main className="mx-auto grid max-w-5xl gap-6 px-4 py-8 lg:grid-cols-[320px_minmax(0,1fr)]">
          <aside className="lg:sticky lg:top-6 lg:self-start">
            <div className="rounded-sm border border-border bg-surface">
              <h2 className="border-b border-border px-4 py-2.5 text-sm font-semibold text-fg">
                Options
              </h2>
              <div className="p-4">
                <OptionsPanel
                  options={options}
                  onChange={setOptions}
                  modelSuggestions={modelSuggestions}
                  disabled={running}
                />
              </div>
            </div>
          </aside>

          <div className="flex min-w-0 flex-col gap-6">
            {job.state.phase === "completed" &&
            job.state.result &&
            job.state.file &&
            job.state.requestId &&
            job.state.options ? (
              <ResultView
                payload={job.state.result}
                file={job.state.file}
                options={job.state.options}
                requestId={job.state.requestId}
                subtitleAvailable={connection.info?.subtitle_formats_available ?? true}
                apiKey={connection.apiKey}
                durationSeconds={
                  job.state.finishedAtMs !== null && job.state.startedAtMs !== null
                    ? (job.state.finishedAtMs - job.state.startedAtMs) / 1000
                    : null
                }
                optionsChanged={
                  JSON.stringify(options) !== JSON.stringify(job.state.options)
                }
                onRerun={retryJob}
                onReset={() => {
                  job.reset();
                  setFile(null);
                }}
              />
            ) : running ? (
              <StagesPanel
                phase={job.state.phase}
                status={job.state.status}
                uploadProgress={job.state.uploadProgress}
                requestId={job.state.requestId}
                fileName={job.state.file?.name ?? null}
                startedAtMs={job.state.startedAtMs}
                onCancel={job.cancel}
              />
            ) : job.state.phase === "failed" && job.state.error ? (
              <ErrorPanel
                error={job.state.error}
                status={job.state.status}
                requestId={job.state.requestId}
                onRetry={job.state.file ? retryJob : null}
                onReset={() => {
                  job.reset();
                  setFile(null);
                }}
              />
            ) : (
              <section aria-label="New transcription" className="flex flex-col gap-4">
                <div className="flex flex-col gap-1">
                  <h2 className="text-lg font-semibold tracking-tight text-fg">
                    transcribe audio &amp; video
                  </h2>
                  <p className="text-sm text-fg-muted">
                    drop a file, set options, run — OpenAI-compatible WhisperX on this
                    origin.
                  </p>
                </div>
                <FileDrop
                  file={file}
                  onSelect={setFile}
                  disabled={!ready}
                  maxUploadBytes={maxUploadBytes}
                />
                <div className="flex items-center gap-3">
                  <Button disabled={!canStart} onClick={startJob}>
                    Transcribe
                  </Button>
                  {!file ? (
                    <p className="text-sm text-fg-muted">select a file to get started.</p>
                  ) : null}
                </div>
              </section>
            )}
          </div>
        </main>
      )}

      <footer className="mx-auto max-w-5xl px-4 pb-8 text-[11px] text-fg-subtle">
        thin client over the OpenAI-compatible WhisperX API on this origin — no data
        leaves this server.
      </footer>

      {externalDragging ? (
        <div className="pointer-events-none fixed inset-0 z-40 flex items-center justify-center bg-canvas/80 backdrop-blur-sm">
          <div className="rounded-sm border-2 border-dashed border-accent bg-surface px-8 py-6 text-center">
            <p className="text-sm font-medium text-fg">drop to load</p>
          </div>
        </div>
      ) : null}
    </div>
  );
}
