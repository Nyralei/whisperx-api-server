import { useEffect, useState } from "react";

import type { RequestStatus } from "../api/types";
import type { JobPhase } from "../hooks/useTranscriptionJob";
import { fmtDuration, prettyStageName } from "../lib/format";
import { Button, CopyButton } from "./ui";

function UploadBar(props: { fraction: number | null }) {
  const pct = props.fraction !== null ? Math.round(props.fraction * 100) : null;
  return (
    <div>
      <div className="mb-1 flex justify-between text-xs text-fg-muted">
        <span>uploading</span>
        <span className="tabular-nums">{pct !== null ? `${pct}%` : "…"}</span>
      </div>
      <div
        role="progressbar"
        aria-label="Upload progress"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={pct ?? undefined}
        className="h-2 overflow-hidden rounded-sm bg-fill"
      >
        <div
          className={`h-full bg-accent transition-[width] duration-300 ${
            pct === null ? "w-1/3 animate-pulse" : ""
          }`}
          style={pct !== null ? { width: `${pct}%` } : undefined}
        />
      </div>
    </div>
  );
}

function StageRow(props: {
  name: string;
  inProgress: boolean;
  duration: number | undefined;
}) {
  const { label, worker } = prettyStageName(props.name);
  return (
    <li className="flex items-center gap-2.5 py-1.5 text-sm">
      <span
        aria-hidden="true"
        className={`tabular-nums ${props.inProgress ? "text-accent-text" : "text-success"}`}
      >
        {props.inProgress ? "[··]" : "[ok]"}
      </span>
      <span className="text-fg-muted">{label}</span>
      {worker ? (
        <span className="rounded-sm bg-fill px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-fg-muted">
          worker
        </span>
      ) : null}
      <span className="ml-auto text-xs tabular-nums text-fg-subtle">
        {props.inProgress
          ? "running"
          : props.duration !== undefined
            ? fmtDuration(props.duration)
            : ""}
      </span>
    </li>
  );
}

function useElapsed(since: number | null): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(t);
  }, []);
  return since === null ? 0 : Math.max(0, (now - since) / 1000);
}

export function StagesPanel(props: {
  phase: JobPhase;
  status: RequestStatus | null;
  uploadProgress: number | null;
  requestId: string | null;
  fileName: string | null;
  startedAtMs: number | null;
  onCancel: () => void;
}) {
  const { phase, status } = props;
  const elapsed = useElapsed(props.startedAtMs);
  const running = phase === "uploading" || phase === "processing";

  const phaseLabel =
    phase === "uploading"
      ? "uploading"
      : status?.status === "queued"
        ? "queued"
        : "processing";

  return (
    <section
      aria-label="Job progress"
      className="rounded-sm border border-border bg-surface p-5"
    >
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <span className="relative flex h-2.5 w-2.5" aria-hidden="true">
          <span className="absolute inline-flex h-full w-full animate-ping bg-accent opacity-75" />
          <span className="relative inline-flex h-2.5 w-2.5 bg-accent" />
        </span>
        <p aria-live="polite" className="text-sm font-semibold text-fg">
          {phaseLabel}
          {props.fileName ? (
            <span className="font-normal text-fg-muted"> — {props.fileName}</span>
          ) : null}
        </p>
        <span className="ml-auto text-xs tabular-nums text-fg-subtle">
          {fmtDuration(elapsed)}
        </span>
        {running ? (
          <Button variant="secondary" onClick={props.onCancel}>
            Cancel
          </Button>
        ) : null}
      </div>

      {phase === "uploading" ? (
        <div className="mb-4">
          <UploadBar fraction={props.uploadProgress} />
        </div>
      ) : null}

      {status && status.stages.length > 0 ? (
        <ol className="divide-y divide-border">
          {status.stages.map((s) => (
            <StageRow
              key={s.name}
              name={s.name}
              inProgress={Boolean(s.in_progress)}
              duration={s.duration_seconds}
            />
          ))}
        </ol>
      ) : (
        <p className="text-sm text-fg-muted">
          {phase === "uploading"
            ? "the server registers the request once the upload completes."
            : "waiting for the first stage report…"}
        </p>
      )}

      {props.requestId ? (
        <p className="mt-4 flex items-center gap-1.5 border-t border-border pt-3 text-[11px] text-fg-subtle">
          request id: <code>{props.requestId}</code>
          <CopyButton value={props.requestId} />
        </p>
      ) : null}
    </section>
  );
}
