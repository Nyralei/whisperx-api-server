import { useEffect, useRef } from "react";

import type { RequestStatus } from "../api/types";
import type { JobError } from "../hooks/useTranscriptionJob";
import { prettyStageName } from "../lib/format";
import { Button, CopyButton } from "./ui";

export function ErrorPanel(props: {
  error: JobError;
  status: RequestStatus | null;
  requestId: string | null;
  onRetry: (() => void) | null;
  onReset: () => void;
}) {
  const { error, status } = props;
  const cancelled = error.cancelled;
  // Prefer the tracker's error detail when the HTTP layer only had a generic one.
  const message =
    !cancelled && status?.error && error.message.startsWith("HTTP ")
      ? status.error
      : error.message;
  const failedStage =
    status && status.stages.length > 0
      ? prettyStageName(status.stages[status.stages.length - 1].name).label
      : null;

  const headingRef = useRef<HTMLHeadingElement>(null);
  useEffect(() => {
    headingRef.current?.focus();
  }, []);

  return (
    <section
      role="alert"
      className={`rounded-sm border p-5 ${
        cancelled ? "border-border bg-surface" : "border-danger-border bg-danger-bg"
      }`}
    >
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <h2
          ref={headingRef}
          tabIndex={-1}
          className={`text-base font-semibold focus:outline-none ${
            cancelled ? "text-fg" : "text-danger-strong"
          }`}
        >
          {cancelled ? "Cancelled" : "Transcription failed"}
        </h2>
        {error.errorType && !cancelled ? (
          <span className="rounded bg-danger-bg px-1.5 py-0.5 text-[11px] text-danger-strong">
            {error.errorType}
          </span>
        ) : null}
      </div>
      <p className={`text-sm ${cancelled ? "text-fg-muted" : "text-danger-text"}`}>
        {message}
      </p>
      {failedStage && !cancelled ? (
        <p className="mt-1 text-xs text-danger-text">Last stage: {failedStage}</p>
      ) : null}
      {props.requestId ? (
        <p className="mt-2 flex items-center gap-1.5 text-[11px] text-fg-subtle">
          Request ID: <code>{props.requestId}</code>
          <CopyButton value={props.requestId} />
        </p>
      ) : null}
      <div className="mt-4 flex gap-2">
        {props.onRetry ? <Button onClick={props.onRetry}>Try again</Button> : null}
        <Button variant="secondary" onClick={props.onReset}>
          Start over
        </Button>
      </div>
    </section>
  );
}
