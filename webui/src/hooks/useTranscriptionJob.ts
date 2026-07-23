/**
 * One transcription job = a synchronous POST plus a concurrent status feed.
 *
 * The POST only returns when the whole pipeline finishes, so the UI generates
 * the request id itself, sends it as X-Request-ID, and follows
 * GET /v1/audio/transcriptions/{id}/events (SSE, falling back to polling
 * /status) to animate the stage timeline while the POST is in flight.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import {
  ApiError,
  fetchStoredResult,
  getStatus,
  isAbort,
  isNetworkError,
  newRequestId,
  postTranscription,
  streamStatusEvents,
} from "../api/client";
import type { RequestStatus, VerbosePayload } from "../api/types";
import { buildFormFields, type TranscriptionOptions, wireFormat } from "../lib/options";

const POLL_INTERVAL_MS = 1000;
const STATUS_STREAM_DELAY_MS = 800;

export type JobPhase = "idle" | "uploading" | "processing" | "completed" | "failed";

export interface JobError {
  message: string;
  errorType: string | null;
  cancelled: boolean;
}

export interface JobState {
  phase: JobPhase;
  requestId: string | null;
  file: File | null;
  /** Snapshot of the options the run was submitted with. */
  options: TranscriptionOptions | null;
  uploadProgress: number | null;
  status: RequestStatus | null;
  result: VerbosePayload | null;
  error: JobError | null;
  startedAtMs: number | null;
  /** Set when the job completes; total wall-clock is finished−started. */
  finishedAtMs: number | null;
}

const IDLE: JobState = {
  phase: "idle",
  requestId: null,
  file: null,
  options: null,
  uploadProgress: null,
  status: null,
  result: null,
  error: null,
  startedAtMs: null,
  finishedAtMs: null,
};

export interface TranscriptionJob {
  state: JobState;
  start: (file: File, options: TranscriptionOptions) => void;
  cancel: () => void;
  reset: () => void;
}

export function useTranscriptionJob(
  apiKey: string | null,
  mode: string | null,
): TranscriptionJob {
  const [state, setState] = useState<JobState>(IDLE);
  const abortRef = useRef<(() => void) | null>(null);
  const pollTimerRef = useRef<number | null>(null);
  const statusTimerRef = useRef<number | null>(null);
  const statusAbortRef = useRef<AbortController | null>(null);
  const runSeqRef = useRef(0);

  const stopStatusFeed = useCallback(() => {
    if (pollTimerRef.current !== null) {
      window.clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
    if (statusTimerRef.current !== null) {
      window.clearTimeout(statusTimerRef.current);
      statusTimerRef.current = null;
    }
    statusAbortRef.current?.abort();
    statusAbortRef.current = null;
  }, []);

  useEffect(
    () => () => {
      stopStatusFeed();
      abortRef.current?.();
    },
    [stopStatusFeed],
  );

  const start = useCallback(
    (file: File, options: TranscriptionOptions) => {
      const seq = ++runSeqRef.current;
      const live = () => runSeqRef.current === seq;
      const requestId = newRequestId();
      const format = wireFormat(options, mode);
      const jobMode = mode;
      const jobKey = apiKey;

      stopStatusFeed();
      abortRef.current?.();

      setState({
        ...IDLE,
        phase: "uploading",
        requestId,
        file,
        options,
        uploadProgress: 0,
        startedAtMs: Date.now(),
      });

      const statusAbort = new AbortController();
      statusAbortRef.current = statusAbort;

      const applyStatus = (status: RequestStatus) => {
        // The tracker sees the request before the pipeline finishes, so a
        // terminal status is only the imminent-outcome hint; keep the phase and
        // let the POST settle it (its error detail is richer).
        setState((s) =>
          s.phase === "completed" || s.phase === "failed" ? s : { ...s, status },
        );
      };

      let pollBusy = false;
      const poll = async () => {
        if (pollBusy || !live()) return;
        pollBusy = true;
        try {
          const status = await getStatus(requestId, jobKey);
          if (live() && status !== null) applyStatus(status);
        } finally {
          pollBusy = false;
        }
      };

      const beginPolling = () => {
        if (pollTimerRef.current !== null || !live()) return;
        pollTimerRef.current = window.setInterval(() => void poll(), POLL_INTERVAL_MS);
      };

      const beginStatusFeed = () => {
        if (!live()) return;
        statusTimerRef.current = window.setTimeout(() => {
          if (!live() || statusAbort.signal.aborted) return;
          void streamStatusEvents(requestId, jobKey, {
            onStatus: (status) => {
              if (live()) applyStatus(status);
            },
            signal: statusAbort.signal,
          }).catch(() => {
            if (live() && !statusAbort.signal.aborted) beginPolling();
          });
        }, STATUS_STREAM_DELAY_MS);
      };

      const handle = postTranscription({
        file,
        fields: buildFormFields(options, format),
        requestId,
        apiKey: jobKey,
        onUploadProgress: (fraction) => {
          if (!live()) return;
          setState((s) =>
            s.phase === "uploading" ? { ...s, uploadProgress: fraction } : s,
          );
        },
        onUploadComplete: () => {
          if (!live()) return;
          setState((s) =>
            s.phase === "uploading"
              ? { ...s, phase: "processing", uploadProgress: 1 }
              : s,
          );
          beginStatusFeed();
        },
      });
      abortRef.current = handle.abort;

      const finalStatus = async (): Promise<RequestStatus | null> =>
        getStatus(requestId, jobKey);

      handle.promise.then(
        async ({ body }) => {
          if (!live()) return;
          stopStatusFeed();
          let result: VerbosePayload;
          try {
            result = JSON.parse(body) as VerbosePayload;
          } catch {
            setState((s) => ({
              ...s,
              phase: "failed",
              error: {
                message: "The server returned an unparseable response.",
                errorType: "bad_response",
                cancelled: false,
              },
            }));
            return;
          }
          const status = await finalStatus();
          if (!live()) return;
          setState((s) => ({
            ...s,
            phase: "completed",
            result,
            status: status ?? s.status,
            finishedAtMs: Date.now(),
          }));
        },
        async (err: unknown) => {
          if (!live()) return;

          if (isAbort(err)) {
            stopStatusFeed();
            setState((s) => ({
              ...s,
              phase: "failed",
              error: {
                message:
                  jobMode === "kafka"
                    ? "Cancelled. The server may still finish this job in the background."
                    : "Cancelled.",
                errorType: "cancelled",
                cancelled: true,
              },
            }));
            return;
          }

          // Connection lost mid-job. In Kafka mode the job keeps running
          // server-side, so keep polling and recover the result from durable
          // storage once the status turns terminal.
          if (isNetworkError(err) && jobMode === "kafka") {
            stopStatusFeed();
            setState((s) => ({
              ...s,
              error: {
                message: "Connection lost — waiting for the job via status polling…",
                errorType: "connection_lost",
                cancelled: false,
              },
            }));
            const recovered = await recoverViaStatus(requestId, jobKey, live);
            if (!live()) return;
            stopStatusFeed();
            setState((s) => applyRecovery(s, recovered));
            return;
          }

          stopStatusFeed();
          const status = await finalStatus();
          if (!live()) return;
          const apiErr = err instanceof ApiError ? err : null;
          setState((s) => ({
            ...s,
            phase: "failed",
            status: status ?? s.status,
            error: {
              message:
                apiErr?.message ??
                (err instanceof Error ? err.message : "Request failed."),
              errorType:
                apiErr?.errorType ??
                status?.error_type ??
                (apiErr ? `http_${apiErr.status}` : null),
              cancelled: false,
            },
          }));
        },
      );
    },
    [apiKey, mode, stopStatusFeed],
  );

  const cancel = useCallback(() => {
    abortRef.current?.();
  }, []);

  const reset = useCallback(() => {
    runSeqRef.current++;
    stopStatusFeed();
    abortRef.current?.();
    abortRef.current = null;
    setState(IDLE);
  }, [stopStatusFeed]);

  return { state, start, cancel, reset };
}

type Recovery =
  | { outcome: "completed"; result: VerbosePayload; status: RequestStatus | null }
  | {
      outcome: "failed";
      message: string;
      errorType: string | null;
      status: RequestStatus | null;
    };

/** Poll until terminal, then pull the stored result (Kafka mode only). */
async function recoverViaStatus(
  requestId: string,
  apiKey: string | null,
  live: () => boolean,
): Promise<Recovery> {
  const deadline = Date.now() + 24 * 60 * 60 * 1000;
  let status: RequestStatus | null = null;

  while (live() && Date.now() < deadline) {
    status = await getStatus(requestId, apiKey);
    if (status?.status === "completed") {
      try {
        const { body } = await fetchStoredResult(
          requestId,
          "verbose_json",
          false,
          apiKey,
        );
        return {
          outcome: "completed",
          result: JSON.parse(body) as VerbosePayload,
          status,
        };
      } catch (e) {
        return {
          outcome: "failed",
          message: `Job completed but the stored result could not be fetched: ${
            e instanceof Error ? e.message : String(e)
          }`,
          errorType: "result_fetch_failed",
          status,
        };
      }
    }
    if (status?.status === "failed") {
      return {
        outcome: "failed",
        message: status.error ?? "The job failed.",
        errorType: status.error_type,
        status,
      };
    }
    await new Promise((r) => setTimeout(r, 2000));
  }
  return {
    outcome: "failed",
    message: "Connection lost and the job outcome could not be recovered.",
    errorType: "connection_lost",
    status,
  };
}

function applyRecovery(s: JobState, recovered: Recovery): JobState {
  if (recovered.outcome === "completed") {
    return {
      ...s,
      phase: "completed",
      result: recovered.result,
      status: recovered.status ?? s.status,
      error: null,
      finishedAtMs: Date.now(),
    };
  }
  return {
    ...s,
    phase: "failed",
    status: recovered.status ?? s.status,
    error: {
      message: recovered.message,
      errorType: recovered.errorType,
      cancelled: false,
    },
  };
}
