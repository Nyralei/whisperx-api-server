import { useState } from "react";

import { fetchStoredResult } from "../api/client";
import type { VerbosePayload } from "../api/types";
import { downloadFile, exportFileName, exportMime, planExport } from "../lib/exports";
import {
  RESPONSE_FORMATS,
  type ResponseFormat,
  type TranscriptionOptions,
} from "../lib/options";
import { useToasts } from "./Toast";
import { Button, Spinner } from "./ui";

function DownloadIcon() {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 24 24"
      className="h-3.5 w-3.5"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M12 3v12m0 0l-4-4m4 4l4-4M5 21h14" />
    </svg>
  );
}

export function ExportBar(props: {
  payload: VerbosePayload;
  requestId: string;
  file: File;
  options: TranscriptionOptions;
  subtitleAvailable: boolean;
  apiKey: string | null;
}) {
  const [busy, setBusy] = useState<ResponseFormat | null>(null);
  const { toast } = useToasts();

  const plan = (format: ResponseFormat) =>
    planExport(format, {
      payload: props.payload,
      align: props.options.align,
      subtitleAvailable: props.subtitleAvailable,
    });

  const doExport = async (format: ResponseFormat) => {
    const p = plan(format);
    const name = exportFileName(props.file.name, format);

    if (p.kind === "unavailable") {
      toast(p.reason, "error");
      return;
    }
    if (p.kind === "local") {
      downloadFile(name, p.content, p.mime);
      return;
    }

    setBusy(format);
    try {
      const { body } = await fetchStoredResult(
        props.requestId,
        format,
        props.options.highlightWords,
        props.apiKey,
      );
      downloadFile(name, body, exportMime(format));
      toast(`Downloaded ${name}`, "success");
    } catch (e) {
      toast(`Export failed: ${e instanceof Error ? e.message : String(e)}`, "error");
    } finally {
      setBusy(null);
    }
  };

  return (
    <fieldset>
      <legend className="mb-1.5 text-xs font-medium text-fg-muted">Download as</legend>
      <div className="flex flex-wrap items-center gap-2">
        {RESPONSE_FORMATS.map((format) => {
          const p = plan(format);
          return (
            <Button
              key={format}
              variant="secondary"
              disabled={busy !== null || p.kind === "unavailable"}
              title={p.kind === "unavailable" ? p.reason : `Download ${format}`}
              onClick={() => void doExport(format)}
            >
              {busy === format ? <Spinner /> : <DownloadIcon />}
              {format}
            </Button>
          );
        })}
      </div>
    </fieldset>
  );
}
