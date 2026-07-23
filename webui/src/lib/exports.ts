/**
 * Export strategy per format: text / json / verbose_json (and vtt / vtt_json when
 * the run produced vtt_text) are derived locally from the payload in hand;
 * anything else is re-formatted server-side via GET .../result.
 */

import type { VerbosePayload } from "../api/types";
import type { ResponseFormat } from "./options";

export type ExportPlan =
  | { kind: "local"; content: string; mime: string }
  | { kind: "stored" }
  | { kind: "unavailable"; reason: string };

const MIME: Record<ResponseFormat, string> = {
  text: "text/plain",
  json: "application/json",
  verbose_json: "application/json",
  vtt_json: "application/json",
  srt: "text/plain",
  vtt: "text/vtt",
  aud: "text/plain",
};

export const EXPORT_EXTENSION: Record<ResponseFormat, string> = {
  text: "txt",
  json: "json",
  verbose_json: "verbose.json",
  vtt_json: "vtt.json",
  srt: "srt",
  vtt: "vtt",
  aud: "aud",
};

function stripVtt(payload: VerbosePayload): VerbosePayload {
  if (!("vtt_text" in payload)) return payload;
  const { vtt_text: _vtt, ...rest } = payload;
  return rest;
}

export function planExport(
  format: ResponseFormat,
  ctx: { payload: VerbosePayload; align: boolean; subtitleAvailable: boolean },
): ExportPlan {
  const { payload, align, subtitleAvailable } = ctx;
  const isSubtitle =
    format === "srt" || format === "vtt" || format === "aud" || format === "vtt_json";

  if (!align && isSubtitle) {
    return {
      kind: "unavailable",
      reason: "Subtitle formats require alignment — re-run with align enabled.",
    };
  }

  switch (format) {
    case "text":
      return { kind: "local", content: payload.text ?? "", mime: MIME.text };
    case "json":
      return {
        kind: "local",
        content: JSON.stringify({ text: payload.text ?? "" }),
        mime: MIME.json,
      };
    case "verbose_json":
      return {
        kind: "local",
        content: JSON.stringify(stripVtt(payload)),
        mime: MIME.verbose_json,
      };
    case "vtt_json":
      if (typeof payload.vtt_text === "string") {
        return { kind: "local", content: JSON.stringify(payload), mime: MIME.vtt_json };
      }
      break;
    case "vtt":
      if (typeof payload.vtt_text === "string") {
        return { kind: "local", content: payload.vtt_text, mime: MIME.vtt };
      }
      break;
    case "srt":
    case "aud":
      break;
  }

  if (isSubtitle && !subtitleAvailable) {
    return {
      kind: "unavailable",
      reason: "This server cannot render subtitle formats (ML extras not installed).",
    };
  }
  return { kind: "stored" };
}

export function exportMime(format: ResponseFormat): string {
  return MIME[format];
}

export function downloadFile(name: string, content: string, mime: string): void {
  const url = URL.createObjectURL(new Blob([content], { type: mime }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

export function exportFileName(sourceName: string, format: ResponseFormat): string {
  const base = sourceName.includes(".")
    ? sourceName.slice(0, sourceName.lastIndexOf("."))
    : sourceName;
  return `${base || "transcript"}.${EXPORT_EXTENSION[format]}`;
}
