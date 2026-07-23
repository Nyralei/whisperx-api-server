/**
 * Thin client over the whisperx-api-server HTTP API. Same-origin (the UI is
 * served by the API), so no base URL and no CORS concerns.
 */

import type { ModelCatalog, RequestStatus, ServerInfo } from "./types";

export class ApiError extends Error {
  /** HTTP status; 0 for network failures, -1 for a client-side abort. */
  readonly status: number;
  readonly errorType: string | null;

  constructor(message: string, status: number, errorType: string | null = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.errorType = errorType;
  }
}

export function isAbort(e: unknown): boolean {
  return e instanceof ApiError && e.status === -1;
}

export function isNetworkError(e: unknown): boolean {
  return e instanceof ApiError && e.status === 0;
}

function authHeaders(apiKey: string | null): Record<string, string> {
  return apiKey ? { Authorization: `Bearer ${apiKey}` } : {};
}

/** FastAPI error bodies: {"detail": string | ValidationError[]}. */
function detailFromBody(body: string, fallback: string): string {
  try {
    const parsed = JSON.parse(body);
    const detail = parsed?.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((d) => (typeof d?.msg === "string" ? d.msg : JSON.stringify(d)))
        .join("; ");
    }
  } catch {
    /* not JSON */
  }
  return fallback;
}

async function throwForResponse(resp: Response): Promise<never> {
  const body = await resp.text().catch(() => "");
  throw new ApiError(detailFromBody(body, `HTTP ${resp.status}`), resp.status, null);
}

/**
 * Request id sent as X-Request-ID; must match the server's [A-Za-z0-9._-]{1,128}.
 * crypto.randomUUID is unavailable in non-secure contexts (plain-HTTP LAN
 * deployments), hence the fallback.
 */
export function newRequestId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  const alphabet = "abcdefghijklmnopqrstuvwxyz0123456789";
  let suffix = "";
  for (let i = 0; i < 20; i++) {
    suffix += alphabet[Math.floor(Math.random() * alphabet.length)];
  }
  return `webui-${Date.now().toString(36)}-${suffix}`;
}

export type InfoResult =
  | { ok: true; info: ServerInfo }
  | { ok: false; status: number | null };

export async function getInfo(apiKey: string | null): Promise<InfoResult> {
  try {
    const resp = await fetch("/info", { headers: authHeaders(apiKey) });
    if (resp.ok) return { ok: true, info: (await resp.json()) as ServerInfo };
    return { ok: false, status: resp.status };
  } catch {
    return { ok: false, status: null };
  }
}

/** Null on any failure — the status poller treats misses as "not yet visible". */
export async function getStatus(
  requestId: string,
  apiKey: string | null,
): Promise<RequestStatus | null> {
  try {
    const resp = await fetch(
      `/v1/audio/transcriptions/${encodeURIComponent(requestId)}/status`,
      { headers: authHeaders(apiKey) },
    );
    if (!resp.ok) return null;
    return (await resp.json()) as RequestStatus;
  } catch {
    return null;
  }
}

export async function getModelCatalog(apiKey: string | null): Promise<ModelCatalog> {
  const empty: ModelCatalog = { models: [], default: null, loaded: null };
  try {
    const resp = await fetch("/models/catalog", { headers: authHeaders(apiKey) });
    if (!resp.ok) return empty;
    const parsed = await resp.json();
    const strings = (v: unknown): string[] =>
      Array.isArray(v) ? v.filter((m): m is string => typeof m === "string") : [];
    return {
      models: strings(parsed?.models),
      default: typeof parsed?.default === "string" ? parsed.default : null,
      loaded: Array.isArray(parsed?.loaded) ? strings(parsed.loaded) : null,
    };
  } catch {
    return empty;
  }
}

export async function streamStatusEvents(
  requestId: string,
  apiKey: string | null,
  opts: { onStatus: (status: RequestStatus) => void; signal?: AbortSignal },
): Promise<void> {
  const resp = await fetch(
    `/v1/audio/transcriptions/${encodeURIComponent(requestId)}/events`,
    { headers: authHeaders(apiKey), signal: opts.signal },
  );
  if (!resp.ok || !resp.body) {
    throw new ApiError(`HTTP ${resp.status}`, resp.status);
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    for (;;) {
      const sep = buffer.indexOf("\n\n");
      if (sep === -1) break;
      const event = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const data = event
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trim())
        .join("\n");
      if (!data || data === "{}") continue;
      let status: RequestStatus | null = null;
      try {
        status = JSON.parse(data) as RequestStatus;
      } catch {
        status = null;
      }
      if (status) opts.onStatus(status);
    }
  }
}

export interface PostHandle {
  promise: Promise<{ contentType: string; body: string }>;
  abort: () => void;
}

/**
 * POST /v1/audio/transcriptions via XHR — fetch() cannot report upload
 * progress, and multi-hundred-MB media files make that progress the main
 * feedback during the first phase of a job.
 */
export function postTranscription(opts: {
  file: File;
  fields: Record<string, string | string[]>;
  requestId: string;
  apiKey: string | null;
  onUploadProgress?: (fraction: number | null) => void;
  onUploadComplete?: () => void;
}): PostHandle {
  const xhr = new XMLHttpRequest();

  const promise = new Promise<{ contentType: string; body: string }>(
    (resolve, reject) => {
      const form = new FormData();
      form.append("file", opts.file, opts.file.name);
      for (const [key, value] of Object.entries(opts.fields)) {
        if (Array.isArray(value)) {
          for (const item of value) form.append(key, item);
        } else {
          form.append(key, value);
        }
      }

      xhr.open("POST", "/v1/audio/transcriptions");
      xhr.setRequestHeader("X-Request-ID", opts.requestId);
      if (opts.apiKey) {
        xhr.setRequestHeader("Authorization", `Bearer ${opts.apiKey}`);
      }

      xhr.upload.onprogress = (e) => {
        opts.onUploadProgress?.(e.lengthComputable ? e.loaded / e.total : null);
      };
      // Fires once the request body is fully sent — the point at which the
      // server registers the request, so status polling can begin without 404s.
      xhr.upload.onload = () => opts.onUploadComplete?.();
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve({
            contentType: xhr.getResponseHeader("Content-Type") ?? "",
            body: xhr.responseText,
          });
        } else {
          reject(
            new ApiError(
              detailFromBody(xhr.responseText, `HTTP ${xhr.status}`),
              xhr.status,
            ),
          );
        }
      };
      xhr.onerror = () => reject(new ApiError("Network error", 0));
      xhr.ontimeout = () => reject(new ApiError("Network timeout", 0));
      xhr.onabort = () => reject(new ApiError("Cancelled", -1, "cancelled"));

      xhr.send(form);
    },
  );

  return { promise, abort: () => xhr.abort() };
}

/** GET /v1/audio/transcriptions/{id}/result — re-formats a finished job (both modes). */
export async function fetchStoredResult(
  requestId: string,
  responseFormat: string,
  highlightWords: boolean,
  apiKey: string | null,
): Promise<{ contentType: string; body: string }> {
  const params = new URLSearchParams({
    response_format: responseFormat,
    highlight_words: String(highlightWords),
  });
  const resp = await fetch(
    `/v1/audio/transcriptions/${encodeURIComponent(requestId)}/result?${params}`,
    { headers: authHeaders(apiKey) },
  );
  if (!resp.ok) await throwForResponse(resp);
  return {
    contentType: resp.headers.get("Content-Type") ?? "",
    body: await resp.text(),
  };
}
