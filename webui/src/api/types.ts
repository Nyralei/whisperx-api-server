/** Shapes returned by the whisperx-api-server endpoints the UI consumes. */

export interface WordTiming {
  word: string;
  /** Absent on tokens the aligner could not time (e.g. numerals). */
  start?: number;
  end?: number;
  score?: number;
  speaker?: string;
}

export interface Segment {
  start: number;
  end: number;
  text: string;
  speaker?: string;
  words?: WordTiming[];
}

/**
 * verbose_json / vtt_json payload. After alignment the server nests the
 * segment list: `segments` is then `{segments: [...], word_segments: [...]}`.
 */
export interface VerbosePayload {
  segments?:
    | Segment[]
    | { segments?: Segment[]; word_segments?: WordTiming[]; [k: string]: unknown };
  language?: string;
  text?: string;
  /** Present only for response_format=vtt_json. */
  vtt_text?: string;
  [k: string]: unknown;
}

export interface StageEntry {
  name: string;
  started_at?: number;
  completed_at?: number;
  duration_seconds?: number;
  in_progress?: boolean;
}

export type JobStatusValue = "queued" | "in_progress" | "completed" | "failed";

export interface RequestStatus {
  request_id: string;
  status: JobStatusValue;
  mode: string;
  stage: string;
  submitted_at: number;
  updated_at: number;
  completed_at: number | null;
  filename: string | null;
  stages: StageEntry[];
  error: string | null;
  error_type: string | null;
}

export interface ServerInfo {
  version: string;
  mode: string;
  uptime_seconds: number;
  max_upload_size_bytes?: number | null;
  subtitle_formats_available?: boolean;
  [k: string]: unknown;
}

export interface ModelCatalog {
  models: string[];
  default: string | null;
  loaded: string[] | null;
}
