/** Form-parameter model for POST /v1/audio/transcriptions. */

export const RESPONSE_FORMATS = [
  "text",
  "json",
  "verbose_json",
  "vtt_json",
  "srt",
  "vtt",
  "aud",
] as const;

export type ResponseFormat = (typeof RESPONSE_FORMATS)[number];

export interface TranscriptionOptions {
  /** "" = server default model. */
  model: string;
  /** ISO-639-1 code; "" = auto-detect (parameter omitted). */
  language: string;
  temperature: string;
  align: boolean;
  diarize: boolean;
  speakerEmbeddings: boolean;
  minSpeakers: string;
  maxSpeakers: string;
  highlightWords: boolean;
  suppressNumerals: boolean;
  hotwords: string;
  prompt: string;
  /** "" = server default. */
  batchSize: string;
  /** "" = server default. */
  chunkSize: string;
}

/** Defaults mirror the API's own form-parameter defaults. */
export const DEFAULT_OPTIONS: TranscriptionOptions = {
  model: "",
  language: "",
  temperature: "0",
  align: true,
  diarize: false,
  speakerEmbeddings: false,
  minSpeakers: "",
  maxSpeakers: "",
  highlightWords: false,
  suppressNumerals: true,
  hotwords: "",
  prompt: "",
  batchSize: "",
  chunkSize: "",
};

/**
 * Enforce the server's parameter invariants so the form can never submit a
 * combination the API would 422: diarize needs align, speaker embeddings need
 * diarize, subtitle formats need align.
 */
export function normalizeOptions(o: TranscriptionOptions): TranscriptionOptions {
  const next = { ...o };
  if (!next.align) next.diarize = false;
  if (!next.diarize) next.speakerEmbeddings = false;
  return next;
}

/**
 * Wire format for the interactive view. Direct mode always has whisperx
 * installed, so vtt_json (verbose payload + ready-made VTT) is free when
 * aligning. A Kafka-mode API replica may be a slim install without whisperx —
 * there the safe verbose_json is used and subtitle exports go through the
 * stored-result endpoint instead.
 */
export function wireFormat(
  o: TranscriptionOptions,
  mode: string | null,
): "vtt_json" | "verbose_json" {
  return o.align && mode === "direct" ? "vtt_json" : "verbose_json";
}

export function buildFormFields(
  o: TranscriptionOptions,
  format: string,
): Record<string, string | string[]> {
  const fields: Record<string, string | string[]> = {
    response_format: format,
    temperature: o.temperature.trim() || "0",
    align: String(o.align),
    diarize: String(o.diarize),
    speaker_embeddings: String(o.speakerEmbeddings),
    highlight_words: String(o.highlightWords),
    suppress_numerals: String(o.suppressNumerals),
  };
  if (o.diarize && o.minSpeakers.trim()) fields.min_speakers = o.minSpeakers.trim();
  if (o.diarize && o.maxSpeakers.trim()) fields.max_speakers = o.maxSpeakers.trim();
  if (o.model.trim()) fields.model = o.model.trim();
  if (o.language) fields.language = o.language;
  if (o.hotwords.trim()) fields.hotwords = o.hotwords.trim();
  if (o.prompt.trim()) fields.prompt = o.prompt.trim();
  if (o.batchSize.trim()) fields.batch_size = o.batchSize.trim();
  if (o.chunkSize.trim()) fields.chunk_size = o.chunkSize.trim();
  return fields;
}
