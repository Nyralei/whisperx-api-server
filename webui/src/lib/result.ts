/** Normalization of the verbose payload's two segment shapes. */

import type { Segment, VerbosePayload, WordTiming } from "../api/types";

export interface NormalizedResult {
  segments: Segment[];
  wordSegments: WordTiming[];
  language: string | null;
  text: string;
  /** Ordered unique speaker labels, first-appearance order. */
  speakers: string[];
}

export function normalizeResult(payload: VerbosePayload): NormalizedResult {
  let segments: Segment[] = [];
  let wordSegments: WordTiming[] = [];

  const raw = payload.segments;
  if (Array.isArray(raw)) {
    segments = raw;
  } else if (raw && typeof raw === "object") {
    if (Array.isArray(raw.segments)) segments = raw.segments;
    if (Array.isArray(raw.word_segments)) wordSegments = raw.word_segments;
  }

  const speakers: string[] = [];
  const seen = new Set<string>();
  for (const seg of segments) {
    if (seg.speaker && !seen.has(seg.speaker)) {
      seen.add(seg.speaker);
      speakers.push(seg.speaker);
    }
  }

  return {
    segments,
    wordSegments,
    language: typeof payload.language === "string" ? payload.language : null,
    text: typeof payload.text === "string" ? payload.text : "",
    speakers,
  };
}

export function hasWordTimings(segments: Segment[]): boolean {
  return segments.some((s) => Array.isArray(s.words) && s.words.length > 0);
}
