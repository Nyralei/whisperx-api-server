import { describe, expect, it } from "vitest";

import type { VerbosePayload } from "../api/types";
import { hasWordTimings, normalizeResult } from "./result";

describe("normalizeResult", () => {
  it("reads a flat segment array and extracts speakers in first-appearance order", () => {
    const payload: VerbosePayload = {
      segments: [
        { start: 0, end: 1, text: "a", speaker: "SPEAKER_01" },
        { start: 1, end: 2, text: "b", speaker: "SPEAKER_00" },
        { start: 2, end: 3, text: "c", speaker: "SPEAKER_01" },
      ],
      language: "en",
      text: "a b c",
    };
    const r = normalizeResult(payload);
    expect(r.segments).toHaveLength(3);
    expect(r.speakers).toEqual(["SPEAKER_01", "SPEAKER_00"]);
    expect(r.language).toBe("en");
    expect(r.text).toBe("a b c");
  });

  it("reads the nested aligned shape (segments.segments + word_segments)", () => {
    const payload: VerbosePayload = {
      segments: {
        segments: [{ start: 0, end: 1, text: "hi", words: [{ word: "hi", start: 0 }] }],
        word_segments: [{ word: "hi", start: 0, end: 1 }],
      },
    };
    const r = normalizeResult(payload);
    expect(r.segments).toHaveLength(1);
    expect(r.wordSegments).toHaveLength(1);
    expect(r.language).toBeNull();
    expect(r.text).toBe("");
  });
});

describe("hasWordTimings", () => {
  it("is true only when a segment carries a non-empty words array", () => {
    expect(
      hasWordTimings([{ start: 0, end: 1, text: "x", words: [{ word: "x" }] }]),
    ).toBe(true);
    expect(hasWordTimings([{ start: 0, end: 1, text: "x" }])).toBe(false);
    expect(hasWordTimings([{ start: 0, end: 1, text: "x", words: [] }])).toBe(false);
  });
});
