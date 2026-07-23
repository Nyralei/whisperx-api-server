import { describe, expect, it } from "vitest";

import type { VerbosePayload } from "../api/types";
import { exportFileName, planExport } from "./exports";

const ctx = (payload: VerbosePayload, align: boolean, subtitleAvailable = true) => ({
  payload,
  align,
  subtitleAvailable,
});

describe("planExport", () => {
  it("derives text / json / verbose_json locally from the payload", () => {
    const payload: VerbosePayload = { text: "hello", vtt_text: "WEBVTT" };

    const text = planExport("text", ctx(payload, true));
    expect(text).toEqual({ kind: "local", content: "hello", mime: "text/plain" });

    const json = planExport("json", ctx(payload, true));
    expect(json).toMatchObject({ kind: "local", mime: "application/json" });
    expect(JSON.parse((json as { content: string }).content)).toEqual({ text: "hello" });
  });

  it("strips vtt_text out of a local verbose_json export", () => {
    const plan = planExport(
      "verbose_json",
      ctx({ text: "hi", vtt_text: "WEBVTT", segments: [] }, true),
    );
    const parsed = JSON.parse((plan as { content: string }).content);
    expect(parsed.vtt_text).toBeUndefined();
    expect(parsed.text).toBe("hi");
  });

  it("serves vtt locally when the payload already carries vtt_text", () => {
    const plan = planExport("vtt", ctx({ vtt_text: "WEBVTT\n\n..." }, true));
    expect(plan).toEqual({ kind: "local", content: "WEBVTT\n\n...", mime: "text/vtt" });
  });

  it("marks subtitle formats unavailable without alignment", () => {
    for (const fmt of ["srt", "vtt", "aud", "vtt_json"] as const) {
      expect(planExport(fmt, ctx({}, false)).kind).toBe("unavailable");
    }
  });

  it("routes server-only formats to stored in both modes", () => {
    expect(planExport("srt", ctx({}, true))).toEqual({ kind: "stored" });
    expect(planExport("vtt", ctx({}, true))).toEqual({ kind: "stored" });
    expect(planExport("aud", ctx({}, true))).toEqual({ kind: "stored" });
  });

  it("marks subtitle formats unavailable when the server cannot render them", () => {
    for (const fmt of ["srt", "aud"] as const) {
      expect(planExport(fmt, ctx({}, true, false)).kind).toBe("unavailable");
    }
  });
});

describe("exportFileName", () => {
  it("swaps the source extension for the format's extension", () => {
    expect(exportFileName("meeting.mp3", "srt")).toBe("meeting.srt");
    expect(exportFileName("clip.final.wav", "verbose_json")).toBe(
      "clip.final.verbose.json",
    );
  });

  it("handles names without an extension and empty names", () => {
    expect(exportFileName("noext", "json")).toBe("noext.json");
    expect(exportFileName("", "text")).toBe("transcript.txt");
  });
});
