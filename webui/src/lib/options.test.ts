import { describe, expect, it } from "vitest";

import {
  buildFormFields,
  DEFAULT_OPTIONS,
  normalizeOptions,
  type TranscriptionOptions,
  wireFormat,
} from "./options";

const opts = (patch: Partial<TranscriptionOptions> = {}): TranscriptionOptions => ({
  ...DEFAULT_OPTIONS,
  ...patch,
});

describe("normalizeOptions", () => {
  it("forces diarize off when alignment is disabled", () => {
    const n = normalizeOptions(opts({ align: false, diarize: true }));
    expect(n.diarize).toBe(false);
  });

  it("forces speaker embeddings off when diarization is off", () => {
    const n = normalizeOptions(opts({ diarize: false, speakerEmbeddings: true }));
    expect(n.speakerEmbeddings).toBe(false);
  });

  it("preserves a valid combination", () => {
    const n = normalizeOptions(
      opts({ align: true, diarize: true, speakerEmbeddings: true }),
    );
    expect(n).toMatchObject({ diarize: true, speakerEmbeddings: true });
  });
});

describe("wireFormat", () => {
  it("uses vtt_json in direct mode when aligning (VTT comes free)", () => {
    expect(wireFormat(opts({ align: true }), "direct")).toBe("vtt_json");
  });

  it("uses verbose_json in kafka mode or without alignment", () => {
    expect(wireFormat(opts({ align: true }), "kafka")).toBe("verbose_json");
    expect(wireFormat(opts({ align: false }), "direct")).toBe("verbose_json");
    expect(wireFormat(opts({ align: true }), null)).toBe("verbose_json");
  });
});

describe("buildFormFields", () => {
  it("emits booleans and a default temperature, omitting empty optionals", () => {
    const f = buildFormFields(opts(), "json");
    expect(f).toMatchObject({
      response_format: "json",
      temperature: "0",
      align: "true",
      diarize: "false",
      speaker_embeddings: "false",
    });
    expect(f.model).toBeUndefined();
    expect(f.language).toBeUndefined();
    expect(f.hotwords).toBeUndefined();
  });

  it("emits min/max speakers only when diarizing", () => {
    const off = buildFormFields(opts({ minSpeakers: "2", maxSpeakers: "4" }), "json");
    expect(off.min_speakers).toBeUndefined();
    expect(off.max_speakers).toBeUndefined();

    const on = buildFormFields(
      opts({ diarize: true, minSpeakers: "2", maxSpeakers: "4" }),
      "json",
    );
    expect(on.min_speakers).toBe("2");
    expect(on.max_speakers).toBe("4");
  });

  it("trims text fields and includes provided optionals", () => {
    const f = buildFormFields(
      opts({
        model: "  large-v3  ",
        language: "en",
        hotwords: " foo, bar ",
        batchSize: "8",
      }),
      "verbose_json",
    );
    expect(f.model).toBe("large-v3");
    expect(f.language).toBe("en");
    expect(f.hotwords).toBe("foo, bar");
    expect(f.batch_size).toBe("8");
  });
});
