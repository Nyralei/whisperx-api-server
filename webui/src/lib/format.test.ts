import { describe, expect, it } from "vitest";

import { fmtBytes, fmtClock, fmtDuration, prettyStageName, speakerColor } from "./format";

describe("fmtClock", () => {
  it("formats sub-hour times with tenths and hour times with h:mm:ss", () => {
    expect(fmtClock(0)).toBe("0:00.0");
    expect(fmtClock(65.5)).toBe("1:05.5");
    expect(fmtClock(3661)).toBe("1:01:01");
  });
  it("clamps invalid input", () => {
    expect(fmtClock(-5)).toBe("0:00.0");
    expect(fmtClock(Number.NaN)).toBe("0:00.0");
  });
});

describe("fmtDuration", () => {
  it("scales units and rejects invalid input", () => {
    expect(fmtDuration(0.5)).toBe("500 ms");
    expect(fmtDuration(5)).toBe("5.0 s");
    expect(fmtDuration(125)).toBe("2m 05s");
    expect(fmtDuration(-1)).toBe("—");
  });
});

describe("fmtBytes", () => {
  it("uses binary units", () => {
    expect(fmtBytes(500)).toBe("500 B");
    expect(fmtBytes(1536)).toBe("1.5 KiB");
    expect(fmtBytes(5 * 1024 * 1024)).toBe("5.0 MiB");
    expect(fmtBytes(2 * 1024 ** 3)).toBe("2.0 GiB");
  });
});

describe("prettyStageName", () => {
  it("splits the worker. prefix into a chip flag and humanizes the label", () => {
    expect(prettyStageName("worker.transcribe")).toEqual({
      label: "Transcribe",
      worker: true,
    });
    expect(prettyStageName("load_audio")).toEqual({ label: "Load audio", worker: false });
  });
});

describe("speakerColor", () => {
  it("is deterministic and walks the hue by the golden angle", () => {
    expect(speakerColor(0)).toBe("hsl(0.0deg 62% 28%)");
    expect(speakerColor(1)).toBe("hsl(137.5deg 62% 28%)");
    expect(speakerColor(2)).toBe(speakerColor(2));
  });
});
