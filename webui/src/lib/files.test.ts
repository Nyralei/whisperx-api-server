import { describe, expect, it } from "vitest";

import { LARGE_FILE_BYTES, MAX_FILE_BYTES, validateFile } from "./files";

function fakeFile(name: string, size: number, type = ""): File {
  const f = new File(["x"], name, { type });
  Object.defineProperty(f, "size", { value: size });
  return f;
}

describe("validateFile", () => {
  it("rejects an empty file", () => {
    const v = validateFile(fakeFile("a.mp3", 0));
    expect(v.ok).toBe(false);
  });

  it("rejects files larger than the ceiling", () => {
    const v = validateFile(fakeFile("a.mp3", MAX_FILE_BYTES + 1));
    expect(v).toMatchObject({ ok: false });
  });

  it("accepts a known media extension without a MIME type", () => {
    const v = validateFile(fakeFile("podcast.mp3", 1000, ""));
    expect(v).toEqual({ ok: true, warning: null });
  });

  it("accepts any file with an audio/* or video/* MIME type", () => {
    const v = validateFile(fakeFile("blob", 1000, "audio/wav"));
    expect(v.ok).toBe(true);
  });

  it("rejects a non-media file", () => {
    const v = validateFile(fakeFile("notes.txt", 1000, "text/plain"));
    expect(v.ok).toBe(false);
  });

  it("warns (but accepts) on large files", () => {
    const v = validateFile(fakeFile("movie.mkv", LARGE_FILE_BYTES + 1));
    expect(v.ok).toBe(true);
    expect((v as { warning: string | null }).warning).toBeTruthy();
  });

  it("rejects files over the server upload cap", () => {
    const v = validateFile(fakeFile("a.mp3", 2000, "audio/wav"), 1000);
    expect(v).toMatchObject({ ok: false });
    expect((v as { reason: string }).reason).toContain("server limit");
  });

  it("ignores a zero or null server cap (unlimited)", () => {
    expect(validateFile(fakeFile("a.mp3", 2000, "audio/wav"), 0).ok).toBe(true);
    expect(validateFile(fakeFile("a.mp3", 2000, "audio/wav"), null).ok).toBe(true);
  });
});
