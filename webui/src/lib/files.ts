/** Client-side upload validation. */

import { fmtBytes } from "./format";

const EXTENSIONS = new Set([
  "3gp",
  "aac",
  "aif",
  "aiff",
  "amr",
  "avi",
  "flac",
  "m4a",
  "m4b",
  "m4v",
  "mka",
  "mkv",
  "mov",
  "mp2",
  "mp3",
  "mp4",
  "mpeg",
  "mpg",
  "mpga",
  "oga",
  "ogg",
  "ogv",
  "opus",
  "ts",
  "wav",
  "webm",
  "wma",
  "wmv",
]);

export const MAX_FILE_BYTES = 4 * 1024 ** 3;
export const LARGE_FILE_BYTES = 1024 ** 3;

export type FileValidation =
  | { ok: true; warning: string | null }
  | { ok: false; reason: string };

export function validateFile(
  file: File,
  maxUploadBytes: number | null = null,
): FileValidation {
  if (file.size === 0) {
    return { ok: false, reason: "The file is empty." };
  }
  if (maxUploadBytes !== null && maxUploadBytes > 0 && file.size > maxUploadBytes) {
    return {
      ok: false,
      reason: `File is ${fmtBytes(file.size)} — over the server limit of ${fmtBytes(
        maxUploadBytes,
      )}.`,
    };
  }
  if (file.size > MAX_FILE_BYTES) {
    return { ok: false, reason: "Files over 4 GiB are not supported." };
  }

  const ext = file.name.includes(".")
    ? (file.name.split(".").pop() ?? "").toLowerCase()
    : "";
  const mimeOk = /^(audio|video)\//.test(file.type);
  if (!mimeOk && !EXTENSIONS.has(ext)) {
    return {
      ok: false,
      reason: `"${file.name}" does not look like an audio or video file.`,
    };
  }

  return {
    ok: true,
    warning:
      file.size > LARGE_FILE_BYTES
        ? "Large file — upload and processing may take a while."
        : null,
  };
}
