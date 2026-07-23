import { useEffect, useState } from "react";

import {
  DEFAULT_OPTIONS,
  normalizeOptions,
  type TranscriptionOptions,
} from "../lib/options";

const STORAGE_KEY = "whisperx-webui-options";

function load(): TranscriptionOptions {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_OPTIONS;
    const parsed = JSON.parse(raw) as Partial<TranscriptionOptions>;
    // Merge onto defaults so a stored blob from an older version still gets any
    // newly-added fields, then re-apply the server invariants.
    return normalizeOptions({ ...DEFAULT_OPTIONS, ...parsed });
  } catch {
    return DEFAULT_OPTIONS;
  }
}

/** Transcription options that persist across reloads (localStorage-backed). */
export function useStoredOptions() {
  const state = useState<TranscriptionOptions>(load);
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state[0]));
    } catch {
      /* storage unavailable / over quota — persistence is best-effort */
    }
  }, [state[0]]);
  return state;
}
