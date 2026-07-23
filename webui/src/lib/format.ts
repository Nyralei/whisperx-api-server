/** Small presentation helpers: time, size, stage names, speaker colors. */

export function fmtClock(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00.0";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  if (h > 0) {
    return `${h}:${String(m).padStart(2, "0")}:${String(Math.floor(s)).padStart(2, "0")}`;
  }
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}

export function fmtDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "—";
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`;
  if (seconds < 90) return `${seconds.toFixed(1)} s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m ${String(s).padStart(2, "0")}s`;
}

export function fmtBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KiB", "MiB", "GiB"];
  let value = bytes;
  let unit = "B";
  for (const u of units) {
    if (value < 1024) break;
    value /= 1024;
    unit = u;
  }
  return `${value.toFixed(value >= 10 ? 0 : 1)} ${unit}`;
}

export interface StageLabel {
  label: string;
  worker: boolean;
}

/**
 * Stage names are backend-defined and differ between direct and Kafka modes;
 * render whatever arrives instead of matching a fixed list. Worker-side stages
 * come prefixed "worker." — surfaced as a chip, not part of the label.
 */
export function prettyStageName(name: string): StageLabel {
  const worker = name.startsWith("worker.");
  const bare = worker ? name.slice("worker.".length) : name;
  const label = bare.replace(/[_.]/g, " ").trim();
  return { label: label.charAt(0).toUpperCase() + label.slice(1), worker };
}

/**
 * Deterministic, well-spaced speaker colors (golden-angle hue walk). Lightness is
 * held low enough that white text clears WCAG AA (≥4.5:1) across every hue.
 */
export function speakerColor(index: number): string {
  const hue = (index * 137.508) % 360;
  return `hsl(${hue.toFixed(1)}deg 62% 28%)`;
}
