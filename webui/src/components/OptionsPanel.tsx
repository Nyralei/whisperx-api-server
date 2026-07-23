import { LANGUAGES } from "../lib/languages";
import { normalizeOptions, type TranscriptionOptions } from "../lib/options";
import { Field, inputClass, Toggle } from "./ui";

const SUGGESTED_MODELS = [
  "large-v3",
  "large-v3-turbo",
  "large-v2",
  "distil-large-v3",
  "medium",
  "small",
  "base",
  "tiny",
];

export function OptionsPanel(props: {
  options: TranscriptionOptions;
  onChange: (options: TranscriptionOptions) => void;
  modelSuggestions: string[];
  disabled?: boolean;
}) {
  const { options, disabled } = props;
  const set = (patch: Partial<TranscriptionOptions>) =>
    props.onChange(normalizeOptions({ ...options, ...patch }));

  const suggestions = [...new Set([...props.modelSuggestions, ...SUGGESTED_MODELS])];

  return (
    <fieldset disabled={disabled} className="flex flex-col gap-4">
      <legend className="sr-only">Transcription options</legend>

      <Field id="opt-model" label="Model" hint="Empty = server default.">
        <input
          id="opt-model"
          list="opt-model-suggestions"
          className={inputClass}
          value={options.model}
          placeholder="server default"
          onChange={(e) => set({ model: e.target.value })}
        />
        <datalist id="opt-model-suggestions">
          {suggestions.map((m) => (
            <option key={m} value={m} />
          ))}
        </datalist>
      </Field>

      <Field id="opt-language" label="Language">
        <select
          id="opt-language"
          className={inputClass}
          value={options.language}
          onChange={(e) => set({ language: e.target.value })}
        >
          <option value="">Auto-detect</option>
          {LANGUAGES.map((l) => (
            <option key={l.code} value={l.code}>
              {l.label} ({l.code})
            </option>
          ))}
        </select>
      </Field>

      <div className="flex flex-col gap-2.5 rounded-lg border border-border bg-surface-2 p-3">
        <p className="text-xs font-semibold uppercase tracking-wide text-fg-muted">
          Pipeline
        </p>
        <Toggle
          id="opt-align"
          label="Align timestamps"
          hint="Word-level timing; required for subtitles and diarization."
          checked={options.align}
          onChange={(v) => set({ align: v })}
        />
        <Toggle
          id="opt-diarize"
          label="Diarize speakers"
          hint={options.align ? undefined : "Requires alignment."}
          checked={options.diarize}
          disabled={!options.align}
          onChange={(v) => set({ diarize: v })}
        />
        <Toggle
          id="opt-speaker-embeddings"
          label="Speaker embeddings"
          hint={
            options.diarize
              ? "Adds per-speaker voiceprint vectors to the downloaded JSON (for speaker identification). No change to the on-screen transcript."
              : "Requires diarization."
          }
          checked={options.speakerEmbeddings}
          disabled={!options.diarize}
          onChange={(v) => set({ speakerEmbeddings: v })}
        />
        {options.diarize ? (
          <div className="grid grid-cols-2 gap-3">
            <Field id="opt-min-speakers" label="Min speakers" hint="Optional.">
              <input
                id="opt-min-speakers"
                type="number"
                min="1"
                className={inputClass}
                value={options.minSpeakers}
                placeholder="auto"
                onChange={(e) => set({ minSpeakers: e.target.value })}
              />
            </Field>
            <Field id="opt-max-speakers" label="Max speakers" hint="Optional.">
              <input
                id="opt-max-speakers"
                type="number"
                min="1"
                className={inputClass}
                value={options.maxSpeakers}
                placeholder="auto"
                onChange={(e) => set({ maxSpeakers: e.target.value })}
              />
            </Field>
          </div>
        ) : null}
      </div>

      <details className="rounded-lg border border-border">
        <summary className="cursor-pointer select-none px-3 py-2 text-sm font-medium text-fg-muted hover:bg-surface-2">
          Advanced
        </summary>
        <div className="flex flex-col gap-4 border-t border-border p-3">
          <Field id="opt-temperature" label="Temperature">
            <input
              id="opt-temperature"
              type="number"
              min="0"
              max="1"
              step="0.1"
              className={inputClass}
              value={options.temperature}
              onChange={(e) => set({ temperature: e.target.value })}
            />
          </Field>
          <Field
            id="opt-prompt"
            label="Prompt"
            hint="Optional context hint for the decoder."
          >
            <textarea
              id="opt-prompt"
              rows={2}
              className={inputClass}
              value={options.prompt}
              onChange={(e) => set({ prompt: e.target.value })}
            />
          </Field>
          <Field
            id="opt-hotwords"
            label="Hotwords"
            hint="Comma-separated terms to bias toward."
          >
            <input
              id="opt-hotwords"
              className={inputClass}
              value={options.hotwords}
              onChange={(e) => set({ hotwords: e.target.value })}
            />
          </Field>
          <div className="grid grid-cols-2 gap-3">
            <Field id="opt-batch" label="Batch size" hint="Empty = default.">
              <input
                id="opt-batch"
                type="number"
                min="1"
                className={inputClass}
                value={options.batchSize}
                placeholder="default"
                onChange={(e) => set({ batchSize: e.target.value })}
              />
            </Field>
            <Field id="opt-chunk" label="Chunk size (s)" hint="Empty = default.">
              <input
                id="opt-chunk"
                type="number"
                min="1"
                className={inputClass}
                value={options.chunkSize}
                placeholder="default"
                onChange={(e) => set({ chunkSize: e.target.value })}
              />
            </Field>
          </div>
          <Toggle
            id="opt-suppress-numerals"
            label="Suppress numerals"
            hint="Spell out numbers instead of digits."
            checked={options.suppressNumerals}
            onChange={(v) => set({ suppressNumerals: v })}
          />
          <Toggle
            id="opt-highlight-words"
            label="Highlight words in subtitles"
            hint="Affects vtt / srt output only."
            checked={options.highlightWords}
            onChange={(v) => set({ highlightWords: v })}
          />
        </div>
      </details>
    </fieldset>
  );
}
