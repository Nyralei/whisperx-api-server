import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { Segment, VerbosePayload, WordTiming } from "../api/types";
import { useAudioSync } from "../hooks/useAudioSync";
import { fmtClock, fmtDuration, speakerColor } from "../lib/format";
import type { TranscriptionOptions } from "../lib/options";
import { hasWordTimings, normalizeResult } from "../lib/result";
import { ExportBar } from "./ExportBar";
import { Button, Toggle } from "./ui";

const prefersReducedMotion = () =>
  window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;

const WordSpan = memo(function WordSpan(props: {
  word: WordTiming;
  active: boolean;
  seekable: boolean;
  onSeek: (t: number) => void;
}) {
  const { word } = props;
  const start = word.start;
  const text = word.word;
  const cls = props.active
    ? "rounded-sm bg-accent text-on-accent"
    : start !== undefined && props.seekable
      ? "rounded-sm hover:bg-accent-subtle"
      : "";

  if (start === undefined || !props.seekable) {
    return <span className={cls}>{text} </span>;
  }
  return (
    <>
      <button
        type="button"
        onClick={() => props.onSeek(start)}
        className={`${cls} focus:outline-none focus-visible:ring-1 focus-visible:ring-focus`}
      >
        {text}
      </button>{" "}
    </>
  );
});

const SegmentRow = memo(function SegmentRow(props: {
  segment: Segment;
  index: number;
  isActive: boolean;
  activeWord: number;
  wordLevel: boolean;
  playable: boolean;
  speakerBg: string | undefined;
  onSeek: (t: number) => void;
}) {
  const { segment, isActive, playable } = props;
  const words = segment.words ?? [];
  return (
    <li
      data-segment={props.index}
      className={`segment-row group flex gap-3 border-b border-border px-4 py-2.5 last:border-b-0 ${
        isActive ? "bg-accent-subtle" : ""
      }`}
    >
      <div className="w-24 shrink-0 pt-0.5 text-right">
        <button
          type="button"
          disabled={!playable}
          onClick={() => props.onSeek(segment.start)}
          title={playable ? "Jump to this segment" : undefined}
          className="text-xs tabular-nums text-fg-subtle hover:text-accent-text focus:outline-none focus-visible:ring-1 focus-visible:ring-focus disabled:hover:text-fg-subtle"
        >
          [{fmtClock(segment.start)}]
        </button>
        {segment.speaker ? (
          <span
            className="mt-1 block truncate rounded px-1 py-0.5 text-[10px] font-medium text-white"
            style={{ backgroundColor: props.speakerBg }}
            title={segment.speaker}
          >
            {segment.speaker}
          </span>
        ) : null}
      </div>
      <p className="min-w-0 flex-1 font-sans text-sm leading-relaxed text-fg">
        {props.wordLevel && words.length > 0 ? (
          words.map((w, wi) => (
            <WordSpan
              key={`${w.start}-${w.end}-${w.word}`}
              word={w}
              active={isActive && wi === props.activeWord}
              seekable={playable}
              onSeek={props.onSeek}
            />
          ))
        ) : playable ? (
          <button
            type="button"
            onClick={() => props.onSeek(segment.start)}
            className="text-left hover:text-accent-text focus:outline-none focus-visible:ring-1 focus-visible:ring-focus"
          >
            {segment.text.trim()}
          </button>
        ) : (
          segment.text.trim()
        )}
      </p>
    </li>
  );
});

export function ResultView(props: {
  payload: VerbosePayload;
  file: File;
  options: TranscriptionOptions;
  requestId: string;
  subtitleAvailable: boolean;
  apiKey: string | null;
  durationSeconds: number | null;
  optionsChanged: boolean;
  onRerun: () => void;
  onReset: () => void;
}) {
  const normalized = useMemo(() => normalizeResult(props.payload), [props.payload]);
  const { segments, speakers } = normalized;
  const wordLevel = useMemo(() => hasWordTimings(segments), [segments]);

  const speakerIndex = useMemo(() => new Map(speakers.map((s, i) => [s, i])), [speakers]);

  const audioRef = useRef<HTMLAudioElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const headingRef = useRef<HTMLHeadingElement>(null);
  const [playable, setPlayable] = useState(true);
  const [follow, setFollow] = useState(true);
  const [copied, setCopied] = useState(false);

  const { active, playing } = useAudioSync(audioRef, segments, wordLevel, playable);

  const objectUrl = useMemo(() => URL.createObjectURL(props.file), [props.file]);
  useEffect(() => () => URL.revokeObjectURL(objectUrl), [objectUrl]);

  // Move focus to the result heading when it appears so keyboard / screen-reader
  // users are taken to the new content.
  useEffect(() => {
    headingRef.current?.focus();
  }, []);

  useEffect(() => {
    if (!follow || !playing || active.segment < 0) return;
    listRef.current?.querySelector(`[data-segment="${active.segment}"]`)?.scrollIntoView({
      block: "nearest",
      behavior: prefersReducedMotion() ? "auto" : "smooth",
    });
  }, [active.segment, follow, playing]);

  const seek = useCallback(
    (t: number) => {
      const audio = audioRef.current;
      if (!audio || !playable) return;
      audio.currentTime = t;
      void audio.play();
    },
    [playable],
  );

  const copyText = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(normalized.text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable (insecure context) */
    }
  }, [normalized.text]);

  return (
    <section aria-label="Transcription result" className="flex flex-col gap-4">
      <div className="rounded-sm border border-border bg-surface p-5">
        <div className="mb-3 flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <h2
            ref={headingRef}
            tabIndex={-1}
            className="text-base font-semibold text-fg focus:outline-none"
          >
            {props.file.name}
          </h2>
          <p className="text-xs text-fg-muted">
            {segments.length} segment{segments.length === 1 ? "" : "s"}
            {normalized.language ? ` · language: ${normalized.language}` : ""}
            {wordLevel ? " · word timings" : ""}
            {props.durationSeconds !== null
              ? ` · done in ${fmtDuration(props.durationSeconds)}`
              : ""}
          </p>
          <div className="ml-auto flex flex-wrap items-center gap-2">
            {props.optionsChanged ? (
              <span className="text-[11px] font-medium text-accent-text">
                options changed
              </span>
            ) : null}
            <Button
              variant={props.optionsChanged ? "primary" : "secondary"}
              onClick={props.onRerun}
              title="Re-transcribe this same file with the current options — no re-upload needed"
            >
              Re-run
            </Button>
            <Button variant="secondary" onClick={props.onReset}>
              New file
            </Button>
          </div>
        </div>

        <ExportBar
          payload={props.payload}
          requestId={props.requestId}
          file={props.file}
          options={props.options}
          subtitleAvailable={props.subtitleAvailable}
          apiKey={props.apiKey}
        />

        {speakers.length > 0 ? (
          <ul aria-label="Speakers" className="mt-3 flex flex-wrap gap-2">
            {speakers.map((s, i) => (
              <li
                key={s}
                className="rounded-sm px-2.5 py-0.5 text-xs font-medium text-white"
                style={{ backgroundColor: speakerColor(i) }}
              >
                {s}
              </li>
            ))}
          </ul>
        ) : null}

        <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-center">
          {playable ? (
            // biome-ignore lint/a11y/useMediaCaption: user-supplied audio has no caption track
            <audio
              ref={audioRef}
              controls
              src={objectUrl}
              onError={() => setPlayable(false)}
              className="h-11 w-full sm:h-10 sm:w-auto sm:min-w-0 sm:flex-1"
            />
          ) : (
            <p className="text-xs text-fg-muted">
              This browser can’t play the source file — click-to-seek is disabled.
            </p>
          )}
          <div className="flex items-center gap-3">
            {playable ? (
              <Toggle
                id="follow-playback"
                label="Follow playback"
                checked={follow}
                onChange={setFollow}
              />
            ) : null}
            <Button variant="secondary" onClick={() => void copyText()}>
              {copied ? "Copied!" : "Copy text"}
            </Button>
            <span role="status" aria-live="polite" className="sr-only">
              {copied ? "Transcript copied to clipboard" : ""}
            </span>
          </div>
        </div>
      </div>

      <div
        ref={listRef}
        className="term-scroll max-h-[60vh] overflow-y-auto rounded-sm border border-border bg-surface"
      >
        {segments.length === 0 ? (
          <p className="p-5 text-sm text-fg-muted">
            The result contains no segments
            {normalized.text ? " — raw text below:" : "."}
            {normalized.text ? (
              <span className="mt-2 block whitespace-pre-wrap text-fg">
                {normalized.text}
              </span>
            ) : null}
          </p>
        ) : (
          <ol>
            {segments.map((seg, i) => {
              const spkIdx = seg.speaker ? speakerIndex.get(seg.speaker) : undefined;
              return (
                <SegmentRow
                  key={`${seg.start}-${seg.end}`}
                  segment={seg}
                  index={i}
                  isActive={i === active.segment}
                  activeWord={i === active.segment ? active.word : -1}
                  wordLevel={wordLevel}
                  playable={playable}
                  speakerBg={seg.speaker ? speakerColor(spkIdx ?? 0) : undefined}
                  onSeek={seek}
                />
              );
            })}
          </ol>
        )}
      </div>
    </section>
  );
}
