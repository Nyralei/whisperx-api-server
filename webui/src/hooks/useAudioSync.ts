import { type RefObject, useEffect, useState } from "react";

import type { Segment } from "../api/types";

export interface ActivePosition {
  segment: number;
  word: number;
}

function findActive(
  segments: Segment[],
  time: number,
  wordLevel: boolean,
): ActivePosition {
  let active = -1;
  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i];
    if (time >= seg.start && time < seg.end) {
      active = i;
      break;
    }
    if (seg.start > time) break;
    active = i; // between segments: stick to the last one started
  }
  if (active === -1 || !wordLevel) return { segment: active, word: -1 };

  const words = segments[active].words ?? [];
  let activeWord = -1;
  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    if (w.start === undefined) continue;
    if (time >= w.start && (w.end === undefined || time < w.end)) {
      activeWord = i;
      break;
    }
    if (w.start > time) break;
    activeWord = i;
  }
  return { segment: active, word: activeWord };
}

/**
 * Tracks the segment/word under the audio playhead. Word-level highlighting needs
 * finer granularity than the ~4 Hz `timeupdate` event, so playback is followed with
 * rAF while playing. State only updates when the active position actually changes.
 */
export function useAudioSync(
  audioRef: RefObject<HTMLAudioElement | null>,
  segments: Segment[],
  wordLevel: boolean,
  enabled: boolean,
): { active: ActivePosition; playing: boolean } {
  const [active, setActive] = useState<ActivePosition>({ segment: -1, word: -1 });
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio || !enabled) return;

    let raf = 0;
    const sync = () => {
      const pos = findActive(segments, audio.currentTime, wordLevel);
      setActive((prev) =>
        prev.segment === pos.segment && prev.word === pos.word ? prev : pos,
      );
    };
    const loop = () => {
      sync();
      raf = requestAnimationFrame(loop);
    };
    const onPlay = () => {
      setPlaying(true);
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(loop);
    };
    const onPause = () => {
      setPlaying(false);
      cancelAnimationFrame(raf);
      sync();
    };

    audio.addEventListener("play", onPlay);
    audio.addEventListener("pause", onPause);
    audio.addEventListener("seeked", sync);
    return () => {
      cancelAnimationFrame(raf);
      audio.removeEventListener("play", onPlay);
      audio.removeEventListener("pause", onPause);
      audio.removeEventListener("seeked", sync);
    };
  }, [audioRef, segments, wordLevel, enabled]);

  return { active, playing };
}
