import { useEffect } from "react";

/** Prompt the browser's "Leave site?" dialog before an unload while `enabled`
 * — used to guard an in-progress or finished transcription (which cannot be
 * restored after a reload) against accidental loss. */
export function useUnloadWarning(enabled: boolean): void {
  useEffect(() => {
    if (!enabled) return;
    const handler = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      // Legacy assignment required by some browsers to trigger the prompt.
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [enabled]);
}
