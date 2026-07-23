import { useEffect, useRef, useState } from "react";

function hasFiles(e: DragEvent): boolean {
  return Array.from(e.dataTransfer?.types ?? []).includes("Files");
}

/**
 * Accept a file dropped anywhere on the page. Returns whether a file is
 * currently being dragged over the window (to show an overlay). Drops already
 * handled by a nested dropzone (which called preventDefault) are ignored, so
 * this composes with the FileDrop button rather than double-handling.
 */
export function useWindowDrop(enabled: boolean, onFile: (file: File) => void): boolean {
  const [dragging, setDragging] = useState(false);
  const depth = useRef(0);

  useEffect(() => {
    if (!enabled) {
      depth.current = 0;
      setDragging(false);
      return;
    }
    const onEnter = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      depth.current += 1;
      setDragging(true);
    };
    const onLeave = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      depth.current = Math.max(0, depth.current - 1);
      if (depth.current === 0) setDragging(false);
    };
    const onOver = (e: DragEvent) => {
      if (hasFiles(e)) e.preventDefault();
    };
    const onDrop = (e: DragEvent) => {
      if (!hasFiles(e)) return;
      depth.current = 0;
      setDragging(false);
      if (e.defaultPrevented) return; // a nested dropzone already took it
      e.preventDefault();
      const file = e.dataTransfer?.files?.[0];
      if (file) onFile(file);
    };

    window.addEventListener("dragenter", onEnter);
    window.addEventListener("dragleave", onLeave);
    window.addEventListener("dragover", onOver);
    window.addEventListener("drop", onDrop);
    return () => {
      window.removeEventListener("dragenter", onEnter);
      window.removeEventListener("dragleave", onLeave);
      window.removeEventListener("dragover", onOver);
      window.removeEventListener("drop", onDrop);
    };
  }, [enabled, onFile]);

  return dragging;
}
