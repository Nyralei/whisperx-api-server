import { useRef, useState } from "react";

import { validateFile } from "../lib/files";
import { fmtBytes } from "../lib/format";
import { buttonSecondaryClass } from "./ui";

export function FileDrop(props: {
  file: File | null;
  onSelect: (file: File | null) => void;
  disabled?: boolean;
  maxUploadBytes?: number | null;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [warning, setWarning] = useState<string | null>(null);

  const takeFile = (file: File | undefined) => {
    if (!file) return;
    const verdict = validateFile(file, props.maxUploadBytes ?? null);
    if (!verdict.ok) {
      setError(verdict.reason);
      setWarning(null);
      props.onSelect(null);
      return;
    }
    setError(null);
    setWarning(verdict.warning);
    props.onSelect(file);
  };

  return (
    <div>
      <button
        type="button"
        disabled={props.disabled}
        onDragOver={(e) => {
          e.preventDefault();
          if (!props.disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (!props.disabled) takeFile(e.dataTransfer.files?.[0]);
        }}
        onClick={() => inputRef.current?.click()}
        className={`flex min-h-40 w-full cursor-pointer flex-col items-center justify-center gap-3 rounded-sm border-2 border-dashed p-6 text-center transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-focus ${
          dragging
            ? "border-accent bg-accent-subtle"
            : "border-border-strong bg-surface hover:border-accent-text"
        } ${props.disabled ? "cursor-not-allowed opacity-60" : ""}`}
      >
        <svg
          aria-hidden="true"
          viewBox="0 0 24 24"
          className="h-8 w-8 text-fg-subtle"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M12 16.5V4m0 0L7.5 8.5M12 4l4.5 4.5M4 19.5h16"
          />
        </svg>
        {props.file ? (
          <span className="text-sm">
            <span className="block font-medium text-fg">{props.file.name}</span>
            <span className="block tabular-nums text-fg-muted">
              {fmtBytes(props.file.size)}
            </span>
          </span>
        ) : (
          <span className="text-sm text-fg-muted">
            drop audio / video here — or click to browse
          </span>
        )}
        <span
          aria-hidden="true"
          className={`${buttonSecondaryClass} pointer-events-none`}
        >
          {props.file ? "Choose a different file" : "Browse files"}
        </span>
      </button>
      <input
        ref={inputRef}
        type="file"
        accept="audio/*,video/*"
        className="sr-only"
        aria-hidden="true"
        tabIndex={-1}
        onChange={(e) => {
          takeFile(e.target.files?.[0]);
          e.target.value = "";
        }}
      />
      {error ? (
        <p role="alert" className="mt-2 text-sm text-danger-text">
          {error}
        </p>
      ) : null}
      {warning ? <p className="mt-2 text-sm text-warn-text">{warning}</p> : null}
    </div>
  );
}
