import { type ButtonHTMLAttributes, type ReactNode, useState } from "react";

export function Field(props: {
  id: string;
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1">
      <label htmlFor={props.id} className="text-xs font-medium text-fg-muted">
        {props.label}
      </label>
      {props.children}
      {props.hint ? <p className="text-[11px] text-fg-subtle">{props.hint}</p> : null}
    </div>
  );
}

export function Toggle(props: {
  id: string;
  label: string;
  checked: boolean;
  onChange: (value: boolean) => void;
  disabled?: boolean;
  hint?: string;
}) {
  return (
    <div className="flex items-start gap-2">
      <input
        id={props.id}
        type="checkbox"
        checked={props.checked}
        disabled={props.disabled}
        onChange={(e) => props.onChange(e.target.checked)}
        className="mt-0.5 h-4 w-4 rounded-none border-border-strong accent-[var(--color-accent)] disabled:opacity-40"
      />
      <label
        htmlFor={props.id}
        className={`text-sm ${props.disabled ? "text-fg-subtle" : "text-fg-muted"}`}
      >
        {props.label}
        {props.hint ? (
          <span className="block text-[11px] text-fg-subtle">{props.hint}</span>
        ) : null}
      </label>
    </div>
  );
}

export const inputClass =
  "w-full rounded-sm border border-border-strong bg-surface px-2.5 py-1.5 text-sm text-fg " +
  "placeholder:text-fg-subtle focus:outline-none focus-visible:ring-2 focus-visible:ring-focus " +
  "disabled:bg-surface-2 disabled:text-fg-subtle";

export const buttonPrimaryClass =
  "inline-flex items-center justify-center gap-2 rounded-sm bg-accent px-4 py-2 text-sm font-semibold " +
  "text-on-accent hover:bg-accent-strong focus:outline-none focus-visible:ring-2 focus-visible:ring-focus " +
  "focus-visible:ring-offset-2 ring-offset-surface " +
  "disabled:cursor-not-allowed disabled:bg-fill disabled:text-fg-subtle";

export const buttonSecondaryClass =
  "inline-flex items-center justify-center gap-1.5 rounded-sm border border-border-strong bg-surface-2 px-3 py-1.5 " +
  "text-sm font-medium text-fg-muted hover:bg-fill hover:text-fg focus:outline-none focus-visible:ring-2 " +
  "focus-visible:ring-focus disabled:cursor-not-allowed disabled:opacity-50";

/**
 * Single source of truth for buttons. Wraps the shared class strings so call sites
 * pick a `variant` instead of repeating them; extra `className` is appended.
 */
export function Button({
  variant = "primary",
  className,
  type,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary";
}) {
  const base = variant === "primary" ? buttonPrimaryClass : buttonSecondaryClass;
  return (
    <button
      type={type ?? "button"}
      className={className ? `${base} ${className}` : base}
      {...rest}
    />
  );
}

export function Spinner(props: { className?: string }) {
  return (
    <span
      aria-hidden="true"
      className={`inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-current border-t-transparent ${props.className ?? ""}`}
    />
  );
}

export function CopyButton(props: { value: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <>
      <button
        type="button"
        onClick={() => {
          void navigator.clipboard
            .writeText(props.value)
            .then(() => {
              setCopied(true);
              window.setTimeout(() => setCopied(false), 1200);
            })
            .catch(() => {
              /* clipboard unavailable (insecure context) */
            });
        }}
        className="rounded-sm px-1 text-fg-subtle hover:text-fg focus:outline-none focus-visible:ring-1 focus-visible:ring-focus"
      >
        {copied ? "copied" : (props.label ?? "copy")}
      </button>
      <span role="status" aria-live="polite" className="sr-only">
        {copied ? "Copied to clipboard" : ""}
      </span>
    </>
  );
}
