import { useEffect, useState } from "react";

export type Theme = "light" | "dark";

const STORAGE_KEY = "whisperx-webui-theme";

function systemTheme(): Theme {
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function storedTheme(): Theme | null {
  const v = localStorage.getItem(STORAGE_KEY);
  return v === "light" || v === "dark" ? v : null;
}

function apply(theme: Theme): void {
  const root = document.documentElement;
  root.classList.toggle("dark", theme === "dark");
  // Drives native controls (form widgets, scrollbars, the audio player).
  root.style.colorScheme = theme;
}

/**
 * Light/dark theme with an explicit user choice persisted to localStorage.
 * Until the user picks, the UI follows the OS preference (and reacts to it
 * live). An inline script in index.html applies the same value before paint,
 * so the initial state here matches and there is no flash.
 */
export function useTheme() {
  const [theme, setThemeState] = useState<Theme>(() => storedTheme() ?? systemTheme());

  useEffect(() => {
    apply(theme);
  }, [theme]);

  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => {
      if (!storedTheme()) setThemeState(mq.matches ? "dark" : "light");
    };
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  const setTheme = (next: Theme) => {
    localStorage.setItem(STORAGE_KEY, next);
    setThemeState(next);
  };

  return { theme, setTheme, toggle: () => setTheme(theme === "dark" ? "light" : "dark") };
}
