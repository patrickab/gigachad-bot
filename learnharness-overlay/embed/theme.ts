import type { LearnTheme } from "@gigachad/chat-ui/learn-protocol";

/** Last theme the shell set, so a reload paints in it before the shell's `hello` arrives (`EmbedMode`). */
export const THEME_STORAGE_KEY = "cf-embed:theme";

/** The parts of `document.documentElement` and `localStorage` the theme switch touches. */
type ThemeRoot = { classList: { toggle(token: string, force: boolean): unknown } };
type ThemeStorage = { setItem(key: string, value: string): void };

/**
 * Show the shell's theme. LearnHarness is dark-only upstream (`<html class="dark">`, its Tailwind `dark:` variant),
 * light is that class removed. `custom/theme/` maps both onto the shell's palette.
 */
export function applyTheme(theme: LearnTheme, root: ThemeRoot = document.documentElement, storage: ThemeStorage | null = safeLocalStorage()) {
  root.classList.toggle("dark", theme === "dark");
  try {
    storage?.setItem(THEME_STORAGE_KEY, theme);
  } catch {
    // Storage refused (quota, privacy mode): the next load paints dark until the shell's hello.
  }
}

function safeLocalStorage(): ThemeStorage | null {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}
