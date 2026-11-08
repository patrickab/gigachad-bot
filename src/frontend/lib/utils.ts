import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/** True for key events typed into a terminal (`data-terminal`: Neovim, the shell), which owns every key while focused. */
export function isTerminalKeyEvent(event: Event): boolean {
  return event.target instanceof Element && event.target.closest("[data-terminal]") !== null
}

/** Message of `cause` when it is an `Error` with a non-empty message, else `fallback`. */
export function errorMessage(cause: unknown, fallback: string): string {
  return cause instanceof Error && cause.message ? cause.message : fallback
}

/** True when `target` takes typed text: an input, a textarea, a contenteditable element, and a select with `includeSelect`. */
export function isEditableTarget(target: EventTarget | null, { includeSelect = false }: { includeSelect?: boolean } = {}): boolean {
  if (!(target instanceof HTMLElement)) return false
  const tag = target.tagName
  return tag === "INPUT" || tag === "TEXTAREA" || (includeSelect && tag === "SELECT") || target.isContentEditable
}

/** True when Ctrl or Cmd is held, the platform-neutral shortcut modifier. */
export function isModKey(event: { ctrlKey: boolean; metaKey: boolean }): boolean {
  return event.ctrlKey || event.metaKey
}

/** True when Alt is the only modifier held, the modifier of the shell's shortcuts (Alt+T, Alt+A). */
export function isAltOnly(event: { altKey: boolean; ctrlKey: boolean; metaKey: boolean; shiftKey: boolean }): boolean {
  return event.altKey && !event.ctrlKey && !event.metaKey && !event.shiftKey
}

/** Compact time from `thenMs` to `nowMs`: "now", "5m", "3h", "2d". */
export function compactElapsed(thenMs: number, nowMs: number): string {
  const minutes = Math.floor((nowMs - thenMs) / 60_000)
  if (minutes < 1) return "now"
  if (minutes < 60) return `${minutes}m`
  const hours = Math.floor(minutes / 60)
  return hours < 24 ? `${hours}h` : `${Math.floor(hours / 24)}d`
}

/** Copy of `set` with `value` removed when present, added otherwise. */
export function toggled<T>(set: ReadonlySet<T>, value: T): Set<T> {
  const next = new Set(set)
  if (!next.delete(value)) next.add(value)
  return next
}

/** True for a plain object (not null, not an array), its values typed `V`: the shape check before reading parsed JSON. */
export function isRecord<V = unknown>(value: unknown): value is Record<string, V> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

/** Read `blob` as a `data:` URL. */
export function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as string)
    reader.onerror = () => reject(reader.error ?? new Error("Could not read blob"))
    reader.readAsDataURL(blob)
  })
}
