import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/** True for key events typed into a Neovim terminal (`data-neovim-terminal`), which owns every key while focused. */
export function isNeovimKeyEvent(event: Event): boolean {
  return event.target instanceof Element && event.target.closest("[data-neovim-terminal]") !== null
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

/** Copy of `set` with `value` removed when present, added otherwise. */
export function toggled<T>(set: ReadonlySet<T>, value: T): Set<T> {
  const next = new Set(set)
  if (!next.delete(value)) next.add(value)
  return next
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
