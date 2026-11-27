/**
 * `NEXT_PUBLIC_API_BASE` as an absolute URL.
 *
 * - A path (`/preview-api/8100/api`, factory previews) resolves against the page's origin: only the browser knows the
 *   origin the preview gate is reached at
 * - The server render has no page, it only needs a parseable URL
 */
function resolveApiBase(base: string): string {
  if (!base.startsWith("/")) return base
  return `${typeof window === "undefined" ? "http://127.0.0.1" : window.location.origin}${base}`
}

const apiBase = resolveApiBase(process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8001/api")

export function getApiBase(): string {
  return apiBase
}

export const REASONING_LEVELS = ["none", "low", "medium", "high"] as const

export const STORAGE_KEY_THEME = "theme"
export const STORAGE_KEY_GLASS = "glass"
export const STORAGE_KEY_NEOVIM = "neovim"
