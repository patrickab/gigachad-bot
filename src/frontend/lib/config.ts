const defaultApiBase = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8001/api"
let apiBase = defaultApiBase

export function getApiBase(): string {
  return apiBase
}

/** Set by the Tauri bootstrap before application providers mount. */
export function setApiBase(base: string): void {
  const url = new URL(base)
  apiBase = url.toString().replace(/\/$/, "")
}

export const REASONING_LEVELS = ["none", "low", "medium", "high"] as const
export type ReasoningLevel = (typeof REASONING_LEVELS)[number]

export const STORAGE_KEY_THEME = "theme"
export const STORAGE_KEY_TRANSPARENT_BG = "transparentBg"

export const CHROME_UNIT_PX = 60
