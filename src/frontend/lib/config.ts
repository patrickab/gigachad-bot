export const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8001/api"

export const REASONING_LEVELS = ["none", "low", "medium", "high"] as const
export type ReasoningLevel = (typeof REASONING_LEVELS)[number]

export const STORAGE_KEY_THEME = "theme"

export const CHROME_UNIT_PX = 60
