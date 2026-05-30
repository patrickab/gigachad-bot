export const API_BASE = "/api"

export const REASONING_LEVELS = ["none", "low", "medium", "high"] as const
export type ReasoningLevel = (typeof REASONING_LEVELS)[number]

export const DEFAULT_TEMPERATURE = 0.2
export const DEFAULT_TOP_P = 0.95
export const DEFAULT_DOWNSCALE_IMAGES = true
export const DEFAULT_VISION_MODEL = "ollama/qwen3-vl:235b-instruct-cloud"

export const IMAGE_DOWNSCALE_MAX = 2048

export const STORAGE_KEY_THEME = "theme"