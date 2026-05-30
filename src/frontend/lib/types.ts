import type { MorphicResultItem } from "./morphic"

export interface Message {
  role: "user" | "assistant" | "system" | "tool"
  content: string
  tool_call_id?: string
  tool_calls?: unknown[]
  morphic_result?: MorphicSearchResult
}

export interface ChatRequest {
  model: string
  user_msg: string
  system_prompt?: string
  temperature?: number
  top_p?: number
  reasoning_effort?: string | null
  img_base64?: string | null
  downscale_images?: boolean
  messages?: Message[]
}

export interface ModelsResponse {
  all: string[]
  ollama: string[]
  gemini: string[]
  openai: string[]
}

export interface ChatHistoriesResponse {
  histories: Record<string, string[]>
}

export type Provider = "Ollama" | "Gemini" | "OpenAI"

export interface ResearchRequest {
  query: string
  fast_model: string
  smart_model: string
  strategic_model: string
  depth: number
  breadth: number
  reasoning_effort?: string | null
  report_type: string
}

export interface ResearchResult {
  report: string
  sources: string[]
  costs: number
}

export interface MorphicSearchParams {
  query: string
  searchDepth: "quick" | "adaptive"
  model?: string
}

export interface MorphicSearchResult {
  query: string
  sources: MorphicResultItem[]
  images: string[]
  citationMap?: Record<string, MorphicResultItem>
}

export interface OCRRequest {
  img_base64: string
  model: string
}