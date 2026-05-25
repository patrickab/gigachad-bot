export interface Message {
  role: "user" | "assistant" | "system" | "tool"
  content: string
  tool_call_id?: string
  tool_calls?: unknown[]
}

export interface ChatRequest {
  model: string
  user_msg: string
  system_prompt?: string
  temperature?: number
  top_p?: number
  reasoning_effort?: string | null
  img_base64?: string | null
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

export interface TavilySearchRequest {
  query: string
  num_queries: number
  results_per_query: number
  expander_model: string
}

export interface TavilySearchResult {
  results: { title: string; url: string; content: string; score: number }[]
  queries: string[]
}
