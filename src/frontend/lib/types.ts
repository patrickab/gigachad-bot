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
