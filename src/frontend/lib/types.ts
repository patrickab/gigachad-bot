import type { MorphicResultItem } from "./morphic"

export interface Attachment {
  name: string
  mime: string
  url: string
  content?: string
  parsedMd?: string

}

export interface Message {
  role: "user" | "assistant" | "system" | "tool"
  content: string
  attachments?: Attachment[]
  hiddenContent?: string
  tool_call_id?: string
  tool_calls?: unknown[]
  morphic_result?: MorphicSearchResult
  research_trace_id?: string
  research_steps?: ResearchTraceStep[]
  research_progress?: ResearchTraceProgress
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

export interface ResearchParams {
  query: string
  fastModel: string
  smartModel: string
  strategicModel: string
  depth: number
  breadth: number
  reasoningEffort: string
  reportType: string
}

export interface ResearchResult {
  report: string
  sources: string[]
  costs: number
}

export interface ResearchTraceStep {
  step: string
  event_type: string
  details: Record<string, unknown>
  timestamp: number
}

export interface ResearchTraceProgress {
  current_depth: number
  total_depth: number
  current_breadth: number
  total_breadth: number
  current_query: string | null
  completed_queries: number
  total_queries: number
}

export interface ResearchTrace {
  run_id: string
  query: string
  started_at: number
  finished_at: number | null
  duration_s: number | null
  steps: ResearchTraceStep[]
  progress: ResearchTraceProgress | null
}

export interface ResearchTraceSummary {
  run_id: string
  query: string
  started_at: number
  finished_at: number | null
  duration_s: number | null
  step_count: number
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

export interface MineruResult {
  filename: string
  output_dir: string
  markdown_content: string
  answer: string | null
  query: string | null
}

export interface MineruBatchResponse {
  results: MineruResult[]
}

export interface StudyProcessRequest {
  markdown: string
  filename: string
  model: string
}

export interface StudyProcessResponse {
  filename: string
  topics: { label: string }[]
  overview: string
  article: string
}