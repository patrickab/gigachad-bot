import type { MorphicResultItem } from "./morphic"

export interface Usage {
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
}

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
  reasoning_effort?: string | null
  img_base64?: string | null
  downscale_images?: boolean
  messages?: { role: string; content: string }[]
  project_slug?: string | null
}

export interface ModelsResponse {
  all: string[]
  ollama: string[]
  gemini: string[]
  openai: string[]
}

export interface ChatHistoriesResponse {
  files: string[]
  histories: Record<string, string[]>
}

export interface BranchMeta {
  chat_id: string | null
  parent_id: string | null
  branch_message_idx: number | null
  children: BranchChild[]
  qa_count: number
}

export interface BranchChild {
  chat_id: string
  branch_message_idx: number
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

export type KanbanColumnId = "backlog" | "doing" | "done"

export interface KanbanCard {
  id: string
  title: string
  description: string
  state: KanbanColumnId
}

export interface ProjectData {
  name: string
  slug: string
  kanban: KanbanCard[]
  tabs: ProjectTab[]
}

export interface ProjectStateUpdate {
  kanban: KanbanCard[]
  tabs: ProjectTab[]
}

export interface ProjectTab {
  filename: string
  name: string | null
  title: string | null
}

export interface ProjectListItem {
  name: string
  slug: string
  tabs?: ProjectTab[]
}

export interface ProposedMemory {
  id: string
  memory: string
  scope: "global" | "project"
  kind?: string
}

export interface MemoryExtractResponse {
  review_id: string
  global: ProposedMemory[]
  project: ProposedMemory[] | null
}

export interface MemoryComposeResponse {
  revised_content: string
}

export interface MemoryProfileMeta {
  filepath: string
  id: string
  parent_id: string | null
  title: string
  updated_at: string
}
