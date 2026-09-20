import type { WebSearchResultItem, WebSearchVideo } from "./webSearch"

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
  active: boolean
  /** Live reference to a FileVault file — no upload copy exists; edits write back to this path. */
  vaultPath?: string
}

export interface VaultFile {
  path: string
  name: string
}

export interface VaultNode {
  name: string
  path: string
  type: "vault" | "folder" | "file"
  /** Project slug this root is mounted to; null/absent = global Vaults section. */
  project?: string | null
  children?: VaultNode[]
}

export interface ProjectDocument {
  path: string
  name: string
  mime: string
}

/** A live Architecture Graph reference promoted to a chat's active context. */
export interface ArchitectureGraphContextReference {
  path: string
}

/** The tools this build can actually offer the model. */
export type ToolName = "web_search" | "deep_research" | "sandbox_plot" | "workspace_agent"

/** A name read back from a saved chat. Widened past `ToolName` so old chats still render,
 *  while only `ToolName` may be sent as an offered tool. */
export type ToolCallName = ToolName | (string & {})

/** A cited source as it reaches the browser: the backend strips the model-only evidence
 *  `content` from every source before emitting the result event. */
export interface ToolSource {
  label?: string
  title: string
  url: string
}

/** Plotly figure JSON exactly as `fig.to_json()` writes it; Plotly validates the interior. */
export interface PlotFigure {
  data?: unknown[]
  layout?: Record<string, unknown>
  frames?: unknown[]
}

export interface WebSearchDetail {
  /** The planned query actually sent to Brave, not the model's raw argument. */
  search_query?: string
}

export interface DeepResearchDetail {
  costs?: number
  report?: string
}

export interface SandboxPlotDetail {
  figure?: PlotFigure
  brief?: string
  script?: string
}

/** Per-tool detail, flattened: one record holds whichever tool ran, and saved chats may carry
 *  keys this build no longer writes. */
export type ToolCallDetail = WebSearchDetail & DeepResearchDetail & SandboxPlotDetail & Record<string, unknown>

export interface ToolStage {
  id: string
  label: string
  status: "running" | "done" | "error"
  started_at: number
  duration: number
}

/** `tool_progress` SSE payload: an updated timeline for a running tool. */
export interface ToolCallProgress {
  id: string
  stages: ToolStage[]
}

/** `tool_call` SSE payload: the call the model made, before it runs. */
export interface ToolCallStarted {
  id: string
  name: ToolName
  arguments: Record<string, unknown>
}

/** `tool_result` SSE payload: every field is always present on the wire. It carries no
 *  arguments or status — the browser keeps the arguments it already has and derives status. */
export interface ToolCallResult {
  id: string
  name: ToolName
  /** Short result headline, e.g. "8 sources". */
  summary: string
  sources: ToolSource[]
  detail: ToolCallDetail
  error: string | null
  sandbox: SandboxToolResultRecord | null
}

/** A tool invocation the model made on its own, rendered as its own chat element.
 *  Unlike a QA pair it has no user turn: it is produced mid-answer and belongs to the
 *  assistant message it interrupted, so pair indexing and branching stay untouched. */
export interface ToolCallRecord {
  id: string
  name: ToolCallName
  arguments: Record<string, unknown>
  status: "running" | "done" | "error"
  summary?: string
  sources?: ToolSource[]
  detail?: ToolCallDetail
  error?: string | null
  sandbox?: SandboxToolResultRecord | null
  stages?: ToolStage[]
}

export interface SandboxAssetRef {
  sha256: string
  media_type: string
  size_bytes: number
  asset_path: string
}

export interface SandboxOutputRecord {
  display_id: string | null
  title: string | null
  mime_bundle: Record<string, SandboxAssetRef>
  text: string | null
}

export interface SandboxToolResultRecord {
  status: "completed" | "failed" | "cancelled"
  manifest_id: string | null
  workspace_changed: boolean
  outputs: SandboxOutputRecord[]
}

export interface Message {
  role: "user" | "assistant" | "system" | "tool"
  content: string
  attachments?: Attachment[]
  hiddenContent?: string
  tool_call_id?: string
  tool_calls?: ToolCallRecord[]
  search_result?: WebSearchResult
  research_steps?: ResearchTraceStep[]
  research_progress?: ResearchTraceProgress
}

export interface ChatRequest {
  model: string
  chat_id: string
  user_msg: string
  system_prompt?: string
  temperature?: number
  reasoning_effort?: string | null
  img_paths: string[]
  downscale_images?: boolean
  messages?: { role: string; content: string }[]
  project_slug?: string | null
  /** Tool names the model may call this turn. Empty or absent means a tool-free completion. */
  tools?: ToolName[]
  tool_options?: ToolOptions
}

/** Tool settings the browser already owns as tab config, forwarded per request. */
export interface ToolOptions {
  search_system_instructions?: string
  search_domain?: string
  research_fast_model?: string
  research_smart_model?: string
  research_strategic_model?: string
  research_depth?: number
  research_breadth?: number
  research_reasoning?: string | null
  research_report_type?: string
}

export interface ModelsResponse {
  ollama: string[]
  providers: ModelProvider[]
  defaults: ModelDefaults
  /** Persisted selector-tab display order, e.g. `["Ollama", "OMP", "OpenAI"]`. Tabs missing from it fall to the end. */
  tab_order: string[]
}

export interface ReasoningSupport { supports_reasoning: boolean }

export interface ModelDefaults { default_model: string; small_model: string; vision_model: string; memory_model: string; omp_model: string }

export interface ModelProvider {
  label: string
  litellm_id: string
  /** Model name without the provider prefix; the selector sends `litellm_id/model`. */
  models: string[]
  /** Where the models came from. `"omp"` rows were picked from OMP's logins. */
  source?: string | null
}

export interface OmpModel {
  /** Gateway model id, already provider-qualified, e.g. `anthropic/claude-opus-5`. */
  id: string
  name: string
  vision: boolean
  context_window: number | null
}

export interface OmpProvider {
  id: string
  label: string
  models: OmpModel[]
}

export interface OmpCatalog {
  /** OMP is present on this machine. */
  installed: boolean
  /** Its gateway answered, so these models are callable right now. */
  online: boolean
  gateway: string
  /** LiteLLM prefix to build a selector with: `${litellm_id}/${model.id}`. */
  litellm_id: string
  providers: OmpProvider[]
  error: string | null
}

export interface BackendConfig {
  small_model: string
  vision_model: string
  memory_model: string
  default_model: string
  temperature: number
  downscale_images: boolean
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

export interface WebSearchParams {
  query: string
  systemInstructions?: string
  domain?: string
  model?: string
}

export interface WebSearchResult {
  query: string
  sources: WebSearchResultItem[]
  images: string[]
  videos: WebSearchVideo[]
  citationMap?: Record<string, WebSearchResultItem>
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
  category?: string
}

export type MemoryStatus = "pre-existing" | "combined" | "new"

export interface PreviewMemory {
  id: string
  text: string
  category: string
  scope: "global" | "project"
  status?: MemoryStatus
  created_at?: string
  updated_at?: string
}

export interface MemoryExtractResponse {
  review_id: string
  global: ProposedMemory[]
  project: ProposedMemory[] | null
}

export interface MemoryPreviewResponse {
  existing_markdown: string
  revised_markdown: string
  existing_memories: PreviewMemory[]
  revised_memories: PreviewMemory[]
}

export interface CategoryDef {
  name: string
  description: string
}
