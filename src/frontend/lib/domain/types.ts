import { isRecord } from "@/lib/utils"

export interface Usage {
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
}

export interface Attachment {
  name: string
  mime: string
  url: string
  /** Smaller preview of an image (`url` is the full one), shown in the message tiles. */
  thumbUrl?: string
  content?: string
  parsedMd?: string
  /** An uploaded PDF's own text while the user has not yet picked it or a MinerU extraction (`parsedMd` filled at send). */
  textLayer?: string
  active: boolean
  /** Live reference to its source document (mount path or app document key): no upload copy exists, it is re-read on every send. */
  vaultPath?: string
  /** The reference's source could not be found on the last send, the content is the last known one. */
  missing?: boolean
}

/** The tools this build can actually offer the model. */
export type ToolName = "web_search" | "deep_research" | "sandbox_plot" | "diagram" | "mindmap" | "latex_ocr" | "workspace_agent" | "start_learn_session" | "memory"

/** A name read back from a saved chat. Widened past `ToolName` so old chats still render,
 *  while only `ToolName` may be sent as an offered tool. */
type ToolCallName = ToolName | (string & {})

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

/** Any JSON object passes: Plotly validates the interior, so the guard only rejects non-objects. */
export function isPlotFigure(value: unknown): value is PlotFigure {
  return isRecord(value)
}

/** Per-tool detail, flattened: one record holds whichever tool ran, and saved chats may carry
 *  keys this build no longer writes. */
interface ToolCallDetail {
  costs?: number
  figure?: PlotFigure
  brief?: string
  script?: string
  mermaid?: string
  mindmap?: string
  [key: string]: unknown
}

interface ToolStage {
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
 *  arguments or status, so the browser keeps the arguments it already has and derives status. */
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

interface SandboxAssetRef {
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
}

export interface ChatRequest {
  model: string
  chat_id: string
  user_msg: string
  system_prompt?: string
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
interface ToolOptions {
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
  /** Persisted selector-tab display order, e.g. `["Ollama", "Pi", "OpenAI"]`. Tabs missing from it fall to the end. */
  tab_order: string[]
  /** Catalog selectors the Pi gateway does not list, so a send would fail: hidden from the pick list, kept under Manage models. */
  unavailable: string[]
}

/** Ordered `reasoning_effort` values the model accepts, `none` = off, empty = can't reason. */
export interface ReasoningLevels { levels: string[] }

export interface ModelDefaults { default_model: string; small_model: string; vision_model: string; sandbox_model: string }

export interface ModelProvider {
  label: string
  provider_id: string
  /** Model name without the provider prefix; the selector sends `provider_id/model`. */
  models: string[]
  /** Where the models came from. `"pi"` rows were picked from Pi's logins through pi-host's gateway. */
  source?: string | null
}

interface PiModel {
  /** Gateway model id, already provider-qualified, e.g. `anthropic/claude-opus-5`. */
  id: string
  name: string
  vision: boolean
  context_window: number | null
  /** Pi's thinking ladder for the model, empty when it cannot reason. */
  thinking_levels: string[]
}

interface PiProvider {
  id: string
  label: string
  models: PiModel[]
}

export interface PiCatalog {
  /** pi-host has run on this machine (its gateway token exists). */
  installed: boolean
  /** Its gateway answered, so these models are callable right now. */
  online: boolean
  gateway: string
  /** Provider id to build a selector with: `${provider_id}/${model.id}`. */
  provider_id: string
  providers: PiProvider[]
  error: string | null
}

export interface BackendConfig {
  small_model: string
  vision_model: string
  default_model: string
  neovim_available: boolean
  /** This user's animated app background setting. */
  animated_background: boolean
  /** Whether this login may use the host-only factory. */
  host_access: boolean
  /** LearnHarness's address, the source of the learn frame. Null for logins without host access. */
  learnharness_url: string | null
}

interface ProviderUsageWindow {
  id: "5h" | "7d" | "30d"
  percent: number
  resets_in_minutes: number | null
}

export interface ProviderUsageReport {
  provider: string
  account: string | null
  no_limits: boolean
  /** pi-host could not read this account's usage this time, e.g. a rate-limited usage API. */
  unavailable?: boolean
  windows: ProviderUsageWindow[]
}

export interface PromptMeta { slug: string; name: string; includes: string[] }

/** What a Pi tab shows: a session in `repo` (a checkout name under `~/git`, as `/api/pi/repos` lists it), `sessionId` null until its first message creates one. */
export interface PiView {
  sessionId: string | null
  repo: string
}

/** What a learn tab shows in LearnHarness's shared frame. */
export type LearnView =
  | { kind: "home" }
  /** A new session, teaching unless `sessionKind` says practice (LearnHarness's Practice entry). */
  | { kind: "new"; sessionKind?: "teach" | "practice" }
  | { kind: "session"; sessionId: string }

/** A LearnHarness learn session and the project it is grouped under (`GET /api/learn/sessions`). */
export interface LearnSession {
  id: string
  title: string
  kind: string | null
  /** A teach turn is running. */
  busy: boolean
  /** The sidebar row's state (`GET /api/learn/sessions` always sends it): a context compaction runs. */
  compacting?: boolean
  /** A question waits for the learner. */
  pending_question?: boolean
  /** Practice: a posed problem waits for the learner. */
  pending_problem?: boolean
  /** When the session last changed (ISO time), null for a fresh one. */
  updated_at?: string | null
  project_slug: string | null
}
