/**
 * Pi RPC protocol shapes the prototype reads (`pi --mode rpc`, pi docs `rpc.md`, `json.md`, `message-types.md`).
 *
 * - Subset only: unknown roles, events and fields are tolerated and ignored
 * - `AgentMessage` values arrive in `message_end`, `get_messages` and `agent_end`
 */

export interface TextContent { type: "text"; text: string }
export interface ImageContent { type: "image"; data: string; mimeType: string }
export interface ThinkingContent { type: "thinking"; thinking: string; redacted?: boolean }
export interface ToolCall { type: "toolCall"; id: string; name: string; arguments: Record<string, unknown> }

export type AssistantBlock = TextContent | ThinkingContent | ToolCall

export interface Usage {
  input: number
  output: number
  cacheRead: number
  cacheWrite: number
  totalTokens: number
  cost: { total: number }
}

export interface UserMessage { role: "user"; content: string | (TextContent | ImageContent)[]; timestamp: number }

export interface AssistantMessage {
  role: "assistant"
  content: AssistantBlock[]
  provider: string
  model: string
  usage?: Usage
  stopReason: "pending" | "stop" | "length" | "toolUse" | "error" | "aborted" | "deferred"
  errorMessage?: string
  timestamp: number
}

export interface ToolResultMessage {
  role: "toolResult"
  toolCallId: string
  toolName: string
  content: (TextContent | ImageContent)[]
  details?: unknown
  isError: boolean
  timestamp: number
}

export interface BashExecutionMessage { role: "bashExecution"; command: string; output: string; exitCode?: number; cancelled: boolean; timestamp: number }
/** `details` is the sending extension's own data for its renderer, never sent to the model. */
export interface CustomMessage { role: "custom"; customType: string; content: string | (TextContent | ImageContent)[]; display: boolean; details?: unknown; timestamp: number }
export interface CompactionSummaryMessage { role: "compactionSummary"; summary: string; tokensBefore: number; timestamp: number }

export type AgentMessage =
  | UserMessage
  | AssistantMessage
  | ToolResultMessage
  | BashExecutionMessage
  | CustomMessage
  | CompactionSummaryMessage
  | { role: "system" | "branchSummary"; timestamp: number }

/** Live state of one tool call, keyed by `toolCallId`, from `tool_execution_*` events. */
export interface ToolRun {
  /** Latest partial result while running, the final result once ended. */
  result?: { content: (TextContent | ImageContent)[]; details?: unknown }
  isError: boolean
  running: boolean
  durationMs?: number
}

/** `get_commands` row: extension command, prompt template or skill (`skill:<name>`). */
export interface PiCommand { name: string; description?: string; source: "extension" | "prompt" | "skill" }

export interface PiModel { id: string; name?: string; provider: string; reasoning?: boolean }

/** `get_state` data. */
export interface PiState {
  model: PiModel | null
  thinkingLevel: string
  isStreaming: boolean
  sessionFile?: string
  sessionId: string
  sessionName?: string
}

/** An `extension_ui_request` dialog that blocks pi until answered (`select`, `confirm`, `input`, `editor`). */
export type ExtensionDialog =
  | { id: string; method: "select"; title: string; options: string[] }
  | { id: string; method: "confirm"; title: string; message?: string }
  | { id: string; method: "input"; title: string; placeholder?: string }
  | { id: string; method: "editor"; title: string; prefill?: string }
