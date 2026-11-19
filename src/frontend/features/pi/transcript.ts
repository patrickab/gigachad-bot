import type { AgentMessage, AssistantBlock, AssistantMessage, CustomMessage, ToolResultMessage, ToolRun, UserMessage } from "./types"

/**
 * What the transcript shows: confirmed messages (persisted entries), then live ones not yet confirmed, the streaming reply and live tool state.
 *
 * Live messages (`unconfirmed`, `optimistic`, `streaming`) are rendered after the confirmed ones and dropped once a snapshot or resync brings their entry (M2).
 */
export interface Transcript {
  /** The active branch's persisted messages, from `attached` and `resync`. */
  messages: AgentMessage[]
  /** Messages a live `message_end` reported whose entry no resync has brought yet. */
  unconfirmed: AgentMessage[]
  /** The prompt this tab sent, shown until Pi reports it. */
  optimistic: UserMessage | null
  /** The assistant message being streamed, null between replies. */
  streaming: AssistantMessage | null
  /** Tool calls executing or finished in this connection, by tool call id. Older ones derive from `toolResult` messages. */
  runs: Record<string, ToolRun>
  /** Between `agent_start` and `agent_settled`: sends steer instead of starting a run. */
  running: boolean
  queue: { steering: string[]; followUp: string[] }
}

export const EMPTY_TRANSCRIPT: Transcript = { messages: [], unconfirmed: [], optimistic: null, streaming: null, runs: {}, running: false, queue: { steering: [], followUp: [] } }

type PiEvent = { type: string; [field: string]: unknown }

/** A persisted session entry (pi docs `session-format.md`); only the types that become transcript messages are read. */
export interface SessionEntry {
  type: string
  id: string
  parentId: string | null
  timestamp: string
  [field: string]: unknown
}

/** `get_entries` data: every entry in append order, all branches, and the current leaf. */
export interface SessionEntries {
  entries: SessionEntry[]
  leafId: string | null
}

/** The message an entry shows as, if any: compactions and branch summaries become their summary messages like in Pi's context. */
function entryMessage(entry: SessionEntry): AgentMessage | null {
  const timestamp = Date.parse(entry.timestamp)
  switch (entry.type) {
    case "message": return entry.message as AgentMessage
    case "compaction": return { role: "compactionSummary", summary: String(entry.summary ?? ""), tokensBefore: Number(entry.tokensBefore ?? 0), timestamp }
    case "branch_summary": return { role: "branchSummary", timestamp }
    case "custom_message": return { role: "custom", customType: String(entry.customType), content: entry.content as CustomMessage["content"], display: Boolean(entry.display), timestamp }
    default: return null
  }
}

/** The active branch's messages, root to leaf, including what compactions summarized (the transcript shows the whole history). */
export function messagesOfEntries({ entries, leafId }: SessionEntries): AgentMessage[] {
  const byId = new Map(entries.map((entry) => [entry.id, entry]))
  const branch: SessionEntry[] = []
  let entry = leafId ? byId.get(leafId) : undefined
  // Bounded by the entry count, so a broken parent chain cannot loop
  while (entry && branch.length < entries.length) {
    branch.push(entry)
    entry = entry.parentId ? byId.get(entry.parentId) : undefined
  }
  return branch.reverse().flatMap((branchEntry) => entryMessage(branchEntry) ?? [])
}

/** Identity of a message across live events and persisted entries: `role:timestamp[:toolCallId]`. */
function messageKey(message: AgentMessage): string {
  return `${message.role}:${message.timestamp}:${message.role === "toolResult" ? (message as ToolResultMessage).toolCallId : ""}`
}

function hasMessage(messages: AgentMessage[], message: AgentMessage): boolean {
  const key = messageKey(message)
  for (let index = messages.length - 1; index >= 0; index--) if (messageKey(messages[index]) === key) return true
  return false
}

function promptsWithText(messages: AgentMessage[], text: string): number {
  return messages.filter((message) => message.role === "user" && textOf(message.content) === text).length
}

/** Replace the confirmed messages, dropping the live messages they now hold. The optimistic prompt goes once a new prompt with its text is confirmed. */
function withConfirmed(transcript: Transcript, messages: AgentMessage[]): Transcript {
  const unconfirmed = transcript.unconfirmed.filter((message) => !hasMessage(messages, message))
  const pending = transcript.optimistic
  const pendingText = pending && textOf(pending.content)
  const confirmed = pendingText !== null && promptsWithText(messages, pendingText) > promptsWithText(transcript.messages, pendingText)
  return { ...transcript, messages, unconfirmed, optimistic: confirmed ? null : pending }
}

/** A host `resync` answer: `append` = entries after the cursor on the current branch, `replace` = the whole current branch. */
export interface ResyncEntries {
  mode: "append" | "replace"
  entries: SessionEntry[]
  leafId: string | null
}

export type TranscriptAction =
  /** One Pi stdout record, live or replayed. */
  | { kind: "event"; event: PiEvent }
  /** An `attached` snapshot: replayed records rebuild the streaming reply and tool runs. */
  | { kind: "snapshot"; entries: SessionEntries; running: boolean; queue: Transcript["queue"] }
  /** A `resync` answer: live state stays for the replay that follows. `running` and `queue` from its `state` and `ui` when sent. */
  | { kind: "resync"; resync: ResyncEntries; running?: boolean; queue?: Transcript["queue"] }
  /** The prompt this tab sent, or null when Pi refused it. */
  | { kind: "prompt"; prompt: UserMessage | null }
  /** A new `streamId`: the live state belongs to a stream that is gone. */
  | { kind: "stream_changed" }
  /** A replay re-sends the current reply from its `message_start`. */
  | { kind: "replay_start" }

/** The transcript after one action from the session hook. */
export function transcriptReducer(transcript: Transcript, action: TranscriptAction): Transcript {
  switch (action.kind) {
    case "event": return reduceTranscript(transcript, action.event)
    case "snapshot": {
      const confirmed = withConfirmed(transcript, messagesOfEntries(action.entries))
      return { ...confirmed, unconfirmed: [], streaming: null, runs: {}, running: action.running, queue: action.queue }
    }
    case "resync": {
      const { mode, entries, leafId } = action.resync
      const messages = mode === "append"
        ? [...transcript.messages, ...entries.flatMap((entry) => entryMessage(entry) ?? [])]
        : messagesOfEntries({ entries, leafId })
      return { ...withConfirmed(transcript, messages), running: action.running ?? transcript.running, queue: action.queue ?? transcript.queue }
    }
    case "prompt": return { ...transcript, optimistic: action.prompt }
    case "stream_changed": return { ...transcript, unconfirmed: [], optimistic: null, streaming: null, runs: {} }
    case "replay_start": return { ...transcript, streaming: null }
  }
}

/** Confirmed messages, then the live ones, then the optimistic prompt: what the transcript renders. */
export function shownMessages(transcript: Transcript): AgentMessage[] {
  const live = transcript.optimistic ? [...transcript.unconfirmed, transcript.optimistic] : transcript.unconfirmed
  return live.length > 0 ? [...transcript.messages, ...live] : transcript.messages
}


interface AssistantMessageEvent {
  type: string
  contentIndex: number
  delta?: string
  content?: string
  id?: string
  toolName?: string
  toolCall?: AssistantBlock
}

function withBlock(message: AssistantMessage, index: number, update: (block: AssistantBlock | undefined) => AssistantBlock): AssistantMessage {
  const content = [...message.content]
  content[index] = update(content[index])
  return { ...message, content }
}

/** Apply one streamed content update to the partial reply (pi docs `json.md`, "Reconstruct streaming messages"). */
function applyStreamUpdate(message: AssistantMessage, update: AssistantMessageEvent): AssistantMessage {
  const index = update.contentIndex
  switch (update.type) {
    case "text_start": return withBlock(message, index, () => ({ type: "text", text: "" }))
    case "text_delta": return withBlock(message, index, (block) => ({ type: "text", text: (block?.type === "text" ? block.text : "") + (update.delta ?? "") }))
    case "text_end": return withBlock(message, index, () => ({ type: "text", text: update.content ?? "" }))
    case "thinking_start": return withBlock(message, index, () => ({ type: "thinking", thinking: "" }))
    case "thinking_delta": return withBlock(message, index, (block) => ({ type: "thinking", thinking: (block?.type === "thinking" ? block.thinking : "") + (update.delta ?? "") }))
    case "thinking_end": return withBlock(message, index, () => ({ type: "thinking", thinking: update.content ?? "" }))
    case "toolcall_start": return withBlock(message, index, () => ({ type: "toolCall", id: update.id ?? "", name: update.toolName ?? "", arguments: {} }))
    case "toolcall_end": return update.toolCall ? withBlock(message, index, () => update.toolCall!) : message
    default: return message
  }
}

/** The transcript after one Pi stdout record. Records it does not know leave it unchanged. */
export function reduceTranscript(transcript: Transcript, event: PiEvent): Transcript {
  switch (event.type) {
    case "agent_start":
      return { ...transcript, running: true }
    case "agent_settled":
      return { ...transcript, running: false, streaming: null }
    case "message_start": {
      const message = event.message as AgentMessage
      return message.role === "assistant" ? { ...transcript, streaming: { ...message, content: [] } } : transcript
    }
    case "message_update": {
      if (!transcript.streaming) return transcript
      return { ...transcript, streaming: applyStreamUpdate(transcript.streaming, event.assistantMessageEvent as AssistantMessageEvent) }
    }
    case "message_end": {
      // No durable entry id yet: the message waits in `unconfirmed` until a resync brings its entry (M2)
      const message = event.message as AgentMessage
      const streaming = message.role === "assistant" ? null : transcript.streaming
      const optimistic = message.role === "user" ? null : transcript.optimistic
      if (hasMessage(transcript.messages, message) || hasMessage(transcript.unconfirmed, message)) return { ...transcript, streaming, optimistic }
      return { ...transcript, unconfirmed: [...transcript.unconfirmed, message], streaming, optimistic }
    }
    case "tool_execution_start":
      return { ...transcript, runs: { ...transcript.runs, [event.toolCallId as string]: { running: true, isError: false } } }
    case "tool_execution_update": {
      const id = event.toolCallId as string
      return { ...transcript, runs: { ...transcript.runs, [id]: { ...transcript.runs[id], running: true, isError: false, result: event.partialResult as ToolRun["result"] } } }
    }
    case "tool_execution_end": {
      const run: ToolRun = { running: false, isError: Boolean(event.isError), result: event.result as ToolRun["result"], durationMs: event.durationMs as number | undefined }
      return { ...transcript, runs: { ...transcript.runs, [event.toolCallId as string]: run } }
    }
    case "queue_update":
      return { ...transcript, queue: { steering: event.steering as string[], followUp: event.followUp as string[] } }
    default:
      return transcript
  }
}

/** Tool state by call id: committed `toolResult` messages, overridden by this connection's live runs (which carry durations). */
export function toolRunsOf(transcript: Transcript): Record<string, ToolRun> {
  const runs: Record<string, ToolRun> = {}
  for (const message of shownMessages(transcript)) {
    if (message.role !== "toolResult") continue
    const result = message as ToolResultMessage
    runs[result.toolCallId] = { running: false, isError: result.isError, result: { content: result.content, details: result.details } }
  }
  return { ...runs, ...transcript.runs }
}

/** One prompt and everything up to the next: `messages[start, end)`, `start` the prompt (or a compaction). */
export interface Turn {
  start: number
  end: number
}

function startsTurn(message: AgentMessage): boolean {
  return message.role === "user" || message.role === "compactionSummary"
}

/** Split the committed messages into turns. Messages before the first prompt (the system message) belong to none. */
export function turnsOf(messages: AgentMessage[]): Turn[] {
  const turns: Turn[] = []
  messages.forEach((message, index) => {
    if (startsTurn(message)) {
      if (turns.length > 0) turns[turns.length - 1].end = index
      turns.push({ start: index, end: messages.length })
    }
  })
  return turns
}

/** Plain text of a user or custom message's content. */
export function textOf(content: string | { type: string; text?: string }[]): string {
  if (typeof content === "string") return content
  return content.flatMap((block) => (block.type === "text" && block.text ? [block.text] : [])).join("\n")
}
