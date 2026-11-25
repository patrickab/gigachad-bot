import type { AgentMessage, AssistantBlock, AssistantMessage, CustomMessage, ImageContent, TextContent, ToolResultMessage, ToolRun, UserMessage } from "./types"
import type { PiSessionEntries, PiSessionEntry } from "@/lib/backend/api"

/** What entry edits left on a confirmed message: a `context_edit` (omitted from or replaced in the model's context) and a `label`. */
export interface MessageMarks {
  context?: "omitted" | "replaced"
  label?: string
}

/**
 * What the transcript shows: confirmed messages (persisted entries), then live ones not yet confirmed, the streaming reply and live tool state.
 *
 * Live messages (`unconfirmed`, `optimistic`, `streaming`) are rendered after the confirmed ones and dropped once a snapshot or resync brings their entry (M2).
 */
export interface Transcript {
  /** The active branch's persisted messages, from `attached` and `resync`, `context_edit` replacements applied. */
  messages: AgentMessage[]
  /** Entry id of each confirmed message, parallel to `messages`. */
  entryIds: string[]
  /** Entry edits by target entry id. */
  marks: Record<string, MessageMarks>
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

export const EMPTY_TRANSCRIPT: Transcript = { messages: [], entryIds: [], marks: {}, unconfirmed: [], optimistic: null, streaming: null, runs: {}, running: false, queue: { steering: [], followUp: [] } }

type PiEvent = { type: string; [field: string]: unknown }

/** A persisted session entry; only the types that become transcript messages are read. */
export type SessionEntry = PiSessionEntry
/** `get_entries` data, live from a session or read from its file (`fetchPiSessionEntries`). */
export type SessionEntries = PiSessionEntries

/** The message an entry shows as, if any: compactions and branch summaries become their summary messages like in Pi's context. */
function entryMessage(entry: SessionEntry): AgentMessage | null {
  const timestamp = Date.parse(entry.timestamp)
  switch (entry.type) {
    case "message": return entry.message as AgentMessage
    case "compaction": return { role: "compactionSummary", summary: String(entry.summary ?? ""), tokensBefore: Number(entry.tokensBefore ?? 0), timestamp }
    case "branch_summary": return { role: "branchSummary", timestamp }
    case "custom_message": return { role: "custom", customType: String(entry.customType), content: entry.content as CustomMessage["content"], display: Boolean(entry.display), details: entry.details, timestamp }
    default: return null
  }
}

/** The active branch's entries, root to leaf. */
function branchEntries({ entries, leafId }: SessionEntries): SessionEntry[] {
  const byId = new Map(entries.map((entry) => [entry.id, entry]))
  const branch: SessionEntry[] = []
  let entry = leafId ? byId.get(leafId) : undefined
  // Bounded by the entry count, so a broken parent chain cannot loop
  while (entry && branch.length < entries.length) {
    branch.push(entry)
    entry = entry.parentId ? byId.get(entry.parentId) : undefined
  }
  return branch.reverse()
}

/** The confirmed part of a transcript. */
type Confirmed = Pick<Transcript, "messages" | "entryIds" | "marks">

const NO_CONFIRMED: Confirmed = { messages: [], entryIds: [], marks: {} }

/** `message` with its content replaced by a `context_edit`; assistant and tool result content is always a block array. */
function withContent(message: AgentMessage, replacement: string | (TextContent | ImageContent)[]): AgentMessage {
  if (!("content" in message)) return message
  const blocks = typeof replacement === "string" ? [{ type: "text" as const, text: replacement }] : replacement
  if (message.role === "assistant") return { ...message, content: blocks.filter((block): block is TextContent => block.type === "text") }
  if (message.role === "toolResult") return { ...message, content: blocks }
  return { ...message, content: replacement } as AgentMessage
}

/**
 * `confirmed` with `entries` appended in order: messages join, `context_edit` and `label` apply to their (earlier) target.
 *
 * - Every edit applies in turn, so the latest one on the branch wins like in Pi
 * - An omitted message stays shown, marked; a replaced one shows its replacement
 */
function appendEntries(confirmed: Confirmed, entries: SessionEntry[]): Confirmed {
  if (entries.length === 0) return confirmed
  const messages = [...confirmed.messages]
  const entryIds = [...confirmed.entryIds]
  const marks = { ...confirmed.marks }
  for (const entry of entries) {
    if (entry.type === "context_edit" || entry.type === "label") {
      const targetId = String(entry.targetId)
      const { label, ...mark } = marks[targetId] ?? {}
      if (entry.type === "label") {
        marks[targetId] = typeof entry.label === "string" && entry.label ? { ...mark, label: entry.label } : mark
        continue
      }
      const index = entryIds.lastIndexOf(targetId)
      const replacement = entry.replacement as string | (TextContent | ImageContent)[] | null | undefined
      if (replacement != null && index >= 0) messages[index] = withContent(messages[index], replacement)
      marks[targetId] = { ...(label ? { label } : {}), context: replacement == null ? "omitted" : "replaced" }
      continue
    }
    const message = entryMessage(entry)
    if (!message) continue
    messages.push(message)
    entryIds.push(entry.id)
  }
  return { messages, entryIds, marks }
}

/** A settled transcript of persisted `entries`, as an `attached` snapshot of an idle session would give. */
export function transcriptOfEntries(entries: SessionEntries): Transcript {
  return transcriptReducer(EMPTY_TRANSCRIPT, { kind: "snapshot", entries, running: false, queue: { steering: [], followUp: [] } })
}

/** The active branch's messages, root to leaf, including what compactions summarized (the transcript shows the whole history). */
export function messagesOfEntries(entries: SessionEntries): AgentMessage[] {
  return appendEntries(NO_CONFIRMED, branchEntries(entries)).messages
}

/** Identity of a message across live events and persisted entries: `role:timestamp[:toolCallId]`. */
function messageKey(message: AgentMessage): string {
  return `${message.role}:${message.timestamp}:${message.role === "toolResult" ? (message as ToolResultMessage).toolCallId : ""}`
}

/**
 * Whether `stored` (confirmed, or an earlier live message) is the live message `live`.
 *
 * A `custom_message` entry is stamped when Pi persists it, not with the message's own timestamp, so a custom message
 * matches a stored one of the same type and content stamped at or after it
 */
function sameMessage(stored: AgentMessage, live: AgentMessage): boolean {
  if (messageKey(stored) === messageKey(live)) return true
  if (stored.role !== "custom" || live.role !== "custom") return false
  return stored.customType === live.customType && stored.timestamp >= live.timestamp
    && JSON.stringify(stored.content) === JSON.stringify(live.content) && JSON.stringify(stored.details) === JSON.stringify(live.details)
}

function hasMessage(messages: AgentMessage[], message: AgentMessage): boolean {
  for (let index = messages.length - 1; index >= 0; index--) if (sameMessage(messages[index], message)) return true
  return false
}

/** Roles whose entries carry the message's own timestamp, so it orders them against live messages. */
const STAMPED_ROLES = new Set<AgentMessage["role"]>(["user", "assistant", "toolResult"])

/**
 * The live messages `messages` does not hold yet.
 *
 * One whose entry arrived goes, and so does every one before it or older than a confirmed message: entries are
 * persisted in order, so a message the session holds nothing of by then is never coming (another branch, a lost write)
 */
function stillUnconfirmed(unconfirmed: AgentMessage[], messages: AgentMessage[]): AgentMessage[] {
  let latest = -Infinity
  for (const message of messages) if (STAMPED_ROLES.has(message.role) && message.timestamp > latest) latest = message.timestamp
  let lastConfirmed = -1
  unconfirmed.forEach((message, index) => { if (hasMessage(messages, message)) lastConfirmed = index })
  return unconfirmed.slice(lastConfirmed + 1).filter((message) => message.timestamp >= latest)
}

function promptsWithText(messages: AgentMessage[], text: string): number {
  return messages.filter((message) => message.role === "user" && textOf(message.content) === text).length
}

/** Replace the confirmed state, dropping the live messages it now holds. The optimistic prompt goes once a new prompt with its text is confirmed. */
function withConfirmed(transcript: Transcript, confirmed: Confirmed): Transcript {
  const { messages } = confirmed
  const unconfirmed = stillUnconfirmed(transcript.unconfirmed, messages)
  const pending = transcript.optimistic
  const pendingText = pending && textOf(pending.content)
  const promptConfirmed = pendingText !== null && promptsWithText(messages, pendingText) > promptsWithText(transcript.messages, pendingText)
  return { ...transcript, ...confirmed, unconfirmed, optimistic: promptConfirmed ? null : pending }
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
  /** Records buffered for one animation frame (`coalescer.ts`), applied in order. */
  | { kind: "events"; events: PiEvent[] }
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
    case "events": return action.events.reduce(reduceTranscript, transcript)
    case "snapshot": {
      const confirmed = withConfirmed(transcript, appendEntries(NO_CONFIRMED, branchEntries(action.entries)))
      return { ...confirmed, unconfirmed: [], streaming: null, runs: {}, running: action.running, queue: action.queue }
    }
    case "resync": {
      const { mode, entries, leafId } = action.resync
      const confirmed = mode === "append" ? appendEntries(transcript, entries) : appendEntries(NO_CONFIRMED, branchEntries({ entries, leafId }))
      return { ...withConfirmed(transcript, confirmed), running: action.running ?? transcript.running, queue: action.queue ?? transcript.queue }
    }
    case "prompt": return { ...transcript, optimistic: action.prompt }
    case "stream_changed": return { ...transcript, unconfirmed: [], optimistic: null, streaming: null, runs: {} }
    case "replay_start": return { ...transcript, streaming: null }
  }
}

/** Confirmed messages, then the live ones, then the optimistic prompt: what the transcript renders. */
export function shownMessages(transcript: Pick<Transcript, "messages" | "unconfirmed" | "optimistic">): AgentMessage[] {
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

/** Most lines of partial tool output kept while a tool runs. */
export const LIVE_OUTPUT_MAX_LINES = 200

/** A partial result with each text block cut to its last `LIVE_OUTPUT_MAX_LINES` lines, so a chatty tool stays bounded. */
function tailResult(result: ToolRun["result"]): ToolRun["result"] {
  if (!result || !Array.isArray(result.content)) return result
  return {
    ...result,
    content: result.content.map((block) => {
      if (block.type !== "text") return block
      const lines = block.text.split("\n")
      return lines.length > LIVE_OUTPUT_MAX_LINES ? { ...block, text: lines.slice(-LIVE_OUTPUT_MAX_LINES).join("\n") } : block
    }),
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
      // A late chunk never reopens a call that already ended: the final result wins
      if (transcript.runs[id]?.running === false) return transcript
      return { ...transcript, runs: { ...transcript.runs, [id]: { ...transcript.runs[id], running: true, isError: false, result: tailResult(event.partialResult as ToolRun["result"]) } } }
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
export function toolRunsOf(transcript: Pick<Transcript, "messages" | "unconfirmed" | "optimistic" | "runs">): Record<string, ToolRun> {
  const runs: Record<string, ToolRun> = {}
  for (const message of shownMessages(transcript)) {
    if (message.role !== "toolResult") continue
    const result = message as ToolResultMessage
    runs[result.toolCallId] = { running: false, isError: result.isError, result: { content: result.content, details: result.details } }
  }
  // A live run still "running" next to its committed result is stale (resync after the end was missed): the committed result wins, keeping the duration
  for (const [id, run] of Object.entries(transcript.runs)) {
    runs[id] = run.running && runs[id] ? { ...runs[id], durationMs: run.durationMs } : run
  }
  return runs
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
