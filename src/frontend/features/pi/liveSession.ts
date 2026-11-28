import { createStore, type Store } from "@/lib/store"
import { BTW_WIDGET_KEY, parseBtwWidget, isBtwCommand, type BtwThread } from "./btw"
import { createTranscriptCoalescer } from "./coalescer"
import { confirmFromTranscript, heldOf, hold, patchHeld, takeBack, takeHeld } from "./outbox"
import { sendPiCommand, sendPiFrame, subscribePiSession } from "./piSocket"
import type { AttachedUi, HostFrame, PiRecord, PiResponse, PiSessionStats } from "./protocol"
import { SessionResync } from "./resync"
import { INSPECT_WIDGET_KEY, inspectCommand, parseInspectReply, type InspectReply, type SubagentRow } from "./subagents"
import { EMPTY_TRANSCRIPT, transcriptReducer, type Transcript } from "./transcript"
import { isDialogRequest, type ExtensionDialog, type ImageContent, type PiCommand, type PiModel, type PiState, type UserMessage } from "./types"

export type ConnectionStatus = { kind: "connecting" } | { kind: "ready" } | { kind: "closed"; reason: string }

export interface Notice { id: string; text: string; tone: "info" | "warning" | "error" }

/** How a message sent while a run streams is queued: delivered after the current tool calls, or once the run ends (F4). */
export type QueueMode = "steer" | "followUp"

export interface LiveSessionState {
  status: ConnectionStatus
  state: PiState | null
  commands: PiCommand[]
  models: PiModel[]
  thinkingLevels: string[]
  stats: PiSessionStats | null
  dialogs: ExtensionDialog[]
  notices: Notice[]
  statusLines: Record<string, string>
  widgets: Record<string, string[]>
  btwThreads: BtwThread[]
  editorText: { text: string } | null
  /** The current run ended in an error Pi does not retry. */
  failed: boolean
}

export type LiveSession = ReturnType<typeof createLiveSession>

/** A run of a session ended, waits on a dialog, or failed (for notifications). */
export interface PiSessionEvent {
  type: "settled" | "waiting" | "failed"
  sessionId: string
  aborted?: boolean
}

interface LiveObserver {
  onStatus: () => void
  onEvent: (event: PiSessionEvent) => void
}

const NOTICE_LIFETIME_MS = 6_000
const INSPECT_TIMEOUT_MS = 15_000
const NO_WIDGETS: Record<string, string[]> = {}
const EMPTY_QUEUE: Transcript["queue"] = { steering: [], followUp: [] }
/** Distinguishes this tab's command ids from other clients' on the same session, their responses reach every attached client. */
const CLIENT_ID = crypto.randomUUID().slice(0, 8)
/** An extension's `notify` asking the GUI to open the path that follows instead of showing a notice (C7, `pi-extensions/files.ts`). */
export const OPEN_FILE_NOTICE_PREFIX = "Open file: "

const INITIAL_STATE: LiveSessionState = {
  status: { kind: "connecting" }, state: null, commands: [], models: [], thinkingLevels: [], stats: null, dialogs: [], notices: [],
  statusLines: {}, widgets: NO_WIDGETS, btwThreads: [], editorText: null, failed: false,
}

const sessions = new Map<string, LiveSession>()
const observers = new Set<LiveObserver>()

/** The live session of `sessionId`, the same object for the page's life. Nothing is attached until a holder asks. */
export function liveSession(sessionId: string): LiveSession {
  let session = sessions.get(sessionId)
  if (!session) sessions.set(sessionId, session = createLiveSession(sessionId))
  return session
}

/** Run state of a session some view holds attached, null otherwise (and until its first `attached`). While it reconnects, the last known state. */
export function liveStatus(sessionId: string): { running: boolean; waiting: boolean; failed: boolean; contextPercent?: number; cost?: number } | null {
  const session = sessions.get(sessionId)
  if (!session) return null
  const { status, state: piState, dialogs, failed, stats } = session.state.get()
  if (piState === null) return null
  const open = status.kind !== "closed"
  return { running: open && session.transcript.get().running, waiting: open && dialogs.length > 0, failed, contextPercent: stats?.contextUsage?.percent ?? undefined, cost: stats?.cost }
}

/** `onStatus` after any change `liveStatus` reads, `onEvent` once per run edge, each after the state it describes is applied. */
export function observeLiveSessions(observer: LiveObserver): () => void {
  observers.add(observer)
  return () => observers.delete(observer)
}

function emit(event: PiSessionEvent) {
  for (const observer of observers) observer.onEvent(event)
}

/** The last assistant message's stop reason of an `agent_end`. */
function stopReasonOf(record: PiRecord): unknown {
  const messages = Array.isArray(record.messages) ? record.messages as { role?: string; stopReason?: unknown }[] : []
  return messages.findLast((message) => message.role === "assistant")?.stopReason
}

/**
 * One Pi session on the page's shared pi-host socket (`piSocket.ts`), the whole RPC protocol in the browser, shared by every view of it.
 *
 * - Attached while held: `attached` sets state, transcript (entries), dialogs, statuses and queue, the replay and live `pi` records fold into it
 * - Catches up with a guarded `resync` (`resync.ts`, M2, M22) on reconnect, tab visible, `online`, `agent_settled`, compaction and every live `message_end`, whose message gets its entry from the resync
 * - A new `streamId` drops the live state (streaming reply, live messages, optimistic prompt) and resyncs from scratch
 * - `send` starts a run (shown as an optimistic prompt until Pi reports it), or queues into the running one as steering or a follow-up (F4)
 * - A message sent during a run is held (`outbox.ts`) until the queue snapshot or the session shows it, flushed in order after a reconnect
 * - `abort` takes the queued messages back (returned for the draft) before stopping, like Pi's Esc (F5)
 * - `runBash` runs `!cmd` (F7), `compact` compacts with optional instructions (F8), `setModel`/`setThinking` switch the session's model and thinking (D10, F10)
 * - `stats` (cost, context use) refresh on attach, after each answer and run, compaction and model switch (F3): the only asker
 * - Run edges (`settled` once per run on `agent_settled`, `waiting` per dialog, `failed` once per run) and `liveStatus` changes reach `observeLiveSessions`
 * - Extension dialogs queue until answered, `notify` shows briefly, `setStatus` keeps a line per key
 * - A `notify` led by `OPEN_FILE_NOTICE_PREFIX` goes to every `onOpenFile` instead of showing
 */
function createLiveSession(sessionId: string) {
  const state = createStore(INITIAL_STATE)
  const transcript = createStore(EMPTY_TRANSCRIPT)
  const pending = new Map<string, (response: PiResponse) => void>()
  const inspects = new Map<string, (reply: InspectReply) => void>()
  const openers = new Set<(path: string) => void>()
  const noticeTimers = new Set<ReturnType<typeof setTimeout>>()
  let nextId = 0
  let holders = 0
  let replaying = false
  let sync: SessionResync | null = null
  let detach: (() => void) | null = null
  /** The current run's `agent_end` reported an abort; `settled` was emitted for it. */
  let aborted = false
  let settled = false

  /** Tell the observers when what `liveStatus` reads changed, not on every patch (a widget, a notice). */
  let lastStatus = ""
  const statusChanged = () => {
    const current = JSON.stringify(liveStatus(sessionId))
    if (current === lastStatus) return
    lastStatus = current
    for (const observer of observers) observer.onStatus()
  }
  state.subscribe(statusChanged)
  transcript.subscribe(statusChanged)

  /** Apply `change` unless every field in it is already current, as React's state setters bail out. */
  const patch = (change: Partial<LiveSessionState>) => {
    const current = state.get()
    if (Object.entries(change).every(([key, value]) => current[key as keyof LiveSessionState] === value)) return
    state.set({ ...current, ...change })
  }

  const coalescer = createTranscriptCoalescer((action) => {
    const before = transcript.get()
    const after = transcriptReducer(before, action)
    if (after === before) return
    transcript.set(after)
    if (after.messages !== before.messages || after.queue !== before.queue) confirmFromTranscript(sessionId, after)
  })
  const dispatch = coalescer.push

  const notify = (text: string, tone: Notice["tone"]) => {
    if (!detach) return
    const id = `notice-${++nextId}`
    patch({ notices: [...state.get().notices, { id, text, tone }] })
    const timer = setTimeout(() => {
      noticeTimers.delete(timer)
      patch({ notices: state.get().notices.filter((notice) => notice.id !== id) })
    }, NOTICE_LIFETIME_MS)
    noticeTimers.add(timer)
  }

  /** Write one command to the socket: its answer, or null when detached or the socket is down and nothing was sent. */
  const post = (type: string, fields: Record<string, unknown> = {}): Promise<PiResponse> | null => {
    const id = `${CLIENT_ID}-${++nextId}`
    if (!detach || !sendPiCommand(sessionId, { id, type, ...fields })) return null
    const { promise, resolve } = Promise.withResolvers<PiResponse>()
    pending.set(id, resolve)
    return promise
  }

  const command = <C extends string>(type: C, fields: Record<string, unknown> = {}): Promise<PiResponse<C>> =>
    // SAFETY: Pi answers command `C` with the data `PiResponse<C>` states
    (post(type, fields) ?? Promise.resolve({ success: false, error: "Not connected" })) as Promise<PiResponse<C>>

  const refreshStats = async () => {
    const response = await command("get_session_stats")
    if (response.success) patch({ stats: response.data })
  }

  const failPending = () => {
    for (const resolve of pending.values()) resolve({ success: false, error: "Disconnected" })
    pending.clear()
  }

  /** Send held messages not yet on the wire, oldest first. A down socket leaves them held for the next flush. A drop before the answer makes delivery uncertain: "not sent", never resent here. */
  const flush = () => {
    for (const entry of heldOf(sessionId)) {
      if (entry.state !== "sending" || entry.dispatched) continue
      const fields = { message: entry.text, ...(entry.images.length > 0 ? { images: entry.images } : {}), streamingBehavior: entry.mode }
      const answer = post("prompt", fields)
      if (!answer) return
      patchHeld(sessionId, entry.id, { dispatched: true })
      void answer.then((response) => {
        if (response.success) patchHeld(sessionId, entry.id, { state: "queued" })
        else patchHeld(sessionId, entry.id, { state: "failed", error: response.error })
      })
    }
  }

  const handleExtensionRequest = (record: PiRecord) => {
    if (isDialogRequest(record)) {
      const { dialogs } = state.get()
      if (dialogs.some((open) => open.id === record.id)) return
      patch({ dialogs: [...dialogs, record] })
      return emit({ type: "waiting", sessionId })
    }
    switch (record.method) {
      case "notify": {
        const message = String(record.message ?? "")
        if (message.startsWith(OPEN_FILE_NOTICE_PREFIX)) for (const open of openers) open(message.slice(OPEN_FILE_NOTICE_PREFIX.length))
        else notify(message, (record.notifyType as Notice["tone"]) ?? "info")
        break
      }
      case "setStatus": {
        const key = String(record.statusKey)
        const rest = Object.fromEntries(Object.entries(state.get().statusLines).filter(([existing]) => existing !== key))
        patch({ statusLines: typeof record.statusText === "string" && record.statusText ? { ...rest, [key]: record.statusText } : rest })
        break
      }
      case "setWidget": {
        const key = String(record.widgetKey)
        const lines = Array.isArray(record.widgetLines) ? record.widgetLines.filter((line): line is string => typeof line === "string") : []
        // The inspect reply is a one-shot answer to `inspectSubagent`, not widget state
        if (key === INSPECT_WIDGET_KEY) {
          const reply = parseInspectReply(lines)
          if (reply) inspects.get(reply.requestId)?.(reply)
          break
        }
        if (key === BTW_WIDGET_KEY) {
          const threads = parseBtwWidget(record)
          if (threads) patch({ btwThreads: threads })
          break
        }
        const { widgets } = state.get()
        if (lines.length > 0) patch({ widgets: { ...widgets, [key]: lines } })
        else if (key in widgets) patch({ widgets: Object.fromEntries(Object.entries(widgets).filter(([existing]) => existing !== key)) })
        break
      }
      case "set_editor_text": patch({ editorText: { text: String(record.text ?? "") } }); break
    }
  }

  const handleRecord = (record: PiRecord) => {
    if (record.type === "response") {
      const resolve = record.id ? pending.get(record.id) : undefined
      if (resolve) {
        // SAFETY: the record is a `response` frame, whose fields are those of PiResponse
        resolve(record as unknown as PiResponse)
        pending.delete(record.id!)
      }
      return
    }
    // Replay rebuilds the transcript only: the snapshot already holds the open dialogs and statuses, old notices are stale
    if (!replaying) {
      if (record.type === "extension_ui_request") return handleExtensionRequest(record)
      if (record.type === "extension_error") return notify(`${String(record.extensionPath)}: ${String(record.error)}`, "error")
      if (record.type === "auto_retry_start") notify(`Retrying (${String(record.attempt)}/${String(record.maxAttempts)}): ${String(record.errorMessage)}`, "warning")
      // A live message_end has no durable entry id: the resync brings it (M2)
      if (record.type === "message_end") {
        sync?.request("message_end")
        if ((record.message as { role?: string } | undefined)?.role === "assistant") void refreshStats()
      }
      if (record.type === "agent_settled" || record.type === "compaction_end") {
        sync?.request()
        void refreshStats()
      }
      if (record.type === "agent_start") {
        aborted = settled = false
        patch({ failed: false })
      }
      if (record.type === "agent_end") {
        const stopReason = stopReasonOf(record)
        if (stopReason === "aborted") aborted = true
        if (stopReason === "error" && !record.willRetry && !state.get().failed) {
          patch({ failed: true })
          emit({ type: "failed", sessionId })
        }
      }
    }
    const current = state.get().state
    if (record.type === "session_info_changed") patch({ state: current && { ...current, sessionName: record.name as string | undefined } })
    if (record.type === "thinking_level_changed") patch({ state: current && { ...current, thinkingLevel: record.level as string } })
    dispatch({ kind: "event", event: record })
    if (record.type === "agent_settled" && !replaying && !settled) {
      settled = true
      emit({ type: "settled", sessionId, aborted })
    }
  }

  const applyUi = (ui: AttachedUi) => {
    patch({ dialogs: ui.dialogs.filter(isDialogRequest), statusLines: ui.statuses, widgets: ui.widgets ?? NO_WIDGETS })
  }

  const handleFrame = (frame: HostFrame) => {
    const { streamChanged, apply } = sync!.receive(frame)
    if (streamChanged) dispatch({ kind: "stream_changed" })
    if (!apply) return
    switch (frame.type) {
      case "attached": {
        const { state: piState, ui } = frame
        patch({ state: piState })
        dispatch({ kind: "snapshot", entries: frame.entries, running: piState.isStreaming, queue: ui.queue ?? EMPTY_QUEUE })
        applyUi(ui)
        patch({ status: { kind: "ready" } })
        void command("get_commands").then((response) => {
          if (!response.success) return
          const listed = response.data.commands
          patch({ commands: listed })
          // The widget is no part of the attach snapshot: ask the extension to publish its state (an unknown command would reach the model)
          if (listed.some((entry) => entry.name === "btw-list")) void command("prompt", { message: "/btw-list" })
        })
        void command("get_available_models").then((response) => {
          if (response.success) patch({ models: response.data.models })
        })
        void command("get_available_thinking_levels").then((response) => {
          if (response.success) patch({ thinkingLevels: response.data.levels })
        })
        void refreshStats()
        return flush()
      }
      case "resync": {
        const { state: piState, ui, mode, entries, leafId } = frame
        if (piState) patch({ state: piState })
        if (ui) applyUi(ui)
        dispatch({ kind: "resync", resync: { mode, entries, leafId }, running: piState?.isStreaming, queue: ui?.queue })
        if (state.get().status.kind === "closed") return
        patch({ status: { kind: "ready" } })
        return flush()
      }
      case "ui_closed": return patch({ dialogs: state.get().dialogs.filter((dialog) => dialog.id !== frame.id) })
      case "replay_start": replaying = true; return dispatch({ kind: "replay_start" })
      case "replay_end": replaying = false; return
      case "pi": return handleRecord(frame.record)
      case "lock_held": return patch({ status: { kind: "closed", reason: "Another Pi process holds this session" } })
      case "exit": return patch({ status: { kind: "closed", reason: `Pi exited (${String(frame.code)}) ${String(frame.stderrTail ?? "")}`.trim() } })
      case "error": return notify(String(frame.message), "error")
    }
  }

  const connect = () => {
    sync = new SessionResync(sessionId, sendPiFrame)
    detach = subscribePiSession(sessionId, {
      onFrame: handleFrame,
      onStatus: (socketStatus) => {
        if (socketStatus.kind === "open") return
        failPending()
        replaying = false
        sync?.disconnected()
        if (state.get().status.kind !== "closed") patch({ status: socketStatus.kind === "connecting" ? { kind: "connecting" } : socketStatus })
      },
      onReopen: () => sync?.reconnected(),
      catchUp: () => sync?.request(),
    })
  }

  const disconnect = () => {
    detach?.()
    detach = null
    failPending()
    sync = null
    replaying = aborted = settled = false
    coalescer.reset()
    for (const timer of noticeTimers) clearTimeout(timer)
    noticeTimers.clear()
    state.set(INITIAL_STATE)
    transcript.set(EMPTY_TRANSCRIPT)
  }

  /** Start a run, or hold the message and deliver it into the running one as `mode`. A new run's prompt shows optimistically until Pi reports it. Failures become a notice; resolves to whether Pi took the message (a held one counts). */
  const send = async (message: string, images: ImageContent[], mode: QueueMode = "steer"): Promise<boolean> => {
    // A btw command is handled at once by its extension, also mid-run: never held, steered or queued, no optimistic prompt
    if (isBtwCommand(message)) {
      const response = await command("prompt", { message })
      if (!response.success) notify(response.error ?? "Side question refused", "error")
      return response.success
    }
    if (transcript.get().running) {
      hold(sessionId, { text: message, images, mode })
      confirmFromTranscript(sessionId, transcript.get())
      flush()
      return true
    }
    const prompt: UserMessage = { role: "user", content: images.length > 0 ? [{ type: "text", text: message }, ...images] : message, timestamp: Date.now() }
    dispatch({ kind: "prompt", prompt })
    const response = await command("prompt", { message, ...(images.length > 0 ? { images } : {}) })
    if (response.success) return true
    dispatch({ kind: "prompt", prompt: null })
    notify(response.error ?? "Prompt refused", "error")
    return false
  }

  return {
    /** Everything but the transcript: a new object per change. */
    state,
    /** A new object per reducer step, at most one per animation frame while a reply streams. */
    transcript,
    /** Keep the session attached until the returned release runs. The first holder attaches, the last detaches and drops the state. */
    attach() {
      if (++holders === 1) connect()
      return () => {
        if (--holders === 0) disconnect()
      }
    },
    /** Called with the path of every `Open file: ` notice. */
    onOpenFile(open: (path: string) => void) {
      openers.add(open)
      return () => openers.delete(open)
    },
    send,
    notify,
    /** Stop the run. Resolves to the queued messages taken back, and the held ones not queued, for the draft. */
    async abort(): Promise<string[]> {
      const taken = takeHeld(sessionId)
      const cleared = await command("clear_queue")
      await command("abort")
      return takeBack(taken, cleared.data ?? EMPTY_QUEUE)
    },
    /** Run `line` in the session's shell (`!cmd`), its output kept out of the model's context with `excludeFromContext` (`!!cmd`). */
    async runBash(line: string, excludeFromContext: boolean) {
      const response = await command("bash", { command: line, ...(excludeFromContext ? { excludeFromContext: true } : {}) })
      if (!response.success) notify(response.error ?? "Command refused", "error")
      sync?.request()
    },
    /** Compact the context now, steered by `customInstructions` when given. */
    async compact(customInstructions?: string) {
      const response = await command("compact", customInstructions ? { customInstructions } : {})
      if (!response.success) notify(response.error ?? "Compaction failed", "error")
      sync?.request()
      void refreshStats()
    },
    async setModel(model: PiModel) {
      const response = await command("set_model", { provider: model.provider, modelId: model.id })
      if (!response.success) return notify(response.error ?? "Model refused", "error")
      // Pi may clamp the thinking level to the new model's
      const [stateResponse, levelsResponse] = await Promise.all([command("get_state"), command("get_available_thinking_levels")])
      if (stateResponse.success) patch({ state: stateResponse.data })
      if (levelsResponse.success) patch({ thinkingLevels: levelsResponse.data.levels })
      void refreshStats()
    },
    async setThinking(level: string) {
      const response = await command("set_thinking_level", { level })
      if (!response.success) return notify(response.error ?? "Thinking level refused", "error")
      const current = state.get().state
      patch({ state: current && { ...current, thinkingLevel: level } })
    },
    /** A background subagent's task, transcript window and output, read through its extension's `/subagents-inspect-rpc` command (no model turn). */
    async inspectSubagent(target: NonNullable<SubagentRow["inspect"]>): Promise<InspectReply> {
      const requestId = `${CLIENT_ID}-i${++nextId}`
      const failed = (message: string): InspectReply => ({ requestId, error: { code: "unavailable", message } })
      const { promise, resolve } = Promise.withResolvers<InspectReply>()
      const timer = setTimeout(() => resolve(failed("No answer from the subagent extension")), INSPECT_TIMEOUT_MS)
      inspects.set(requestId, resolve)
      const response = await command("prompt", { message: inspectCommand(requestId, target) })
      if (!response.success) resolve(failed(response.error ?? "Inspection refused"))
      const reply = await promise
      clearTimeout(timer)
      inspects.delete(requestId)
      return reply
    },
    answerDialog(id: string, answer: { value: string } | { confirmed: boolean } | { cancelled: true }) {
      sendPiCommand(sessionId, { type: "extension_ui_response", id, ...answer })
      patch({ dialogs: state.get().dialogs.filter((dialog) => dialog.id !== id) })
    },
    retryHeld(id: string) {
      patchHeld(sessionId, id, { state: "sending", dispatched: false, error: undefined })
      flush()
    },
  }
}
