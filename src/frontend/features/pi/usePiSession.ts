"use client"

import { useCallback, useEffect, useEffectEvent, useReducer, useMemo, useRef, useState } from "react"
import { BTW_WIDGET_KEY, isBtwCommand, parseBtwWidget, type BtwThread } from "./btw"
import { useCoalescedDispatch } from "./coalescer"
import { confirmedIds, heldOf, hold, patchHeld, removeHeld, takeHeld, useHeld, type HeldMessage } from "./outbox"
import { sendPiCommand, sendPiFrame, subscribePiSession, type HostFrame } from "./piSocket"
import { SessionResync } from "./resync"
import { INSPECT_WIDGET_KEY, inspectCommand, parseInspectReply, type InspectReply, type SubagentRow } from "./subagents"
import { EMPTY_TRANSCRIPT, textOf, transcriptReducer, type ResyncEntries, type SessionEntries, type Transcript } from "./transcript"
import type { ExtensionDialog, ImageContent, PiCommand, PiModel, PiState, UserMessage } from "./types"

export type ConnectionStatus = { kind: "connecting" } | { kind: "ready" } | { kind: "closed"; reason: string }

export interface Notice { id: string; text: string; tone: "info" | "warning" | "error" }

/** `get_session_stats` data the top bar reads: the session's cost and the current context window use (F3). */
export interface PiSessionStats {
  cost: number
  /** Absent without a model, `tokens`/`percent` null right after a compaction until the next reply. */
  contextUsage?: { tokens: number | null; contextWindow: number; percent: number | null }
}

/** How a message sent while a run streams is queued: delivered after the current tool calls, or once the run ends (F4). */
export type QueueMode = "steer" | "followUp"

type PiRecord = { type: string; id?: string; [field: string]: unknown }
type Response = { success: boolean; data?: unknown; error?: string }
type Queue = Transcript["queue"]

/** pi-host's view of the session's open UI when a client attaches (M3). */
interface AttachedUi {
  dialogs: PiRecord[]
  statuses: Record<string, string>
  /** Open text widgets by key; absent from an older pi-host. */
  widgets?: Record<string, string[]>
  queue: Queue
}

const NOTICE_LIFETIME_MS = 6_000
const INSPECT_TIMEOUT_MS = 15_000
const NO_WIDGETS: Record<string, string[]> = {}
const DIALOG_METHODS = new Set(["select", "confirm", "input", "editor"])
const EMPTY_QUEUE: Queue = { steering: [], followUp: [] }
/** Distinguishes this tab's command ids from other clients' on the same session, their responses reach every attached client. */
const CLIENT_ID = crypto.randomUUID().slice(0, 8)
/** An extension's `notify` asking the GUI to open the path that follows instead of showing a notice (C7, `pi-extensions/files.ts`). */
export const OPEN_FILE_NOTICE_PREFIX = "Open file: "

function isDialogRequest(record: PiRecord): boolean {
  return record.type === "extension_ui_request" && DIALOG_METHODS.has(record.method as string)
}

/** `get_entries` data, tolerating a bare entry array (leaf = last entry). */
function entriesOf(data: unknown): SessionEntries {
  if (Array.isArray(data)) return { entries: data, leafId: data.length > 0 ? data[data.length - 1].id : null }
  return data as SessionEntries
}

/**
 * One Pi session on the page's shared pi-host socket (`piSocket.ts`), the whole RPC protocol in the browser.
 *
 * - Attaches `sessionId` on mount: `attached` sets state, transcript (entries), dialogs, statuses and queue, the replay and live `pi` records fold into it
 * - Catches up with a guarded `resync` (`resync.ts`, M2, M22) on reconnect, tab visible, `online`, `agent_settled`, compaction and every live `message_end`, whose message gets its entry from the resync
 * - A new `streamId` drops the live state (streaming reply, live messages, optimistic prompt) and resyncs from scratch
 * - `send` starts a run (shown as an optimistic prompt until Pi reports it), or queues into the running one as steering or a follow-up (F4)
 * - A message sent during a run is held (`outbox.ts`) until the queue snapshot or the session shows it, flushed in order after a reconnect
 * - `abort` takes the queued messages back (returned for the draft) before stopping, like Pi's Esc (F5)
 * - `runBash` runs `!cmd` (F7), `compact` compacts with optional instructions (F8), `setModel`/`setThinking` switch the session's model and thinking (D10, F10)
 * - `stats` (cost, context use) refresh on attach, after each run, compaction and model switch (F3)
 * - Extension dialogs queue until answered, `notify` shows briefly, `setStatus` keeps a line per key
 * - A `notify` led by `OPEN_FILE_NOTICE_PREFIX` goes to `onOpenFile` instead of showing
 */
export function usePiSession(sessionId: string, onOpenFile: (path: string) => void) {
  const [transcript, dispatchNow] = useReducer(transcriptReducer, EMPTY_TRANSCRIPT)
  const dispatch = useCoalescedDispatch(dispatchNow)
  const [status, setStatus] = useState<ConnectionStatus>({ kind: "connecting" })
  const [state, setState] = useState<PiState | null>(null)
  const [commands, setCommands] = useState<PiCommand[]>([])
  const [dialogs, setDialogs] = useState<ExtensionDialog[]>([])
  const [notices, setNotices] = useState<Notice[]>([])
  const [statusLines, setStatusLines] = useState<Record<string, string>>({})
  const [widgets, setWidgets] = useState<Record<string, string[]>>(NO_WIDGETS)
  const [editorText, setEditorText] = useState<{ text: string } | null>(null)
  const [models, setModels] = useState<PiModel[]>([])
  const [thinkingLevels, setThinkingLevels] = useState<string[]>([])
  const [stats, setStats] = useState<PiSessionStats | null>(null)
  const [btwThreads, setBtwThreads] = useState<BtwThread[]>([])
  const pendingRef = useRef(new Map<string, (response: Response) => void>())
  const nextIdRef = useRef(0)
  const inspectsRef = useRef(new Map<string, (reply: InspectReply) => void>())
  const syncRef = useRef<SessionResync | null>(null)
  const openFile = useEffectEvent(onOpenFile)

  const notify = useCallback((text: string, tone: Notice["tone"]) => {
    const id = `notice-${++nextIdRef.current}`
    setNotices((current) => [...current, { id, text, tone }])
    setTimeout(() => setNotices((current) => current.filter((notice) => notice.id !== id)), NOTICE_LIFETIME_MS)
  }, [])

  /** Write one command to the socket: its answer, or null when the socket is down and nothing was sent. */
  const post = useCallback((type: string, fields: Record<string, unknown> = {}): Promise<Response> | null => {
    const id = `${CLIENT_ID}-${++nextIdRef.current}`
    if (!sendPiCommand(sessionId, { id, type, ...fields })) return null
    const { promise, resolve } = Promise.withResolvers<Response>()
    pendingRef.current.set(id, resolve)
    return promise
  }, [sessionId])

  const command = useCallback((type: string, fields: Record<string, unknown> = {}): Promise<Response> => {
    return post(type, fields) ?? Promise.resolve({ success: false, error: "Not connected" })
  }, [post])

  const refreshStats = useCallback(async () => {
    const response = await command("get_session_stats")
    if (response.success) setStats(response.data as PiSessionStats)
  }, [command])

  /** Model, thinking level and the levels the model offers, after switching models (Pi may clamp the level). */
  const refreshModelState = useCallback(async () => {
    const [stateResponse, levelsResponse] = await Promise.all([command("get_state"), command("get_available_thinking_levels")])
    if (stateResponse.success) setState(stateResponse.data as PiState)
    if (levelsResponse.success) setThinkingLevels((levelsResponse.data as { levels: string[] }).levels)
  }, [command])

  useEffect(() => {
    const pending = pendingRef.current
    let replaying = false
    const sync = new SessionResync(sessionId, sendPiFrame)
    syncRef.current = sync

    const failPending = () => {
      for (const resolve of pending.values()) resolve({ success: false, error: "Disconnected" })
      pending.clear()
    }

    const handleExtensionRequest = (record: PiRecord) => {
      if (isDialogRequest(record)) {
        const dialog = record as unknown as ExtensionDialog
        setDialogs((current) => current.some((open) => open.id === dialog.id) ? current : [...current, dialog])
        return
      }
      switch (record.method) {
        case "notify": {
          const message = String(record.message ?? "")
          if (message.startsWith(OPEN_FILE_NOTICE_PREFIX)) openFile(message.slice(OPEN_FILE_NOTICE_PREFIX.length))
          else notify(message, (record.notifyType as Notice["tone"]) ?? "info")
          break
        }
        case "setStatus": setStatusLines((current) => {
          const key = String(record.statusKey)
          const rest = Object.fromEntries(Object.entries(current).filter(([existing]) => existing !== key))
          return typeof record.statusText === "string" && record.statusText ? { ...rest, [key]: record.statusText } : rest
        }); break
        case "setWidget": {
          const key = String(record.widgetKey)
          const lines = Array.isArray(record.widgetLines) ? record.widgetLines.filter((line): line is string => typeof line === "string") : []
          // The inspect reply is a one-shot answer to `inspectSubagent`, not widget state
          if (key === INSPECT_WIDGET_KEY) {
            const reply = parseInspectReply(lines)
            if (reply) inspectsRef.current.get(reply.requestId)?.(reply)
            break
          }
          if (key === BTW_WIDGET_KEY) {
            setBtwThreads((current) => parseBtwWidget(record) ?? current)
            break
          }
          setWidgets((current) => {
            if (lines.length === 0) return key in current ? Object.fromEntries(Object.entries(current).filter(([existing]) => existing !== key)) : current
            return { ...current, [key]: lines }
          })
          break
        }
        case "set_editor_text": setEditorText({ text: String(record.text ?? "") }); break
      }
    }

    const handleRecord = (record: PiRecord) => {
      if (record.type === "response") {
        const resolve = record.id ? pending.get(record.id) : undefined
        if (resolve) {
          resolve(record as unknown as Response)
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
        if (record.type === "message_end") sync.request("message_end")
        if (record.type === "agent_settled" || record.type === "compaction_end") {
          sync.request()
          void refreshStats()
        }
      }
      if (record.type === "session_info_changed") setState((current) => current && { ...current, sessionName: record.name as string | undefined })
      if (record.type === "thinking_level_changed") setState((current) => current && { ...current, thinkingLevel: record.level as string })
      dispatch({ kind: "event", event: record })
    }

    const applyUi = (ui: AttachedUi) => {
      setDialogs(ui.dialogs.filter(isDialogRequest) as unknown as ExtensionDialog[])
      setStatusLines(ui.statuses)
      setWidgets(ui.widgets ?? NO_WIDGETS)
    }

    const handleFrame = (frame: HostFrame) => {
      const { streamChanged, apply } = sync.receive(frame)
      if (streamChanged) dispatch({ kind: "stream_changed" })
      if (!apply) return
      switch (frame.type) {
        case "attached": {
          const piState = frame.state as PiState
          const ui = frame.ui as AttachedUi
          setState(piState)
          dispatch({ kind: "snapshot", entries: entriesOf(frame.entries), running: piState.isStreaming, queue: ui.queue ?? EMPTY_QUEUE })
          applyUi(ui)
          setStatus({ kind: "ready" })
          void command("get_commands").then((response) => {
            if (!response.success) return
            const listed = (response.data as { commands: PiCommand[] }).commands
            setCommands(listed)
            // The widget is no part of the attach snapshot: ask the extension to publish its state (an unknown command would reach the model)
            if (listed.some((entry) => entry.name === "btw-list")) void command("prompt", { message: "/btw-list" })
          })
          void command("get_available_models").then((response) => {
            if (response.success) setModels((response.data as { models: PiModel[] }).models)
          })
          void command("get_available_thinking_levels").then((response) => {
            if (response.success) setThinkingLevels((response.data as { levels: string[] }).levels)
          })
          void refreshStats()
          return
        }
        case "resync": {
          // pi-host adds `state` and `ui` like `attached`; tolerated missing
          const piState = frame.state as PiState | undefined
          const ui = frame.ui as AttachedUi | undefined
          const resync = { mode: frame.mode, entries: frame.entries, leafId: frame.leafId } as ResyncEntries
          if (piState) setState(piState)
          if (ui) applyUi(ui)
          dispatch({ kind: "resync", resync, running: piState?.isStreaming, queue: ui?.queue })
          setStatus((current) => current.kind === "closed" ? current : { kind: "ready" })
          return
        }
        case "ui_closed": return setDialogs((current) => current.filter((dialog) => dialog.id !== frame.id))
        case "replay_start": replaying = true; return dispatch({ kind: "replay_start" })
        case "replay_end": replaying = false; return
        case "pi": return handleRecord(frame.record as PiRecord)
        case "lock_held": return setStatus({ kind: "closed", reason: "Another Pi process holds this session" })
        case "exit": return setStatus({ kind: "closed", reason: `Pi exited (${String(frame.code)}) ${String(frame.stderrTail ?? "")}`.trim() })
        case "error": return notify(String(frame.message), "error")
      }
    }

    const catchUpWhenVisible = () => {
      if (document.visibilityState === "visible") sync.request()
    }
    const catchUp = () => sync.request()
    document.addEventListener("visibilitychange", catchUpWhenVisible)
    window.addEventListener("online", catchUp)

    const unsubscribe = subscribePiSession(sessionId, {
      onFrame: handleFrame,
      onStatus: (socketStatus) => {
        if (socketStatus.kind === "open") return
        failPending()
        replaying = false
        sync.disconnected()
        setStatus((current) => current.kind === "closed" ? current : socketStatus.kind === "connecting" ? { kind: "connecting" } : socketStatus)
      },
      onReopen: () => sync.reconnected(),
    })
    return () => {
      document.removeEventListener("visibilitychange", catchUpWhenVisible)
      window.removeEventListener("online", catchUp)
      unsubscribe()
      failPending()
      syncRef.current = null
    }
  }, [sessionId, command, notify, dispatch, refreshStats])

  /** Send held messages not yet on the wire, oldest first. A down socket leaves them held for the next flush. A drop before the answer makes delivery uncertain: "not sent", never resent here. */
  const flush = useCallback(() => {
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
  }, [sessionId, post])

  /** Start a run, or hold the message and deliver it into the running one as `mode`. A new run's prompt shows optimistically until Pi reports it. Failures become a notice. */
  const send = useCallback(async (message: string, images: ImageContent[], mode: QueueMode = "steer") => {
    // A btw command is handled at once by its extension, also mid-run: never held, steered or queued, no optimistic prompt
    if (isBtwCommand(message)) {
      const response = await command("prompt", { message })
      if (!response.success) notify(response.error ?? "Side question refused", "error")
      return
    }
    if (transcript.running) {
      hold(sessionId, { text: message, images, mode })
      flush()
      return
    }
    const prompt: UserMessage = { role: "user", content: images.length > 0 ? [{ type: "text", text: message }, ...images] : message, timestamp: Date.now() }
    dispatch({ kind: "prompt", prompt })
    const response = await command("prompt", { message, ...(images.length > 0 ? { images } : {}) })
    if (response.success) return
    dispatch({ kind: "prompt", prompt: null })
    notify(response.error ?? "Prompt refused", "error")
  }, [command, notify, dispatch, transcript.running, sessionId, flush])

  const btw = useMemo(() => ({
    threads: btwThreads,
    followUp: (id: string, question: string) => void send(`/btw-followup ${id} ${question}`, []),
    cancel: (id: string) => void send(`/btw-cancel ${id}`, []),
    dismiss: (id: string) => void send(`/btw-dismiss ${id}`, []),
  }), [btwThreads, send])

  const held = useHeld(sessionId)

  // Flush once attached (also after a reconnect), in order
  useEffect(() => {
    if (status.kind === "ready" && held.some((entry) => entry.state === "sending" && !entry.dispatched)) flush()
  }, [status.kind, held, flush])

  // Pi's queue snapshot or the user message in the session confirms a held message
  useEffect(() => {
    if (held.length === 0) return
    const userTexts = (since: number) => transcript.messages
      .filter((message) => message.role === "user" && message.timestamp >= since - 2_000)
      .map((message) => textOf((message as UserMessage).content))
    const ids = confirmedIds(held, transcript.queue, userTexts)
    if (ids.length > 0) removeHeld(sessionId, ids)
  }, [held, transcript.queue, transcript.messages, sessionId])

  const retryHeld = useCallback((id: string) => {
    patchHeld(sessionId, id, { state: "sending", dispatched: false, error: undefined })
    flush()
  }, [sessionId, flush])
  const deleteHeld = useCallback((id: string) => removeHeld(sessionId, [id]), [sessionId])

  /** Stop the run. Resolves to the queued messages taken back, and the held ones not queued, for the draft. */
  const abort = useCallback(async (): Promise<string[]> => {
    const taken = takeHeld(sessionId)
    const cleared = await command("clear_queue")
    await command("abort")
    const queue = (cleared.data ?? EMPTY_QUEUE) as Queue
    const back = [...queue.steering, ...queue.followUp]
    const unqueued: string[] = []
    for (const entry of taken as readonly HeldMessage[]) {
      const at = back.indexOf(entry.text)
      if (at >= 0) back.splice(at, 1)
      else unqueued.push(entry.text)
    }
    return [...queue.steering, ...queue.followUp, ...unqueued]
  }, [command, sessionId])

  /** Run `line` in the session's shell (`!cmd`), its output kept out of the model's context with `excludeFromContext` (`!!cmd`). */
  const runBash = useCallback(async (line: string, excludeFromContext: boolean) => {
    const response = await command("bash", { command: line, ...(excludeFromContext ? { excludeFromContext: true } : {}) })
    if (!response.success) notify(response.error ?? "Command refused", "error")
    syncRef.current?.request()
  }, [command, notify])

  /** Compact the context now, steered by `customInstructions` when given. */
  const compact = useCallback(async (customInstructions?: string) => {
    const response = await command("compact", customInstructions ? { customInstructions } : {})
    if (!response.success) notify(response.error ?? "Compaction failed", "error")
    syncRef.current?.request()
    void refreshStats()
  }, [command, notify, refreshStats])

  const setModel = useCallback(async (model: PiModel) => {
    const response = await command("set_model", { provider: model.provider, modelId: model.id })
    if (!response.success) return notify(response.error ?? "Model refused", "error")
    await refreshModelState()
    void refreshStats()
  }, [command, notify, refreshModelState, refreshStats])

  const setThinking = useCallback(async (level: string) => {
    const response = await command("set_thinking_level", { level })
    if (!response.success) return notify(response.error ?? "Thinking level refused", "error")
    setState((current) => current && { ...current, thinkingLevel: level })
  }, [command, notify])

  /** A background subagent's task, transcript window and output, read through its extension's `/subagents-inspect-rpc` command (no model turn). */
  const inspectSubagent = useCallback(async (target: NonNullable<SubagentRow["inspect"]>): Promise<InspectReply> => {
    const requestId = `${CLIENT_ID}-i${++nextIdRef.current}`
    const failed = (message: string): InspectReply => ({ requestId, error: { code: "unavailable", message } })
    const waiters = inspectsRef.current
    const { promise, resolve } = Promise.withResolvers<InspectReply>()
    const timer = setTimeout(() => resolve(failed("No answer from the subagent extension")), INSPECT_TIMEOUT_MS)
    waiters.set(requestId, resolve)
    const response = await command("prompt", { message: inspectCommand(requestId, target) })
    if (!response.success) resolve(failed(response.error ?? "Inspection refused"))
    const reply = await promise
    clearTimeout(timer)
    waiters.delete(requestId)
    return reply
  }, [command])

  const answerDialog = useCallback((id: string, answer: { value: string } | { confirmed: boolean } | { cancelled: true }) => {
    sendPiCommand(sessionId, { type: "extension_ui_response", id, ...answer })
    setDialogs((current) => current.filter((dialog) => dialog.id !== id))
  }, [sessionId])

  return {
    transcript, status, state, commands, dialogs, notices, statusLines, widgets, editorText, models, thinkingLevels, stats,
    btw, held, retryHeld, deleteHeld, send, abort, runBash, compact, setModel, setThinking, answerDialog, inspectSubagent, notify,
  }
}
