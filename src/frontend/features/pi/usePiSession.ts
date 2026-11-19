"use client"

import { useCallback, useEffect, useReducer, useRef, useState } from "react"
import { useCoalescedDispatch } from "./coalescer"
import { sendPiCommand, sendPiFrame, subscribePiSession, type HostFrame } from "./piSocket"
import { SessionResync } from "./resync"
import { EMPTY_TRANSCRIPT, transcriptReducer, type ResyncEntries, type SessionEntries, type Transcript } from "./transcript"
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
  queue: Queue
}

const NOTICE_LIFETIME_MS = 6_000
const DIALOG_METHODS = new Set(["select", "confirm", "input", "editor"])
const EMPTY_QUEUE: Queue = { steering: [], followUp: [] }
/** Distinguishes this tab's command ids from other clients' on the same session, their responses reach every attached client. */
const CLIENT_ID = crypto.randomUUID().slice(0, 8)

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
 * - `abort` takes the queued messages back (returned for the draft) before stopping, like Pi's Esc (F5)
 * - `runBash` runs `!cmd` (F7), `compact` compacts with optional instructions (F8), `setModel`/`setThinking` switch the session's model and thinking (D10, F10)
 * - `stats` (cost, context use) refresh on attach, after each run, compaction and model switch (F3)
 * - Extension dialogs queue until answered, `notify` shows briefly, `setStatus` keeps a line per key
 */
export function usePiSession(sessionId: string) {
  const [transcript, dispatchNow] = useReducer(transcriptReducer, EMPTY_TRANSCRIPT)
  const dispatch = useCoalescedDispatch(dispatchNow)
  const [status, setStatus] = useState<ConnectionStatus>({ kind: "connecting" })
  const [state, setState] = useState<PiState | null>(null)
  const [commands, setCommands] = useState<PiCommand[]>([])
  const [dialogs, setDialogs] = useState<ExtensionDialog[]>([])
  const [notices, setNotices] = useState<Notice[]>([])
  const [statusLines, setStatusLines] = useState<Record<string, string>>({})
  const [editorText, setEditorText] = useState<{ text: string } | null>(null)
  const [models, setModels] = useState<PiModel[]>([])
  const [thinkingLevels, setThinkingLevels] = useState<string[]>([])
  const [stats, setStats] = useState<PiSessionStats | null>(null)
  const pendingRef = useRef(new Map<string, (response: Response) => void>())
  const nextIdRef = useRef(0)
  const syncRef = useRef<SessionResync | null>(null)

  const notify = useCallback((text: string, tone: Notice["tone"]) => {
    const id = `notice-${++nextIdRef.current}`
    setNotices((current) => [...current, { id, text, tone }])
    setTimeout(() => setNotices((current) => current.filter((notice) => notice.id !== id)), NOTICE_LIFETIME_MS)
  }, [])

  const command = useCallback((type: string, fields: Record<string, unknown> = {}): Promise<Response> => {
    const id = `${CLIENT_ID}-${++nextIdRef.current}`
    if (!sendPiCommand(sessionId, { id, type, ...fields })) return Promise.resolve({ success: false, error: "Not connected" })
    const { promise, resolve } = Promise.withResolvers<Response>()
    pendingRef.current.set(id, resolve)
    return promise
  }, [sessionId])

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
        case "notify": notify(String(record.message ?? ""), (record.notifyType as Notice["tone"]) ?? "info"); break
        case "setStatus": setStatusLines((current) => {
          const key = String(record.statusKey)
          const rest = Object.fromEntries(Object.entries(current).filter(([existing]) => existing !== key))
          return typeof record.statusText === "string" && record.statusText ? { ...rest, [key]: record.statusText } : rest
        }); break
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
            if (response.success) setCommands((response.data as { commands: PiCommand[] }).commands)
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

  /** Start a run, or queue into the running one as `mode`. A new run's prompt shows optimistically until Pi reports it. Failures become a notice. */
  const send = useCallback(async (message: string, images: ImageContent[], mode: QueueMode = "steer") => {
    const queueing = transcript.running
    const fields = { message, ...(images.length > 0 ? { images } : {}), ...(queueing ? { streamingBehavior: mode } : {}) }
    if (!queueing) {
      const prompt: UserMessage = { role: "user", content: images.length > 0 ? [{ type: "text", text: message }, ...images] : message, timestamp: Date.now() }
      dispatch({ kind: "prompt", prompt })
    }
    const response = await command("prompt", fields)
    if (response.success) return
    if (!queueing) dispatch({ kind: "prompt", prompt: null })
    notify(response.error ?? "Prompt refused", "error")
  }, [command, notify, dispatch, transcript.running])

  /** Stop the run. Resolves to the queued messages taken back, for the draft. */
  const abort = useCallback(async (): Promise<string[]> => {
    const cleared = await command("clear_queue")
    await command("abort")
    const queue = (cleared.data ?? EMPTY_QUEUE) as Queue
    return [...queue.steering, ...queue.followUp]
  }, [command])

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

  const answerDialog = useCallback((id: string, answer: { value: string } | { confirmed: boolean } | { cancelled: true }) => {
    sendPiCommand(sessionId, { type: "extension_ui_response", id, ...answer })
    setDialogs((current) => current.filter((dialog) => dialog.id !== id))
  }, [sessionId])

  return {
    transcript, status, state, commands, dialogs, notices, statusLines, editorText, models, thinkingLevels, stats,
    send, abort, runBash, compact, setModel, setThinking, answerDialog, notify,
  }
}
