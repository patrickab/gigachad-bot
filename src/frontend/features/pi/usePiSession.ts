"use client"

import { useCallback, useEffect, useReducer, useRef, useState } from "react"
import { sendPiCommand, subscribePiSession, type HostFrame } from "./piSocket"
import { EMPTY_TRANSCRIPT, reduceTranscript, transcriptOfSnapshot, type SessionEntries, type Transcript } from "./transcript"
import type { ExtensionDialog, ImageContent, PiCommand, PiState } from "./types"

export type ConnectionStatus = { kind: "connecting" } | { kind: "ready" } | { kind: "closed"; reason: string }

export interface Notice { id: string; text: string; tone: "info" | "warning" | "error" }

type PiRecord = { type: string; id?: string; [field: string]: unknown }
type Response = { success: boolean; data?: unknown; error?: string }
type Queue = Transcript["queue"]

/** pi-host's view of the session's open UI when a client attaches (M3). */
interface AttachedUi {
  dialogs: PiRecord[]
  statuses: Record<string, string>
  queue: Queue
}

type TranscriptAction =
  | { kind: "event"; event: PiRecord }
  | { kind: "snapshot"; entries: SessionEntries; running: boolean; queue: Queue }

function transcriptReducer(transcript: Transcript, action: TranscriptAction): Transcript {
  if (action.kind === "snapshot") return transcriptOfSnapshot(action.entries, action.running, action.queue)
  return reduceTranscript(transcript, action.event)
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
 * - A reconnect re-attaches and rebuilds everything from the new snapshot
 * - `send` starts a run, or steers the running one (Pi delivers it after the current tool calls)
 * - `abort` takes the queued messages back (returned for the draft) before stopping, like Pi's Esc
 * - Extension dialogs queue until answered, `notify` shows briefly, `setStatus` keeps a line per key
 */
export function usePiSession(sessionId: string) {
  const [transcript, dispatch] = useReducer(transcriptReducer, EMPTY_TRANSCRIPT)
  const [status, setStatus] = useState<ConnectionStatus>({ kind: "connecting" })
  const [state, setState] = useState<PiState | null>(null)
  const [commands, setCommands] = useState<PiCommand[]>([])
  const [dialogs, setDialogs] = useState<ExtensionDialog[]>([])
  const [notices, setNotices] = useState<Notice[]>([])
  const [statusLines, setStatusLines] = useState<Record<string, string>>({})
  const [editorText, setEditorText] = useState<{ text: string } | null>(null)
  const pendingRef = useRef(new Map<string, (response: Response) => void>())
  const nextIdRef = useRef(0)

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

  useEffect(() => {
    const pending = pendingRef.current
    let replaying = false

    const failPending = () => {
      for (const resolve of pending.values()) resolve({ success: false, error: "Disconnected" })
      pending.clear()
    }

    const reloadEntries = async () => {
      const [entriesResponse, stateResponse] = await Promise.all([command("get_entries"), command("get_state")])
      if (!entriesResponse.success || !stateResponse.success) return
      const piState = stateResponse.data as PiState
      setState(piState)
      dispatch({ kind: "snapshot", entries: entriesOf(entriesResponse.data), running: piState.isStreaming, queue: EMPTY_QUEUE })
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
        if (record.type === "compaction_end") void reloadEntries()
      }
      if (record.type === "session_info_changed") setState((current) => current && { ...current, sessionName: record.name as string | undefined })
      if (record.type === "thinking_level_changed") setState((current) => current && { ...current, thinkingLevel: record.level as string })
      dispatch({ kind: "event", event: record })
    }

    const handleFrame = (frame: HostFrame) => {
      switch (frame.type) {
        case "attached": {
          const piState = frame.state as PiState
          const ui = frame.ui as AttachedUi
          setState(piState)
          dispatch({ kind: "snapshot", entries: entriesOf(frame.entries), running: piState.isStreaming, queue: ui.queue ?? EMPTY_QUEUE })
          setDialogs(ui.dialogs.filter(isDialogRequest) as unknown as ExtensionDialog[])
          setStatusLines(ui.statuses)
          setStatus({ kind: "ready" })
          void command("get_commands").then((response) => {
            if (response.success) setCommands((response.data as { commands: PiCommand[] }).commands)
          })
          return
        }
        case "ui_closed": return setDialogs((current) => current.filter((dialog) => dialog.id !== frame.id))
        case "replay_start": replaying = true; return
        case "replay_end": replaying = false; return
        case "pi": return handleRecord(frame.record as PiRecord)
        case "lock_held": return setStatus({ kind: "closed", reason: "Another Pi process holds this session" })
        case "exit": return setStatus({ kind: "closed", reason: `Pi exited (${String(frame.code)}) ${String(frame.stderrTail ?? "")}`.trim() })
        case "error": return notify(String(frame.message), "error")
      }
    }

    const unsubscribe = subscribePiSession(sessionId, {
      onFrame: handleFrame,
      onStatus: (socketStatus) => {
        if (socketStatus.kind === "open") return
        failPending()
        replaying = false
        setStatus((current) => current.kind === "closed" ? current : socketStatus.kind === "connecting" ? { kind: "connecting" } : socketStatus)
      },
    })
    return () => {
      unsubscribe()
      failPending()
    }
  }, [sessionId, command, notify])

  /** Start a run, or steer the running one. Failures become a notice. */
  const send = useCallback(async (message: string, images: ImageContent[]) => {
    const fields = { message, ...(images.length > 0 ? { images } : {}), ...(transcript.running ? { streamingBehavior: "steer" } : {}) }
    const response = await command("prompt", fields)
    if (!response.success) notify(response.error ?? "Prompt refused", "error")
  }, [command, notify, transcript.running])

  /** Stop the run. Resolves to the queued messages taken back, for the draft. */
  const abort = useCallback(async (): Promise<string[]> => {
    const cleared = await command("clear_queue")
    await command("abort")
    const queue = (cleared.data ?? EMPTY_QUEUE) as Queue
    return [...queue.steering, ...queue.followUp]
  }, [command])

  const answerDialog = useCallback((id: string, answer: { value: string } | { confirmed: boolean } | { cancelled: true }) => {
    sendPiCommand(sessionId, { type: "extension_ui_response", id, ...answer })
    setDialogs((current) => current.filter((dialog) => dialog.id !== id))
  }, [sessionId])

  return { transcript, status, state, commands, dialogs, notices, statusLines, editorText, send, abort, answerDialog }
}
