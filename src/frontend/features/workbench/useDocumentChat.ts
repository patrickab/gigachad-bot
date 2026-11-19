"use client"

import { useEffect, useEffectEvent, useState } from "react"
import { usePolling } from "@/hooks/usePolling"
import { ApiError, markWorkbenchChatInformed, openWorkbenchChat, type OpenedWorkbenchChat, type PiSessionOptions } from "@/lib/backend/api"
import { relativeIn } from "@/lib/paths"
import { errorMessage } from "@/lib/utils"

/** Poll interval while the backend extracts a PDF chat's text. */
const PDF_TEXT_POLL_MS = 5000

/** The session a tab's item shows, for owners that keep it (a workbench or notebook item), and how the item learns another one. */
export interface DocumentChatBinding {
  sessionId: string | undefined
  onSessionChange: (sessionId: string) => void
}

/** What `DocumentChat` and the chat's owner share about one file's Pi chat. */
export interface DocumentChat {
  /** The chat's link key: an absolute file, a LaTeX project folder or a canvas key. */
  path: string
  /** The open session, null while it opens. */
  opened: OpenedWorkbenchChat | null
  error: string | null
  /** The lines the next prompt starts with, undefined when the session already knows the file. */
  lead: string | undefined
  /** A PDF whose text is still extracted while the session still has to be told where it is: nothing may be sent yet. */
  pending: boolean
  onLeadSent: () => void
  /** Show session `sessionId` of the file. */
  select: (sessionId: string) => void
  /** Start a new session of the file on `options` (the shown session's model and thinking). */
  reset: (options: PiSessionOptions) => void
}

/**
 * The Pi `document` chat of `path` (`openWorkbenchChat`): the session `binding` names, else the file's active one, else a new one.
 *
 * - The backend links every session to `path`, so every device reopens the same session and lists the same ones
 * - `lead(cwd)` names the file to a new session (`File:`, `Project:`, `Canvas:`), and once more to a session `path` moved to (a renamed file, `markWorkbenchChatInformed`).
 *   A PDF's lead also names its extracted text (`Text: <path>`, G3); while that is extracted the chat is `pending` and polls until it is ready
 * - A session the item names that pi-host no longer has (404) falls back to the file's active one
 */
export function useDocumentChat(path: string, lead: (cwd: string) => string, binding?: DocumentChatBinding): DocumentChat {
  const [opened, setOpened] = useState<{ path: string; chat: OpenedWorkbenchChat } | null>(null)
  const [failure, setFailure] = useState<{ path: string; message: string } | null>(null)
  const [localId, setLocalId] = useState<string | undefined>(undefined)
  const [leadSent, setLeadSent] = useState<ReadonlySet<string>>(() => new Set())
  const requested = binding ? binding.sessionId : localId
  const current = opened?.path === path ? opened.chat : null
  const shownId = current?.session.sessionId

  const show = (chat: OpenedWorkbenchChat) => {
    setOpened({ path, chat })
    setFailure(null)
    const sessionId = chat.session.sessionId
    if (binding) {
      if (sessionId !== binding.sessionId) binding.onSessionChange(sessionId)
    } else setLocalId(sessionId)
  }
  const fail = (cause: unknown) => setFailure({ path, message: errorMessage(cause, "Could not open the chat of this file") })
  const onOpened = useEffectEvent(show)
  const onFailed = useEffectEvent(fail)

  useEffect(() => {
    if (shownId && (requested === undefined || requested === shownId)) return
    let cancelled = false
    openWorkbenchChat({ path, sessionId: requested })
      // A session the item names that pi-host lost falls back to the file's active one
      .catch((cause: unknown) => (requested !== undefined && cause instanceof ApiError && cause.status === 404 ? openWorkbenchChat({ path }) : Promise.reject(cause)))
      .then((chat) => { if (!cancelled) onOpened(chat) }, (cause: unknown) => { if (!cancelled) onFailed(cause) })
    return () => { cancelled = true }
  }, [path, requested, shownId])

  const owed = !!current && !!shownId && !leadSent.has(shownId) && (current.session.messageCount === 0 || current.link.moved.includes(shownId))
  const pdfText = current?.pdfText ?? null
  const pending = owed && pdfText?.state === "pending"

  usePolling(async (signal) => {
    try {
      const chat = await openWorkbenchChat({ path, sessionId: shownId })
      if (!signal.aborted) setOpened({ path, chat })
    } catch (cause) {
      if (!signal.aborted) fail(cause)
    }
  }, { intervalMs: PDF_TEXT_POLL_MS, active: pending, runWhileHidden: true })

  let shownLead: string | undefined
  if (owed && current && !pending) {
    const named = lead(current.cwd)
    shownLead = pdfText?.state === "ready" ? `${named}\nText: ${relativeIn(current.cwd, pdfText.path)}` : named
  }

  const onLeadSent = () => {
    if (!current || !shownId) return
    setLeadSent((sent) => new Set(sent).add(shownId))
    if (!current.link.moved.includes(shownId)) return
    markWorkbenchChatInformed(path, shownId).then(
      (link) => setOpened((held) => (held?.path === path && held.chat.session.sessionId === shownId ? { path, chat: { ...held.chat, link } } : held)),
      (cause: unknown) => setFailure({ path, message: errorMessage(cause, "Could not update the chat of this file") }),
    )
  }

  const select = (sessionId: string) => {
    if (sessionId === shownId) return
    if (binding) binding.onSessionChange(sessionId)
    else setLocalId(sessionId)
  }
  const reset = (options: PiSessionOptions) => {
    openWorkbenchChat({ path, new: true, ...options }).then(show, fail)
  }

  return {
    path,
    opened: current,
    error: failure?.path === path ? failure.message : null,
    lead: shownLead,
    pending,
    onLeadSent,
    select,
    reset,
  }
}
