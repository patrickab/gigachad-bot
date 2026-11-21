"use client"

import { useState } from "react"
import { createPortal } from "react-dom"
import { History, Plus, Trash2 } from "lucide-react"
import { ErrorText } from "@/components/ui/ErrorText"
import { MenuPopover } from "@/components/ui/MenuPopover"
import { FilePanel } from "@/features/editor/FilePanel"
import type { PiInsertion } from "@/features/pi/PiComposer"
import { PiSessionView, type PiShellCommands } from "@/features/pi/PiTab"
import { PiPanelBar, type PiCatalogProps, type PiSessionControls, type PiSessionHeader } from "@/features/pi/PiTopBar"
import { archivePiSession, fetchWorkbenchChats, type PiSessionInfo, type PiSessionOptions } from "@/lib/backend/api"
import { cn, errorMessage } from "@/lib/utils"
import { withoutFileLead } from "./fileLead"
import type { DocumentChat as DocumentChatState } from "./useDocumentChat"

const CENTERED_NOTE = "flex flex-1 items-center justify-center text-xs text-ink-faint"
const ICON_BUTTON = "rounded-md p-1 text-ink-subtle transition-colors hover:bg-surface-elevated hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"

/** A session's title in the menu: its first request without the lead that named the file. */
function titleOf(session: PiSessionInfo): string {
  return withoutFileLead(session.firstMessage).trim() || "New session"
}

/** What a new session starts on: the shown session's model and thinking level. */
function optionsOf(controls: PiSessionControls | null): PiSessionOptions {
  if (!controls) return {}
  const thinking = controls.thinkingLevel ?? undefined
  return controls.model ? { model: { provider: controls.model.provider, id: controls.model.id }, thinking } : { thinking }
}

/**
 * The file's sessions and a button starting a new one, for the chat panel's header.
 *
 * - The list loads when the menu opens: the sessions linked to the file that pi-host still has, newest first
 * - The trash button archives a session, archiving the shown one starts a new session
 */
function SessionMenu({ chat, onNew }: { chat: DocumentChatState; onNew: () => void }) {
  const [sessions, setSessions] = useState<PiSessionInfo[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const activeId = chat.opened?.session.sessionId ?? null
  const load = () => {
    fetchWorkbenchChats(chat.path).then(
      (listed) => {
        setSessions([...listed.sessions].sort((a, b) => b.modifiedAt.localeCompare(a.modifiedAt)))
        setError(null)
      },
      (cause: unknown) => setError(errorMessage(cause, "Could not list the sessions of this file")),
    )
  }
  const archive = async (sessionId: string) => {
    try {
      await archivePiSession(sessionId)
      setSessions((listed) => listed?.filter((session) => session.sessionId !== sessionId) ?? null)
      if (sessionId === activeId) onNew()
    } catch (cause) {
      setError(errorMessage(cause, "Could not archive the session"))
    }
  }

  return <>
    <MenuPopover
      className="relative"
      trigger={(open, toggle) => (
        <button
          type="button"
          aria-label="Sessions of this file"
          aria-expanded={open}
          onClick={() => {
            if (!open) load()
            toggle()
          }}
          className={ICON_BUTTON}
        >
          <History aria-hidden className="h-3.5 w-3.5" />
        </button>
      )}
    >
      {(open, close) => open && (
        <ul className="absolute right-0 top-full z-50 mt-1 flex max-h-[60vh] w-72 max-w-[80vw] flex-col overflow-y-auto rounded-2xl border border-divider/50 bg-surface p-1 text-xs shadow-[var(--shadow-lg)]">
          {sessions === null && !error && <li className="px-2.5 py-1.5 text-ink-faint">Loading…</li>}
          {sessions?.map((session) => (
            <li key={session.sessionId} className="group flex items-center rounded-xl transition-colors hover:bg-hover">
              <button
                type="button"
                aria-current={session.sessionId === activeId ? "true" : undefined}
                onClick={() => { close(); chat.select(session.sessionId) }}
                className={cn(
                  "min-w-0 flex-1 truncate rounded-xl px-2.5 py-1.5 text-left focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted",
                  session.sessionId === activeId ? "text-ink" : "text-ink-subtle",
                )}
              >
                {titleOf(session)}
              </button>
              <button
                type="button"
                aria-label={`Archive ${titleOf(session)}`}
                onClick={() => void archive(session.sessionId)}
                className={cn(ICON_BUTTON, "mr-1 shrink-0 opacity-0 group-hover:opacity-100 focus-visible:opacity-100")}
              >
                <Trash2 aria-hidden className="h-3.5 w-3.5" />
              </button>
            </li>
          ))}
          {error && <ErrorText className="px-2.5 py-1">{error}</ErrorText>}
        </ul>
      )}
    </MenuPopover>
    <button type="button" aria-label="New session" onClick={onNew} className={ICON_BUTTON}>
      <Plus aria-hidden className="h-3.5 w-3.5" />
    </button>
  </>
}

/**
 * One file's Pi chat (`useDocumentChat`) in the tab's file panel: `PiSessionView` below a compact header.
 *
 * - The header (`PiPanelBar` in the panel's top row) shows the session's live dot, thinking, model (the chats' dropdown over `modelProps`), name and
 *   context readout, then the file's sessions and New session
 * - A new session starts on the shown session's model and thinking level
 * - `insertion` (a quote, a canvas screenshot) goes into the composer once the session is ready
 * - A PDF whose text is still extracted shows that instead of the session until it is ready (`pending`)
 */
export function DocumentChat({ chat, width, onClose, shellCommands, modelProps, insertion, onInserted }: {
  chat: DocumentChatState
  width: number
  onClose: () => void
  shellCommands?: PiShellCommands
  modelProps: PiCatalogProps
  insertion?: PiInsertion | null
  onInserted?: () => void
}) {
  const [bar, setBar] = useState<HTMLDivElement | null>(null)
  const { opened, error, pending } = chat
  const inBar = (header: PiSessionHeader | null) => bar && createPortal(
    <PiPanelBar header={header} modelProps={modelProps} actions={<SessionMenu chat={chat} onNew={() => chat.reset(optionsOf(header?.controls ?? null))} />} />,
    bar,
  )

  return (
    <FilePanel label="Chat" bare onClose={onClose} width={width} actions={<div ref={setBar} className="flex min-w-0 flex-1" />}>
      {opened && error && <ErrorText className="px-4 pt-2">{error}</ErrorText>}
      {opened && !pending ? (
        <PiSessionView
          docked
          key={opened.session.sessionId}
          sessionId={opened.session.sessionId}
          cwd={opened.cwd}
          files={{ cwd: opened.cwd }}
          renderHeader={inBar}
          shellCommands={shellCommands}
          insertion={insertion}
          onInserted={onInserted}
          lead={chat.lead}
          onLeadSent={chat.onLeadSent}
        />
      ) : <>
        {inBar(null)}
        <div role="status" className={CENTERED_NOTE}>
          {error && !opened ? <ErrorText>{error}</ErrorText> : pending ? "Extracting the PDF text…" : "Loading…"}
        </div>
      </>}
    </FilePanel>
  )
}
