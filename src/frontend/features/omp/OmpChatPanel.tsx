"use client"

import { ErrorText } from "@/components/ui/ErrorText"
import { FilePanel } from "@/features/editor/FilePanel"
import { ChatSessionMenu } from "./ChatSessionMenu"
import type { ComposerInsertion, ShellCommands } from "./OmpComposer"
import { OmpSurface } from "./OmpSurface"
import type { FileChat } from "./useFileChat"

const CENTERED_NOTE = "flex h-full items-center justify-center text-xs text-ink-faint"

/**
 * One file's omp chat in the tab's file panel: its session menu and New session in the header, the session below.
 *
 * - `chat` comes from `useFileChat` in the owner, which also reads its `view` (streaming, top-bar model controls)
 * - The file's line leading a message comes from `chat` (`useFileChat`)
 */
export function OmpChatPanel({ chat, width, onClose, shellCommands, insertion, onInserted }: {
  chat: FileChat
  width: number
  onClose: () => void
  shellCommands: ShellCommands
  /** Text to put into the composer draft, e.g. a quoted selection. */
  insertion?: ComposerInsertion | null
  onInserted?: () => void
}) {
  const { view, adopt, reset, select, sessionIds, error, lead, onLeadSent } = chat
  return (
    <FilePanel
      label="Chat"
      bare
      onClose={onClose}
      width={width}
      actions={<ChatSessionMenu view={view} sessionIds={sessionIds} onSelect={select} onReset={reset} />}
    >
      {view && <ErrorText className="px-4 pt-2">{error}</ErrorText>}
      <div className="relative min-h-0 flex-1">
        {view ? (
          <OmpSurface
            view={view}
            onViewChange={adopt}
            shellCommands={shellCommands}
            topInset={0}
            adoptTitle={false}
            backdrop="surface"
            insertion={insertion}
            onInserted={onInserted}
            lead={lead}
            onLeadSent={onLeadSent}
          />
        ) : (
          <div className={CENTERED_NOTE}>{error ? <ErrorText>{error}</ErrorText> : "Loading…"}</div>
        )}
      </div>
    </FilePanel>
  )
}
