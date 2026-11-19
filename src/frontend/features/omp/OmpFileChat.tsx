"use client"

import { useEffect, useState } from "react"
import { ErrorText } from "@/components/ui/ErrorText"
import { FilePanel } from "@/features/editor/FilePanel"
import { prepareWorkbench, type PreparedWorkbench } from "@/lib/backend/api"
import { errorMessage } from "@/lib/utils"
import type { ComposerInsertion, ShellCommands } from "./OmpComposer"
import { OmpChatPanel } from "./OmpChatPanel"
import { useFileChat } from "./useFileChat"

interface Prepared extends PreparedWorkbench {
  requested: string
}

interface FileChatProps {
  width: number
  onClose: () => void
  shellCommands: ShellCommands
  /** Line leading a new session's first message, by default the file's `File:` line. */
  lead?: string
  /** Text and files to put into the composer draft, e.g. a canvas screenshot. */
  insertion?: ComposerInsertion | null
  onInserted?: () => void
}

/**
 * The omp chat of a file its owner only views, e.g. a PDF or a canvas: sessions, menu and New session like the workbench, no editor.
 *
 * - The backend resolves the file to read on disk and the session directory of its kind, nothing moves. A canvas stays stored, its key links its sessions
 * - Which session shows lives only in the file's links, the owner keeps no view state
 */
export function OmpFileChat({ path, ...props }: FileChatProps & { path: string }) {
  const [prepared, setPrepared] = useState<Prepared | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    prepareWorkbench(path).then(
      (file) => { if (!cancelled) setPrepared({ requested: path, ...file }) },
      (cause: unknown) => { if (!cancelled) setError(errorMessage(cause, "Could not open the chat of this file")) },
    )
    return () => { cancelled = true }
  }, [path])

  if (prepared?.requested !== path) {
    return (
      <FilePanel label="Chat" bare onClose={props.onClose} width={props.width}>
        <div className="flex flex-1 items-center justify-center text-xs text-ink-faint">{error ? <ErrorText>{error}</ErrorText> : "Loading…"}</div>
      </FilePanel>
    )
  }
  return <PreparedFileChat path={prepared.path} cwd={prepared.cwd} {...props} />
}

function PreparedFileChat({ path, cwd, width, onClose, shellCommands, lead, insertion, onInserted }: FileChatProps & { path: string; cwd: string }) {
  const chat = useFileChat(path, cwd, undefined, () => {}, lead)
  return <OmpChatPanel chat={chat} width={width} onClose={onClose} shellCommands={shellCommands} insertion={insertion} onInserted={onInserted} />
}
