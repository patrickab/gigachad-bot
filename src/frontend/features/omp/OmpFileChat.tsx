"use client"

import { useEffect, useState } from "react"
import { ErrorText } from "@/components/ui/ErrorText"
import { FilePanel } from "@/features/editor/FilePanel"
import { prepareWorkbench, type PreparedWorkbench } from "@/lib/backend/api"
import { errorMessage } from "@/lib/utils"
import type { ShellCommands } from "./OmpComposer"
import { OmpChatPanel } from "./OmpChatPanel"
import { useFileChat } from "./useFileChat"

interface Prepared extends PreparedWorkbench {
  requested: string
}

/**
 * The omp chat of a file its owner only views, e.g. a PDF: sessions, menu and New session like the workbench, no editor.
 *
 * - The backend resolves the file to read on disk and the session directory of its kind, nothing moves
 * - Which session shows lives only in the file's links, the owner keeps no view state
 */
export function OmpFileChat({ path, width, onClose, shellCommands }: {
  path: string
  width: number
  onClose: () => void
  shellCommands: ShellCommands
}) {
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
      <FilePanel label="Chat" bare onClose={onClose} width={width}>
        <div className="flex flex-1 items-center justify-center text-xs text-ink-faint">{error ? <ErrorText>{error}</ErrorText> : "Loading…"}</div>
      </FilePanel>
    )
  }
  return <PreparedFileChat path={prepared.path} cwd={prepared.cwd} width={width} onClose={onClose} shellCommands={shellCommands} />
}

function PreparedFileChat({ path, cwd, width, onClose, shellCommands }: {
  path: string
  cwd: string
  width: number
  onClose: () => void
  shellCommands: ShellCommands
}) {
  const chat = useFileChat(path, cwd, undefined, () => {})
  return <OmpChatPanel chat={chat} width={width} onClose={onClose} shellCommands={shellCommands} />
}
