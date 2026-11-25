"use client"

import { useEffect, useState } from "react"
import { ErrorText } from "@/components/ui/ErrorText"
import { FilePanel } from "@/features/editor/FilePanel"
import type { PiInsertion } from "@/features/pi/PiComposer"
import type { PiCatalogProps } from "@/features/pi/PiTopBar"
import { prepareWorkbench } from "@/lib/backend/api"
import { errorMessage } from "@/lib/utils"
import { DocumentChat } from "./DocumentChat"
import { fileLead } from "./fileLead"
import { useDocumentChat } from "./useDocumentChat"

interface FileChatProps {
  width: number
  onClose: () => void
  /** The chats' model catalog, for the header's model dropdown. */
  modelProps: PiCatalogProps
  /** Names the file to a new session in `cwd`, by default its `File:` line. */
  lead?: (cwd: string) => string
  /** Text and files to put into the composer draft, e.g. a canvas screenshot. */
  insertion?: PiInsertion | null
  onInserted?: () => void
}

/**
 * The Pi chat of a file its owner only views, a PDF or a canvas: sessions, menu and New session like the workbench, no editor.
 *
 * - The backend resolves the file to read on disk (a stored PDF's library copy), nothing moves. A canvas stays stored, its key links its sessions
 * - Which session shows lives only in the file's links, the owner keeps no view state
 */
export function FileChat({ path, ...props }: FileChatProps & { path: string }) {
  const [prepared, setPrepared] = useState<{ requested: string; path: string } | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    prepareWorkbench(path).then(
      (file) => { if (!cancelled) setPrepared({ requested: path, path: file.path }) },
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
  return <PreparedFileChat key={prepared.path} path={prepared.path} {...props} />
}

function PreparedFileChat({ path, lead, ...props }: FileChatProps & { path: string }) {
  const chat = useDocumentChat(path, lead ?? ((cwd) => fileLead(path, cwd)))
  return <DocumentChat chat={chat} {...props} />
}
