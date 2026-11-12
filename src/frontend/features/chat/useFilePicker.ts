"use client"

import { useCallback, useState, type RefObject } from "react"
import { useResource } from "@/hooks/useResource"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import type { ChatInputHandle } from "@/features/chat/composer/ChatInput"
import { attachToComposer } from "@/features/chat/attachToComposer"
import { listRootFiles } from "@/lib/backend/api"
import { fileRootsEnabledResource } from "@/lib/backend/resources"
import type { RootFile } from "@/lib/domain/files"
import type { WorkspaceItem } from "@/contexts/TabContext"
import { isNotebookPath } from "@/lib/paths"

/** A picked file opened on its own: a PDF viewer or a text editor. */
export type StandaloneFile = { kind: "pdf" | "edit"; path: string }

/** `files` is null until the picker first opens: listing walks every root, startup only needs `enabled`. */
type RootFiles = { enabled: boolean; files: RootFile[] | null }

/** Own the mounted-file picker: the root file listing, standalone opening in an empty chat, and staging on the composer otherwise. */
export function useFilePicker({ hasMessages, chatInputRef, track, onOpenItem }: {
  hasMessages: boolean
  chatInputRef: RefObject<ChatInputHandle | null>
  track: <T>(work: () => Promise<T>) => Promise<T>
  /** Opens a picked notebook in a tab, it has no overlay form. */
  onOpenItem: (item: WorkspaceItem) => void
}) {
  const enabled = useResource("Root files", fileRootsEnabledResource) ?? false
  const [files, setFiles] = useState<RootFile[] | null>(null)
  const [filePickerOpen, setFilePickerOpen] = useState(false)
  const [standaloneFile, setStandaloneFile] = useState<StandaloneFile | null>(null)
  const rootFiles: RootFiles = { enabled, files }

  const openFilePicker = useCallback(() => {
    setFilePickerOpen(true)
    // Load on open, so the listing reflects file root changes without walking every root at startup.
    listRootFiles().then((listing) => {
      fileRootsEnabledResource.set(listing.enabled)
      setFiles(listing.files)
    }).catch(() => {})
  }, [])

  // Opens a file without a chat: a notebook in a tab, a PDF or a text file standalone
  const openFile = useCallback((path: string) => {
    if (isNotebookPath(path)) {
      setStandaloneFile(null)
      onOpenItem({ kind: "document", path, projectSlug: null, rootFile: true })
    } else setStandaloneFile({ kind: path.toLowerCase().endsWith(".pdf") ? "pdf" : "edit", path })
  }, [onOpenItem])

  const handleFileSelect = useCallback(async (path: string) => {
    setFilePickerOpen(false)
    // Empty chat → open the file; active conversation → stage a live reference on the chat input (the *next*
    // message), not the already-sent latest message.
    if (hasMessages) await attachToComposer(chatInputRef, track, path)
    else openFile(path)
  }, [hasMessages, chatInputRef, track, openFile])

  useEscapeLayer(() => setStandaloneFile(null), standaloneFile?.kind === "pdf")

  return {
    rootFiles,
    filePickerOpen,
    setFilePickerOpen,
    standaloneFile,
    setStandaloneFile,
    openFilePicker,
    handleFileSelect,
    openFile,
  }
}
