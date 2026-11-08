"use client"

import { useCallback, useState, type RefObject } from "react"
import { useStartupRead } from "@/contexts/StartupReadContext"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import type { ChatInputHandle } from "@/features/chat/composer/ChatInput"
import { attachToComposer } from "@/features/chat/attachToComposer"
import { fileRootsEnabled, listRootFiles } from "@/lib/backend/api"
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
  const [rootFiles, setRootFiles] = useState<RootFiles>({ enabled: false, files: null })
  const [filePickerOpen, setFilePickerOpen] = useState(false)
  const [standaloneFile, setStandaloneFile] = useState<StandaloneFile | null>(null)

  const applyRootFiles = useCallback(({ enabled, files }: { enabled: boolean; files: RootFile[] }) => setRootFiles({ enabled, files }), [])
  const applyEnabled = useCallback(({ enabled }: { enabled: boolean }) => setRootFiles((current) => ({ ...current, enabled })), [])
  useStartupRead("Root files", fileRootsEnabled, applyEnabled)

  const openFilePicker = useCallback(() => {
    setFilePickerOpen(true)
    // Load on open, so the listing reflects file root changes without walking every root at startup.
    listRootFiles().then(applyRootFiles).catch(() => {})
  }, [applyRootFiles])

  /** Mounting or unmounting a root can flip `enabled`, the next open lists the files again anyway. */
  const refreshRoots = useCallback(() => {
    fileRootsEnabled().then(applyEnabled).catch(() => {})
  }, [applyEnabled])

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
    refreshRoots,
    openFilePicker,
    handleFileSelect,
    openFile,
  }
}
