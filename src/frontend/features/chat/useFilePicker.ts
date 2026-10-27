"use client"

import { useCallback, useState, type RefObject } from "react"
import { useStartupRead } from "@/contexts/StartupReadContext"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import type { ChatInputHandle } from "@/features/chat/composer/ChatInput"
import { attachToComposer } from "@/features/chat/attachToComposer"
import { listRootFiles } from "@/lib/backend/api"
import type { RootFile } from "@/lib/domain/files"

/** A picked file opened on its own: a PDF viewer or a text editor. */
export type StandaloneFile = { kind: "pdf" | "edit"; path: string }

const NO_ROOT_FILES = { enabled: false, files: [] as RootFile[] }

/** Own the mounted-file picker: the root file listing, standalone opening in an empty chat, and staging on the composer otherwise. */
export function useFilePicker({ hasMessages, chatInputRef, track }: {
  hasMessages: boolean
  chatInputRef: RefObject<ChatInputHandle | null>
  track: <T>(work: () => Promise<T>) => Promise<T>
}) {
  const [rootFiles, setRootFiles] = useState<{ enabled: boolean; files: RootFile[] }>(NO_ROOT_FILES)
  const [filePickerOpen, setFilePickerOpen] = useState(false)
  const [standaloneFile, setStandaloneFile] = useState<StandaloneFile | null>(null)

  const applyRootFiles = useCallback(({ enabled, files }: { enabled: boolean; files: RootFile[] }) => setRootFiles({ enabled, files }), [])
  useStartupRead("Root files", listRootFiles, applyRootFiles)

  const refreshFileList = useCallback(() => {
    listRootFiles().then(applyRootFiles).catch(() => {})
  }, [applyRootFiles])

  const openFilePicker = useCallback(() => {
    setFilePickerOpen(true)
    // Refresh the listing on open so file root changes are reflected.
    refreshFileList()
  }, [refreshFileList])

  const handleFileSelect = useCallback(async (path: string) => {
    setFilePickerOpen(false)
    // Empty chat → open standalone (edit text files, view PDFs); active
    // conversation → stage a live reference on the chat input (the *next*
    // message), not the already-sent latest message.
    if (!hasMessages) setStandaloneFile({ kind: path.toLowerCase().endsWith(".pdf") ? "pdf" : "edit", path })
    else await attachToComposer(chatInputRef, track, path)
  }, [hasMessages, chatInputRef, track])

  useEscapeLayer(() => setStandaloneFile(null), standaloneFile?.kind === "pdf")

  return {
    rootFiles,
    filePickerOpen,
    setFilePickerOpen,
    standaloneFile,
    setStandaloneFile,
    refreshFileList,
    openFilePicker,
    handleFileSelect,
  }
}
