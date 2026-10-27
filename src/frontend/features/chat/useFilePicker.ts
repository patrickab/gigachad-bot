"use client"

import { useCallback, useState, type Dispatch, type RefObject, type SetStateAction } from "react"
import { useStartupRead } from "@/contexts/StartupReadContext"
import { useEventListener } from "@/hooks/useEventListener"
import type { ChatInputHandle } from "@/features/chat/composer/ChatInput"
import { attachReference, listRootFiles } from "@/lib/backend/api"
import type { RootFile } from "@/lib/domain/types"

export function useFilePicker({ isActive, hasMessages, chatInputRef, setExtracting }: {
  isActive: boolean
  hasMessages: boolean
  chatInputRef: RefObject<ChatInputHandle | null>
  setExtracting: Dispatch<SetStateAction<number>>
}) {
  const [filesEnabled, setFilesEnabled] = useState(false)
  const [pickerFiles, setRootFiles] = useState<RootFile[]>([])
  const [filePickerOpen, setFilePickerOpen] = useState(false)
  const [fileEditPath, setFileEditPath] = useState<string | null>(null)
  const [filePdfPath, setFilePdfPath] = useState<string | null>(null)

  useStartupRead("Root files", listRootFiles, (result) => {
    setFilesEnabled(result.enabled)
    setRootFiles(result.files)
  })

  const refreshFileList = useCallback(() => {
    listRootFiles()
      .then((r) => { setFilesEnabled(r.enabled); setRootFiles(r.files) })
      .catch(() => {})
  }, [])

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
    if (!hasMessages) {
      if (path.toLowerCase().endsWith(".pdf")) setFilePdfPath(path)
      else setFileEditPath(path)
    } else {
      setExtracting((n) => n + 1)
      try { chatInputRef.current?.addAttachment(await attachReference(path)) } catch { /* attach failure leaves the input unchanged */ }
      finally { setExtracting((n) => n - 1) }
    }
  }, [hasMessages, chatInputRef, setExtracting])

  // Capture phase so Escape closes the PDF viewer before other handlers see it.
  useEventListener("window", "keydown", (e) => {
    if (e.key === "Escape") { e.preventDefault(); setFilePdfPath(null) }
  }, { active: isActive && !!filePdfPath, capture: true })

  return {
    filesEnabled,
    pickerFiles,
    filePickerOpen,
    setFilePickerOpen,
    fileEditPath,
    setFileEditPath,
    filePdfPath,
    setFilePdfPath,
    refreshFileList,
    openFilePicker,
    handleFileSelect,
  }
}
