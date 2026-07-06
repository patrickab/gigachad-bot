"use client"

import { useCallback, useEffect, useState, type Dispatch, type RefObject, type SetStateAction } from "react"
import type { ChatInputHandle } from "@/components/ChatInput"
import { attachFileVaultFile, listFileVaultFiles } from "@/lib/api"
import type { VaultFile } from "@/lib/types"

export function useVaultPicker({ isActive, hasMessages, chatInputRef, setExtracting }: {
  isActive: boolean
  hasMessages: boolean
  chatInputRef: RefObject<ChatInputHandle | null>
  setExtracting: Dispatch<SetStateAction<number>>
}) {
  const [vaultEnabled, setVaultEnabled] = useState(false)
  const [vaultFiles, setVaultFiles] = useState<VaultFile[]>([])
  const [vaultPickerOpen, setVaultPickerOpen] = useState(false)
  const [vaultEditPath, setVaultEditPath] = useState<string | null>(null)
  const [vaultPdfPath, setVaultPdfPath] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    listFileVaultFiles()
      .then((r) => { if (!cancelled) { setVaultEnabled(r.enabled); setVaultFiles(r.files) } })
      .catch(() => {})
    return () => { cancelled = true }
  }, [])

  const refreshVaultList = useCallback(() => {
    listFileVaultFiles()
      .then((r) => { setVaultEnabled(r.enabled); setVaultFiles(r.files) })
      .catch(() => {})
  }, [])

  const openVaultPicker = useCallback(() => {
    setVaultPickerOpen(true)
    // Refresh the listing on open so vault changes are reflected.
    refreshVaultList()
  }, [refreshVaultList])

  const handleVaultSelect = useCallback(async (path: string) => {
    setVaultPickerOpen(false)
    // Empty chat → open standalone (edit text files, view PDFs); active
    // conversation → stage a live reference on the chat input (the *next*
    // message), not the already-sent latest message.
    if (!hasMessages) {
      if (path.toLowerCase().endsWith(".pdf")) setVaultPdfPath(path)
      else setVaultEditPath(path)
    } else {
      setExtracting((n) => n + 1)
      try { chatInputRef.current?.addAttachment(await attachFileVaultFile(path)) } catch { /* */ }
      finally { setExtracting((n) => n - 1) }
    }
  }, [hasMessages, chatInputRef, setExtracting])

  useEffect(() => {
    if (!isActive || !vaultPdfPath) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") { e.preventDefault(); setVaultPdfPath(null) }
    }
    window.addEventListener("keydown", onKey, true)
    return () => window.removeEventListener("keydown", onKey, true)
  }, [isActive, vaultPdfPath])

  return {
    vaultEnabled,
    vaultFiles,
    vaultPickerOpen,
    setVaultPickerOpen,
    vaultEditPath,
    setVaultEditPath,
    vaultPdfPath,
    setVaultPdfPath,
    refreshVaultList,
    openVaultPicker,
    handleVaultSelect,
  }
}
