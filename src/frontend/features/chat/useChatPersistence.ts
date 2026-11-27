"use client"

import { useCallback, useRef, useState } from "react"
import type { Tab } from "@/contexts/TabContext"
import { saveChat } from "@/lib/backend/api"
import { buildHistoryFile, parseHistoryFile } from "@/lib/domain/historyFile"
import type { Message, Usage } from "@/lib/domain/types"

/** Save a tab's chat: explicit save-as, quick save, and chained autosaves once it has a history file. */
export function useChatPersistence({
  tab,
  projectSlug,
  messages,
  chatId,
  usage,
  refreshAll,
  setHistoryFile,
}: {
  tab: Tab
  projectSlug: string | null
  messages: Message[]
  chatId: string
  usage: Usage | undefined
  refreshAll: () => Promise<void>
  setHistoryFile: (historyFile: string) => void
}) {
  const [saveModalOpen, setSaveModalOpen] = useState(false)
  const autosaveRef = useRef(Promise.resolve())

  const isTitled = !!tab.historyFile

  // No-op for untitled chats. Saved histories retain their own project scope.
  const saveHistory = useCallback(async (messagesToSave: Message[], usageToSave: Usage | undefined) => {
    if (!tab.historyFile) return
    const title = tab.title ?? parseHistoryFile(tab.historyFile).filename.replace(".json", "")
    await saveChat(tab.historyFile, projectSlug, messagesToSave, { chatId, title, usage: usageToSave })
    // The file is written: a failed refresh of the lists must not read as a failed save
    await refreshAll().catch(() => {})
  }, [tab.historyFile, tab.title, projectSlug, chatId, refreshAll])

  const handleQuickSave = useCallback(async () => {
    await saveHistory(messages, usage)
  }, [saveHistory, messages, usage])

  // Saves are chained so overlapping saves never write out of order. A failed one rejects: the caller tells the reader.
  const handleSave = useCallback((messagesToSave: Message[], usageToSave: Usage) => {
    if (!tab.historyFile) return Promise.resolve()
    const queued = autosaveRef.current.catch(() => {}).then(() => saveHistory(messagesToSave, usageToSave))
    autosaveRef.current = queued
    return queued
  }, [tab.historyFile, saveHistory])

  const handleAutosave = useCallback((messagesToSave: Message[], usageToSave: Usage) => {
    return handleSave(messagesToSave, usageToSave).catch(() => {})
  }, [handleSave])

  const handleSaveSubmit = useCallback(async (name: string) => {
    const historyFile = buildHistoryFile(name + ".json", projectSlug)
    // A failed save is swallowed so the tab still adopts the new history file.
    await saveChat(historyFile, projectSlug, messages, { chatId, title: name, usage }).catch(() => {})
    setHistoryFile(historyFile)
    await refreshAll()
    setSaveModalOpen(false)
  }, [messages, chatId, projectSlug, usage, setHistoryFile, refreshAll])

  return {
    isTitled,
    saveModalOpen,
    setSaveModalOpen,
    handleQuickSave,
    handleSave,
    handleAutosave,
    handleSaveSubmit,
  }
}
