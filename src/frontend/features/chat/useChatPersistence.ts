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
    await saveChat(tab.historyFile, projectSlug, messagesToSave, { chatId, tabName: tab.name ?? undefined, title, usage: usageToSave })
    await refreshAll()
  }, [tab.historyFile, tab.title, tab.name, projectSlug, chatId, refreshAll])

  const handleQuickSave = useCallback(async () => {
    await saveHistory(messages, usage)
  }, [saveHistory, messages, usage])

  // Saves are chained so overlapping autosaves never write out of order.
  const handleAutosave = useCallback((messagesToSave: Message[], usageToSave: Usage) => {
    if (!tab.historyFile) return Promise.resolve()
    const queued = autosaveRef.current.catch(() => {}).then(() => saveHistory(messagesToSave, usageToSave))
    autosaveRef.current = queued
    return queued.catch(() => {})
  }, [tab.historyFile, saveHistory])

  const handleSaveSubmit = useCallback(async (name: string) => {
    const historyFile = buildHistoryFile(name + ".json", projectSlug)
    // A failed save is swallowed so the tab still adopts the new history file.
    await saveChat(historyFile, projectSlug, messages, { chatId, tabName: tab.name ?? undefined, title: name, usage }).catch(() => {})
    setHistoryFile(historyFile)
    await refreshAll()
    setSaveModalOpen(false)
  }, [messages, chatId, tab.name, projectSlug, usage, setHistoryFile, refreshAll])

  return {
    isTitled,
    saveModalOpen,
    setSaveModalOpen,
    handleQuickSave,
    handleAutosave,
    handleSaveSubmit,
  }
}
