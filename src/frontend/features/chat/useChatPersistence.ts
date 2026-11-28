"use client"

import { useCallback, useRef } from "react"
import type { Tab } from "@/contexts/TabContext"
import type { ChatStore } from "@/features/chat/chatStore"
import { saveChat } from "@/lib/backend/api"
import { buildHistoryFile, parseHistoryFile } from "@/lib/domain/historyFile"
import type { Message, Usage } from "@/lib/domain/types"

/** A chat's usage as saved: none before its first reply. */
const savedUsage = (usage: Usage) => usage.total_tokens > 0 ? usage : undefined

/** Save a tab's chat: explicit save-as, quick save, and chained autosaves once it has a history file. */
export function useChatPersistence({
  tab,
  projectSlug,
  chat,
  chatId,
  refreshAll,
  setHistoryFile,
  setTitle,
}: {
  tab: Tab
  projectSlug: string | null
  chat: ChatStore
  chatId: string
  refreshAll: () => Promise<void>
  setHistoryFile: (historyFile: string) => void
  setTitle: (title: string) => void
}) {
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
    const { messages, usage } = chat.get()
    await saveHistory(messages, savedUsage(usage))
  }, [saveHistory, chat])

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
    const { messages, usage } = chat.get()
    await saveChat(historyFile, projectSlug, messages, { chatId, title: name, usage: savedUsage(usage) }).catch(() => {})
    // The tab keeps its messages under the new file (see useTabChat), so it takes the title the file holds
    setTitle(name)
    setHistoryFile(historyFile)
    await refreshAll()
  }, [chat, chatId, projectSlug, setHistoryFile, setTitle, refreshAll])

  return {
    isTitled,
    handleQuickSave,
    handleSave,
    handleAutosave,
    handleSaveSubmit,
  }
}
