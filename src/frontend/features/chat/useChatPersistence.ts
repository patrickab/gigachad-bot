"use client"

import { useCallback, useRef, useState } from "react"
import type { Tab } from "@/contexts/TabContext"
import { saveChatHistory as apiSaveChatHistory, saveProjectTab } from "@/lib/backend/api"
import { buildHistoryFile, parseHistoryFile } from "@/lib/domain/historyFile"
import type { Message, Usage } from "@/lib/domain/types"

/** Save a tab's chat: explicit save-as, quick save, and chained autosaves once it has a history file. */
export function useChatPersistence({
  tab,
  activeProject,
  messages,
  chatId,
  usage,
  refreshAll,
  setHistoryFile,
}: {
  tab: Tab
  activeProject: string | null
  messages: Message[]
  chatId: string
  usage: Usage | undefined
  refreshAll: () => Promise<void>
  setHistoryFile: (historyFile: string) => void
}) {
  const [saveModalOpen, setSaveModalOpen] = useState(false)
  const autosaveRef = useRef(Promise.resolve())

  const isTitled = !!tab.historyFile

  // No-op for untitled chats. Routes to the project store when the history file belongs to the active project.
  const saveHistory = useCallback(async (messagesToSave: Message[], usageToSave: Usage | undefined) => {
    if (!tab.historyFile) return
    const { slug, filename } = parseHistoryFile(tab.historyFile)
    const title = tab.title ?? filename.replace(".json", "")
    if (slug && slug === activeProject) {
      await saveProjectTab(activeProject, filename, messagesToSave, { chatId, tabName: tab.name ?? undefined, title, usage: usageToSave })
    } else {
      await apiSaveChatHistory(tab.historyFile, messagesToSave, { chatId, title, usage: usageToSave })
    }
    await refreshAll()
  }, [tab.historyFile, tab.title, tab.name, activeProject, chatId, refreshAll])

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
    const newFilename = name + ".json"

    // Save failures are swallowed so the tab still adopts the new history file.
    if (activeProject) {
      try {
        await saveProjectTab(activeProject, newFilename, messages, { chatId, tabName: tab.name ?? undefined, title: name, usage })
      } catch { }
      setHistoryFile(buildHistoryFile(newFilename, activeProject))
    } else {
      try {
        await apiSaveChatHistory(newFilename, messages, { chatId, title: name, usage })
      } catch { }
      setHistoryFile(newFilename)
    }
    await refreshAll()
    setSaveModalOpen(false)
  }, [messages, chatId, tab.name, activeProject, usage, setHistoryFile, refreshAll])

  return {
    isTitled,
    saveModalOpen,
    setSaveModalOpen,
    handleQuickSave,
    handleAutosave,
    handleSaveSubmit,
  }
}
