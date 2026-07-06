"use client"

import { useCallback, useState, type Dispatch, type SetStateAction } from "react"
import type { Tab } from "@/components/TabManager"
import { updateLastMsg as updateLastAssistant } from "@/hooks/useStudyHandler"
import {
  saveChatHistory as apiSaveChatHistory,
  buildHistoryFile,
  generateMindmap,
  parseHistoryFile,
  saveProjectTab,
} from "@/lib/api"
import type { Attachment, Message, Usage } from "@/lib/types"

export function useChatModals({
  tab,
  activeProject,
  messages,
  chatId,
  hasUsage,
  selectedModel,
  refreshAll,
  onHistoryFileChanged,
  setMessages,
}: {
  tab: Tab
  activeProject: string | null
  messages: Message[]
  chatId: string
  hasUsage: Usage | undefined
  selectedModel: string
  refreshAll: () => Promise<void>
  onHistoryFileChanged: (tabId: string, historyFile: string) => void
  setMessages: Dispatch<SetStateAction<Message[]>>
}) {
  const [saveModalOpen, setSaveModalOpen] = useState(false)
  const [mindmapModalOpen, setMindmapModalOpen] = useState(false)
  const [mindmapAttachments, setMindmapAttachments] = useState<Attachment[]>([])
  const [promptEditorOpen, setPromptEditorOpen] = useState(false)

  const isTitled = !!tab.historyFile

  const handleQuickSave = useCallback(async () => {
    if (!tab.historyFile) return
    const { slug, filename } = parseHistoryFile(tab.historyFile)
    const title = tab.title ?? filename.replace(".json", "")
    if (slug && slug === activeProject) {
      await saveProjectTab(activeProject!, filename, messages, chatId, tab.name ?? undefined, title, hasUsage)
    } else {
      await apiSaveChatHistory(tab.historyFile, messages, chatId, title, hasUsage)
    }
    await refreshAll()
  }, [tab.historyFile, tab.title, tab.name, activeProject, messages, chatId, hasUsage, refreshAll])

  const handleSaveSubmit = useCallback(async (name: string) => {
    const newFilename = name + ".json"

    if (activeProject) {
      try {
        await saveProjectTab(activeProject, newFilename, messages, chatId, tab.name ?? undefined, name, hasUsage)
      } catch { }
      onHistoryFileChanged(tab.id, buildHistoryFile(newFilename, activeProject))
    } else {
      try {
        await apiSaveChatHistory(newFilename, messages, chatId, name, hasUsage)
      } catch { }
      onHistoryFileChanged(tab.id, newFilename)
    }
    await refreshAll()
    setSaveModalOpen(false)
  }, [messages, chatId, tab.id, tab.name, activeProject, hasUsage, onHistoryFileChanged, refreshAll])

  const handleMindmapSubmit = useCallback(async (prompt: string, attachments: Attachment[] = []) => {
    if (messages.length === 0) return
    const userMsg = prompt ? `Provide a mindmap. ${prompt}` : "Provide a mindmap."
    setMessages(prev => [
      ...prev,
      { role: "user" as const, content: userMsg },
      { role: "assistant" as const, content: "Generating mind map…" },
    ])
    setMindmapModalOpen(false)
    try {
      const context = attachments.map(a => a.parsedMd || a.content || "").filter(Boolean).join("\n\n")
      const mindmapMessages = messages.map((m) => ({ role: m.role, content: m.content }))
      if (context) mindmapMessages.push({ role: "user" as const, content: context })

      const mindmap = await generateMindmap(
        mindmapMessages,
        selectedModel,
        prompt,
      )
      updateLastAssistant(setMessages, m => ({ ...m, content: mindmap }))
    } catch {
      updateLastAssistant(setMessages, m => ({ ...m, content: "Mind map generation failed." }))
    }
  }, [messages, selectedModel, setMessages])

  return {
    isTitled,
    saveModalOpen,
    setSaveModalOpen,
    handleQuickSave,
    handleSaveSubmit,
    mindmapModalOpen,
    setMindmapModalOpen,
    mindmapAttachments,
    setMindmapAttachments,
    handleMindmapSubmit,
    promptEditorOpen,
    setPromptEditorOpen,
  }
}
