"use client"

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react"
import { Sidebar } from "@/components/Sidebar"
import { ChatContainer } from "@/components/ChatContainer"
import { MAX_SIDEBAR_WIDTH } from "@/components/ChatSidebar"
import { ModelSelector } from "@/components/ModelSelector"
import { MoreOptionsMenu } from "@/components/MoreOptionsMenu"
import { ReasoningSelector } from "@/components/ReasoningSelector"
import { ResearchModelsBar } from "@/components/ResearchModelsBar"
import { ThemeToggle } from "@/components/ThemeToggle"
import { OCRPanel } from "@/components/OCRPanel"
import { SaveChatModal } from "@/components/SaveChatModal"
import { TabManager, nextTab } from "@/components/TabManager"
import type { Tab, TabManagerHandle } from "@/components/TabManager"
import { ProjectDashboard } from "@/components/ProjectDashboard"
import { TokenCounter } from "@/components/TokenCounter"
import { useChat } from "@/hooks/useChat"
import { useModeState, ModeProvider } from "@/hooks/useModeState"
import { useSettings, SettingsProvider } from "@/contexts/SettingsContext"
import { useProject, ProjectProvider } from "@/contexts/ProjectContext"
import { handleStudyPdf, updateLastMsg as updateLastAssistant } from "@/hooks/useStudyHandler"
import {
  saveChatHistory as apiSaveChatHistory,
  deleteChatHistory as apiDeleteChatHistory,
  loadChatHistory as apiLoadChatHistory,
  deleteChatUploads,
  parseFiles,
  deleteAttachment,
  saveProjectTab,
  deleteProjectTab,
  loadProjectTab,
  renameChatHistory,
} from "@/lib/api"
import { DEFAULT_VISION_MODEL } from "@/lib/config"
import type { Attachment, Message, Usage } from "@/lib/types"

function shortIdFromChatId(chatId: string): string {
  const cleaned = chatId.replace(/[^a-zA-Z0-9]/g, "")
  return cleaned.slice(0, 12).toLowerCase() || "tab"
}

function buildHistoryFile(filename: string, slug: string | null): string {
  return slug ? `${slug}/${filename}` : filename
}

function parseHistoryFile(historyFile: string): { slug: string | null; filename: string } {
  const parts = historyFile.split("/")
  if (parts.length > 1) {
    return { slug: parts[0], filename: parts.slice(1).join("/") }
  }
  return { slug: null, filename: historyFile }
}

function untitledFilename(chatId: string): string {
  return `untitled-${shortIdFromChatId(chatId)}.json`
}

function buildLLMMessage(text: string, attachments: Attachment[]): { userMsg: string; hiddenContent: string; imgBase64: string | null } {
  let userMsg = text
  let imgBase64: string | null = null

  const textParts: string[] = []
  const imgAtts: Attachment[] = []

  for (const a of attachments) {
    if (a.mime.startsWith("image/")) {
      imgAtts.push(a)
    } else {
      const content = a.parsedMd ?? a.content
      if (content) textParts.push(`### ${a.name}\n\n${content}`)
    }
  }

  if (imgAtts.length > 0) {
    imgBase64 = "pending"
  }

  const hiddenContent = textParts.length > 0
    ? "**Attached files:**\n\n" + textParts.join("\n\n") + "\n\n## END"
    : ""

  return { userMsg, hiddenContent, imgBase64 }
}

function defaultSendParams(settings: ReturnType<typeof useSettings>, overrides: Record<string, unknown> = {}) {
  return {
    model: settings.selectedModel,
    system_prompt: settings.selectedPrompt ?? "",
    temperature: settings.temperature,
    reasoning_effort: settings.reasoningEffort === "none" ? null : settings.reasoningEffort,
    downscale_images: settings.downscaleImages,
    img_base64: null,
    ...overrides,
  }
}

function TabContent({ tab, isActive, onModeLabel, onHistoryFileChanged, onTitleLoaded, onOpenChat, sidebarCollapsed, onSidebarToggle, projectsOpen, onProjectsOpenChange, historiesOpen, onHistoriesOpenChange, activeProject }: {
  tab: Tab
  isActive: boolean
  onModeLabel: (label: string) => void
  onHistoryFileChanged: (tabId: string, historyFile: string) => void
  onTitleLoaded: (tabId: string, title: string | null) => void
  onOpenChat: (historyFile: string) => void
  sidebarCollapsed: boolean
  onSidebarToggle: () => void
  projectsOpen: boolean
  onProjectsOpenChange: (open: boolean) => void
  historiesOpen: boolean
  onHistoriesOpenChange: (open: boolean) => void
  activeProject: string | null
}) {
  const {
    messages,
    isStreaming,
    send,
    cancel,
    research,
    morphicSearch,
    reset,
    models,
    prompts,
    rootFiles,
    histories,
    historiesLoading,
    refreshHistories,
    deleteMessagePair,
    addMessagePair,
    setMessages,
    totalUsage,
    setTotalUsage,
  } = useChat()

  const {
    researchEnabled,
    morphicSearchEnabled,
    ocrEnabled,
    studyEnabled,
    toggleResearch,
    toggleMorphicSearch,
    toggleOCR,
    toggleStudy,
  } = useModeState()

  const settings = useSettings()
  const hasUsage = totalUsage.total_tokens > 0 ? totalUsage : undefined

  const [ocrImage, setOCRImage] = useState<string | null>(null)
  const [chatId, setChatId] = useState(() => tab.chatId)
  const [saveModalOpen, setSaveModalOpen] = useState(false)
  const loadedRef = useRef(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const [measuredWidth, setMeasuredWidth] = useState(0)

  useLayoutEffect(() => {
    if (!isActive) return
    const el = containerRef.current
    if (!el) return
    setMeasuredWidth(el.clientWidth)
  }, [isActive])

  const chatMaxWidth = measuredWidth > 0 ? Math.max(0, measuredWidth - MAX_SIDEBAR_WIDTH) : undefined

  useEffect(() => {
    if (!tab.historyFile || loadedRef.current) return
    loadedRef.current = true
    const { slug, filename } = parseHistoryFile(tab.historyFile)
    const loader = slug && activeProject
      ? loadProjectTab(slug, filename)
      : apiLoadChatHistory(tab.historyFile)
    loader.then((data) => {
      if (data.messages.length > 0) setMessages(data.messages)
      if (data.chat_id) setChatId(data.chat_id)
      if (data.title) onTitleLoaded(tab.id, data.title)
      if (data.usage) setTotalUsage(data.usage)
    }).catch(() => { })
  }, [tab.historyFile, activeProject, setMessages, onTitleLoaded])

  const autoSaveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  useEffect(() => {
    if (messages.length === 0) return

    // Only untitled conversations in projects are allowed to persist/autosave on disk.
    // Non-project untitled chats live strictly in memory and are never written to disk.
    if (!activeProject) return

    // Only autosave to untitled draft files — never overwrite a titled history
    // automatically. Titled files are modified exclusively via Alt+S (SaveChatModal).
    if (tab.historyFile) {
      const { filename } = parseHistoryFile(tab.historyFile)
      if (!filename.startsWith("untitled-")) return
    }

    if (autoSaveTimerRef.current) clearTimeout(autoSaveTimerRef.current)
    autoSaveTimerRef.current = setTimeout(() => {
      const filename = tab.historyFile
        ? parseHistoryFile(tab.historyFile).filename
        : untitledFilename(chatId)
      const historyFile = buildHistoryFile(filename, activeProject)
      if (historyFile !== tab.historyFile) {
        onHistoryFileChanged(tab.id, historyFile)
      }
      const title = tab.title ?? undefined
      saveProjectTab(activeProject, filename, messages, chatId, tab.name ?? undefined, title, hasUsage).catch(() => { })
    }, 2000)
    return () => { if (autoSaveTimerRef.current) clearTimeout(autoSaveTimerRef.current) }
  }, [messages, activeProject, tab.historyFile, tab.name, tab.title, chatId, tab.id, totalUsage, onHistoryFileChanged])

  const handleHistoryLoad = useCallback((filename: string) => {
    onOpenChat(filename)
    // Loading is handled by the target tab's init effect — do NOT call loadHistory
    // here; that would load the clicked history into the current tab's messages,
    // causing the autosave to overwrite it.
  }, [onOpenChat])

  useEffect(() => {
    if (researchEnabled) onModeLabel("Deep Research")
    else if (morphicSearchEnabled) onModeLabel("Search")
    else if (ocrEnabled) onModeLabel("LaTeX OCR")
    else if (studyEnabled) onModeLabel("PDF Study")
    else onModeLabel("Chat")
  }, [researchEnabled, morphicSearchEnabled, ocrEnabled, studyEnabled, onModeLabel])

  const openSaveModal = useCallback(() => {
    setSaveModalOpen(true)
  }, [])

  const handleSaveSubmit = useCallback(async (name: string) => {
    const newFilename = name + ".json"
    const oldHistoryFile = tab.historyFile
    const oldUntitled = oldHistoryFile
      ? parseHistoryFile(oldHistoryFile).filename
      : null
    const wasUntitled = oldUntitled?.startsWith("untitled-") ?? false

    if (activeProject) {
      const targetFilename = newFilename
      try {
        await saveProjectTab(activeProject, targetFilename, messages, chatId, tab.name ?? undefined, name, hasUsage)
        if (wasUntitled && oldUntitled) await deleteProjectTab(activeProject, oldUntitled)
      } catch { }
      onHistoryFileChanged(tab.id, buildHistoryFile(targetFilename, activeProject))
    } else {
      let resultPath = newFilename
      try {
        if (wasUntitled && oldHistoryFile) {
          const result = await renameChatHistory(oldHistoryFile, name)
          resultPath = result.new_path
          apiSaveChatHistory(result.filename, messages, chatId, name, hasUsage).catch(() => { })
        } else {
          await apiSaveChatHistory(newFilename, messages, chatId, name, hasUsage)
        }
      } catch { }
      onHistoryFileChanged(tab.id, resultPath)
    }
    refreshHistories()
    setSaveModalOpen(false)
  }, [messages, chatId, tab.id, tab.name, tab.historyFile, tab.title, totalUsage, activeProject, refreshHistories, onHistoryFileChanged])

  useEffect(() => {
    function handler(e: KeyboardEvent) {
      if (e.altKey && e.key === "s") { e.preventDefault(); openSaveModal() }
      if (e.altKey && e.key === "r") { e.preventDefault(); reset() }
    }
    document.addEventListener("keydown", handler)
    return () => document.removeEventListener("keydown", handler)
  }, [openSaveModal, reset])

  const handleSend = useCallback(
    async (text: string, attachments: Attachment[]) => {
      if (morphicSearchEnabled || researchEnabled) {
        if (attachments.length > 0) return
      }

      if (studyEnabled && attachments.some(a => a.mime === "application/pdf")) {
        await handleStudyPdf(text, attachments, chatId, settings.selectedModel, setMessages)
        toggleStudy()
        return
      }

      if (attachments.length > 0) {
        setMessages(prev => [
          ...prev,
          { role: "user" as const, content: text, attachments },
          { role: "assistant" as const, content: "" },
        ])

        const pdfNames = attachments.filter(a => a.mime === "application/pdf").map(a => a.name)
        let enriched = attachments

        if (pdfNames.length > 0) {
          try {
            const parsed = await parseFiles(chatId, pdfNames, activeProject)
            enriched = attachments.map(a => {
              if (a.mime !== "application/pdf") return a
              const p = parsed.find(r => r.name === a.name)
              return { ...a, parsedMd: p?.parsedMd ?? undefined }
            })
            if (!text.trim()) {
              updateLastAssistant(setMessages, m => ({ ...m, content: "Documents parsed - how can I help you?" }))
              return
            }
          } catch { }
        }

        const { hiddenContent } = buildLLMMessage(text, enriched)

        setMessages(prev => {
          const copy = [...prev]
          const idx = copy.length - 2
          if (idx >= 0 && copy[idx].role === "user") copy[idx] = { ...copy[idx], attachments: enriched, hiddenContent }
          return copy
        })

        const imgAtt = enriched.find(a => a.mime.startsWith("image/"))
        let imgBase64: string | null = null
        if (imgAtt) {
          try {
            const resp = await fetch(imgAtt.url)
            const blob = await resp.blob()
            imgBase64 = await new Promise<string>(resolve => { const r = new FileReader(); r.onload = () => resolve(r.result as string); r.readAsDataURL(blob) })
          } catch { }
        }

        const llmMsg = hiddenContent ? (text ? `${hiddenContent}\n\n${text}` : hiddenContent) : text
        send({ ...defaultSendParams(settings), user_msg: llmMsg, img_base64: imgBase64, downscale_images: settings.downscaleImages }, true)
        return
      }

      if (morphicSearchEnabled) {
        morphicSearch({ query: text, searchDepth: settings.searchDepth, model: settings.selectedModel || undefined })
      } else if (researchEnabled) {
        research({
          query: text, fastModel: settings.researchFastModel, smartModel: settings.researchSmartModel, strategicModel: settings.researchStrategicModel,
          depth: settings.researchDepth, breadth: settings.researchBreadth, reasoningEffort: settings.researchReasoning, reportType: settings.researchReportType,
        })
      } else {
        send({ ...defaultSendParams(settings), user_msg: text })
      }
    },
    [morphicSearchEnabled, researchEnabled, studyEnabled, chatId, activeProject, settings, send, research, morphicSearch, setMessages, toggleStudy],
  )

  const handleRemoveAttachment = useCallback((messageIndex: number, attachmentName: string) => {
    setMessages(prev => {
      const copy = [...prev]
      if (copy[messageIndex] && copy[messageIndex].attachments) {
        const att = copy[messageIndex].attachments!.find(a => a.name === attachmentName)
        if (att) deleteAttachment(chatId, attachmentName, activeProject).catch(() => { })
        copy[messageIndex] = { ...copy[messageIndex], attachments: copy[messageIndex].attachments!.filter(a => a.name !== attachmentName) }
      }
      return copy
    })
  }, [setMessages, chatId, activeProject])

  const handleAttachmentContentChange = useCallback((messageIndex: number, attachmentName: string, newContent: string) => {
    setMessages(prev => {
      const copy = [...prev]
      const msg = copy[messageIndex]
      if (msg?.attachments) {
        copy[messageIndex] = {
          ...msg,
          attachments: msg.attachments.map(a => a.name === attachmentName ? { ...a, content: newContent } : a),
        }
      }
      return copy
    })
  }, [setMessages])

  return (
    <div ref={containerRef} className="flex h-full overflow-hidden">
      <Sidebar
        collapsed={sidebarCollapsed}
        onToggle={onSidebarToggle}
        rootFiles={rootFiles}
        histories={histories}
        historiesLoading={historiesLoading}
        onHistoryLoad={handleHistoryLoad}
        onHistoryRefresh={refreshHistories}
        onSave={openSaveModal}
        onReset={reset}
        projectsOpen={projectsOpen}
        onProjectsOpenChange={onProjectsOpenChange}
        historiesOpen={historiesOpen}
        onHistoriesOpenChange={onHistoriesOpenChange}
      />
      <main className="flex-1 min-w-0 flex flex-col relative bg-zinc-950">
        <header className="h-[60px] shrink-0 flex items-center px-4 gap-4 bg-zinc-950 z-40 border-b border-zinc-800/50">
          {researchEnabled ? (
            <ResearchModelsBar
              models={models}
              fastModel={settings.researchFastModel}
              smartModel={settings.researchSmartModel}
              strategicModel={settings.researchStrategicModel}
              onFastModelChange={settings.setResearchFastModel}
              onSmartModelChange={settings.setResearchSmartModel}
              onStrategicModelChange={settings.setResearchStrategicModel}
            />
          ) : (
            <div className="flex items-center gap-4">
              <ModelSelector models={models} selectedModel={settings.selectedModel} onSelect={settings.setSelectedModel} />
              <div className="w-px h-4 bg-zinc-800" />
              <ReasoningSelector reasoningEffort={settings.reasoningEffort} onReasoningChange={settings.setReasoningEffort} />
            </div>
          )}

          <div className="flex-1" />

          <div className="flex items-center gap-2">
            <TokenCounter usage={totalUsage} />
            <ThemeToggle />
            <MoreOptionsMenu prompts={prompts} />
          </div>
        </header>
        <div className="flex-1 overflow-hidden relative">
          <ChatContainer
            chatId={chatId}
            messages={messages}
            isStreaming={isStreaming}
            onSend={handleSend}
            onCancel={cancel}
            onDeletePair={deleteMessagePair}
            onOCRRequest={setOCRImage}
            onRemoveAttachment={handleRemoveAttachment}
            onAttachmentContentChange={handleAttachmentContentChange}
            slug={activeProject}
            chatMaxWidth={chatMaxWidth}
          />
          {ocrEnabled && ocrImage && (
            <OCRPanel
              image={ocrImage}
              model={settings.ocrModel || DEFAULT_VISION_MODEL}
              onComplete={(output) => { addMessagePair("OCR Request", output); setOCRImage(null) }}
              onClose={() => { setOCRImage(null); toggleOCR() }}
            />
          )}
        </div>
      </main>
      <SaveChatModal open={saveModalOpen} onClose={() => setSaveModalOpen(false)} onSave={handleSaveSubmit} />
    </div>
  )
}

function AppContent() {
  const tabManagerRef = useRef<TabManagerHandle>(null)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(true)
  const [projectsOpen, setProjectsOpen] = useState(true)
  const [historiesOpen, setHistoriesOpen] = useState(false)
  const { activeProject, projectData, dashboardOpen, syncTabs, setDashboardOpen, closeProject } = useProject()
  const pendingHistoryRef = useRef<string | null>(null)

  const handleHistoryFileChanged = useCallback((tabId: string, historyFile: string) => {
    tabManagerRef.current?.updateTabHistoryFile(tabId, historyFile)
  }, [])

  const handleTitleLoaded = useCallback((tabId: string, title: string | null) => {
    tabManagerRef.current?.updateTabTitle(tabId, title)
  }, [])

  const handleOpenChat = useCallback((historyFile: string) => {
    const tabs = tabManagerRef.current?.getTabs() ?? []
    const existing = tabs.find((t) => t.historyFile === historyFile)
    if (existing) {
      tabManagerRef.current?.switchToTab(existing.id)
      return
    }
    // Context switch: clicking a non-project history while a project is active
    // closes the project first, then opens the history in a fresh tab
    if (activeProject && !historyFile.includes("/")) {
      pendingHistoryRef.current = historyFile
      closeProject()
      return
    }
    const filename = historyFile.includes("/")
      ? historyFile.split("/").pop()!
      : historyFile
    const label = filename.replace(".json", "")
    tabManagerRef.current?.addTabWithName(label, historyFile)
  }, [activeProject, closeProject])

  const handleTabClose = useCallback((tab: Tab) => {
    if (tab.historyFile && !tab.title) {
      const { slug, filename } = parseHistoryFile(tab.historyFile)
      if (slug && activeProject) {
        deleteProjectTab(slug, filename).catch(() => { })
      } else {
        apiDeleteChatHistory(tab.historyFile).catch(() => { })
        deleteChatUploads(tab.chatId, null).catch(() => { })
      }
    } else if (tab.historyFile && tab.title) {
      deleteChatUploads(tab.chatId, activeProject).catch(() => { })
    }
  }, [activeProject])

  const handleTabsChange = useCallback((tabs: Tab[]) => {
    if (!activeProject) return
    syncTabs(
      tabs.map((t) => ({ id: t.id, name: t.name, historyFile: t.historyFile, title: t.title, chatId: t.chatId })),
      (tabId, historyFile) => {
        tabManagerRef.current?.updateTabHistoryFile(tabId, historyFile)
      },
    ).catch(() => { })
  }, [activeProject, syncTabs])

  const loadedProjectRef = useRef<string | null>(null)

  useEffect(() => {
    if (activeProject === loadedProjectRef.current) return
    loadedProjectRef.current = activeProject

    if (!activeProject || !projectData) {
      const pending = pendingHistoryRef.current
      if (pending) {
        pendingHistoryRef.current = null
        const label = pending.replace(".json", "")
        tabManagerRef.current?.initTabs([nextTab(label, pending)])
      } else {
        tabManagerRef.current?.initTabs([nextTab()])
      }
    } else {
      const newTabs: Tab[] = projectData.tabs.map((t) =>
        nextTab(t.name, `${activeProject}/${t.filename}`, t.title)
      )
      if (newTabs.length === 0) newTabs.push(nextTab())
      tabManagerRef.current?.initTabs(newTabs)
    }
  }, [activeProject, projectData])

  useEffect(() => {
    function handler(e: KeyboardEvent) {
      if (e.altKey && e.key === "d") { e.preventDefault(); setDashboardOpen(!dashboardOpen) }
    }
    document.addEventListener("keydown", handler)
    return () => document.removeEventListener("keydown", handler)
  }, [setDashboardOpen, dashboardOpen])

  return (
    <ModeProvider>
      <TabManager
        ref={tabManagerRef}
        onCloseTab={handleTabClose}
        onTabsChange={handleTabsChange}
        renderContent={(tab, onModeLabel, isActive) => (
          <TabContent
            tab={tab}
            isActive={isActive}
            onModeLabel={onModeLabel}
            onHistoryFileChanged={handleHistoryFileChanged}
            onTitleLoaded={handleTitleLoaded}
            onOpenChat={handleOpenChat}
            sidebarCollapsed={sidebarCollapsed}
            onSidebarToggle={() => setSidebarCollapsed((c) => !c)}
            projectsOpen={projectsOpen}
            onProjectsOpenChange={setProjectsOpen}
            historiesOpen={historiesOpen}
            onHistoriesOpenChange={setHistoriesOpen}
            activeProject={activeProject}
          />
        )}
      />
      {dashboardOpen && <ProjectDashboard />}
    </ModeProvider>
  )
}

export default function Home() {
  return (
    <SettingsProvider>
      <ProjectProvider>
        <AppContent />
      </ProjectProvider>
    </SettingsProvider>
  )
}
