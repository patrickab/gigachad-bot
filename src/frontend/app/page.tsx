"use client"

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react"
import { Sidebar } from "@/components/Sidebar"
import { ChatContainer } from "@/components/ChatContainer"
import { MAX_SIDEBAR_WIDTH } from "@/components/ChatSidebar"
import { ModelDropdown } from "@/components/ModelDropdown"
import { MoreOptionsMenu } from "@/components/MoreOptionsMenu"
import { ReasoningSelector } from "@/components/ReasoningSelector"
import { ResearchModelsBar } from "@/components/ResearchModelsBar"
import { ThemeToggle } from "@/components/ThemeToggle"
import { OCRPanel } from "@/components/OCRPanel"
import { SaveChatModal } from "@/components/SaveChatModal"
import { TabManager, nextTab, settingsToTabConfig, type TabConfig } from "@/components/TabManager"
import type { Tab, TabManagerHandle } from "@/components/TabManager"
import { ProjectDashboard } from "@/components/ProjectDashboard"
import { TokenCounter } from "@/components/TokenCounter"
import { useChat } from "@/hooks/useChat"
import { useModeState, ModeProvider } from "@/hooks/useModeState"
import { useSettings, SettingsProvider } from "@/contexts/SettingsContext"
import { useProject, ProjectProvider } from "@/contexts/ProjectContext"
import { useBranches } from "@/hooks/useBranches"
import { BranchProvider } from "@/contexts/BranchContext"
import { SidebarProvider } from "@/contexts/SidebarContext"
import { handleStudyPdf, updateLastMsg as updateLastAssistant } from "@/hooks/useStudyHandler"
import {
  saveChatHistory as apiSaveChatHistory,
  loadChatHistory as apiLoadChatHistory,
  parseFiles,
  deleteAttachment,
  saveProjectTab,
  deleteProjectTab,
  loadProjectTab,
  parseHistoryFile,
  buildHistoryFile,
} from "@/lib/api"
import { DEFAULT_VISION_MODEL } from "@/lib/config"
import type { Attachment, Message, Usage } from "@/lib/types"



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

function defaultSendParams(config: TabConfig, prompts: Record<string, string>, overrides: Record<string, unknown> = {}) {
  return {
    model: config.selectedModel,
    system_prompt: config.selectedPrompt ? (prompts[config.selectedPrompt] ?? "") : "",
    temperature: config.temperature,
    reasoning_effort: config.reasoningEffort === "none" ? null : config.reasoningEffort,
    downscale_images: config.downscaleImages,
    img_base64: null,
    ...overrides,
  }
}

function TabContent({ tab, isActive, onModeLabel, onHistoryFileChanged, onTitleLoaded, onOpenChat, activeProject, focusQaIndex, focusKey, onConfigChange }: {
  tab: Tab
  isActive: boolean
  onModeLabel: (label: string) => void
  onHistoryFileChanged: (tabId: string, historyFile: string) => void
  onTitleLoaded: (tabId: string, title: string | null) => void
  onOpenChat: (historyFile: string, qaIndex?: number) => void
  activeProject: string | null
  focusQaIndex: number | null
  focusKey: number
  onConfigChange: (config: Partial<TabConfig>) => void
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
    loadHistory,
    deleteMessagePair,
    addMessagePair,
    setMessages,
    totalUsage,
    setTotalUsage,
  } = useChat()

  const {
    refreshAll,
    createBranch: doCreateBranch,
    mergeBranch: doMergeBranch,
    cascadeDelete: doCascadeDelete,
    orphanChildren: doOrphanChildren,
    setActiveFile,
    setActiveQaIndex,
  } = useBranches()

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
  const config = tab.config
  const hasUsage = totalUsage.total_tokens > 0 ? totalUsage : undefined

  const [ocrImage, setOCRImage] = useState<string | null>(null)
  const [chatId, setChatId] = useState(() => tab.chatId)
  const [branchMessageIdx, setBranchMessageIdx] = useState<number | null>(null)
  const [saveModalOpen, setSaveModalOpen] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const loadIdRef = useRef(0)
  const [measuredWidth, setMeasuredWidth] = useState(0)

  useLayoutEffect(() => {
    if (!isActive) return
    const el = containerRef.current
    if (!el) return
    setMeasuredWidth(el.clientWidth)
  }, [isActive])

  useEffect(() => {
    if (isActive && tab.historyFile) {
      setActiveFile(tab.historyFile)
    }
  }, [isActive, tab.historyFile, setActiveFile])

  useEffect(() => {
    if (isActive) {
      setActiveQaIndex(focusQaIndex)
    }
  }, [isActive, focusQaIndex, setActiveQaIndex])

  const chatMaxWidth = measuredWidth > 0 ? Math.max(0, measuredWidth - MAX_SIDEBAR_WIDTH) : undefined

  useEffect(() => {
    if (!tab.historyFile) return
    const loadId = ++loadIdRef.current
    setMessages([])
    setTotalUsage({ total_tokens: 0, prompt_tokens: 0, completion_tokens: 0 })
    setBranchMessageIdx(null)
    const { slug, filename } = parseHistoryFile(tab.historyFile)
    const loader = slug && activeProject
      ? loadProjectTab(slug, filename)
      : apiLoadChatHistory(tab.historyFile)
    loader.then((data) => {
      if (loadIdRef.current !== loadId) return
      if (data.messages.length > 0) setMessages(data.messages)
      if (data.chat_id) setChatId(data.chat_id)
      if (data.title) onTitleLoaded(tab.id, data.title)
      if (data.usage) setTotalUsage(data.usage)
      setBranchMessageIdx(data.branch_message_idx ?? null)
    }).catch(() => { })
  }, [tab.historyFile, activeProject, setMessages, onTitleLoaded, setTotalUsage])

  useEffect(() => {
    if (researchEnabled) onModeLabel("Deep Research")
    else if (morphicSearchEnabled) onModeLabel("Search")
    else if (ocrEnabled) onModeLabel("LaTeX OCR")
    else if (studyEnabled) onModeLabel("PDF Study")
    else onModeLabel("Chat")
  }, [researchEnabled, morphicSearchEnabled, ocrEnabled, studyEnabled, onModeLabel])

  const isTitled = !!tab.historyFile

  const openSaveModal = useCallback(() => {
    setSaveModalOpen(true)
  }, [])

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

  useEffect(() => {
    function handler(e: KeyboardEvent) {
      if (e.altKey && e.key === "s") { e.preventDefault(); isTitled ? handleQuickSave() : openSaveModal() }
      if (e.altKey && e.key === "r") { e.preventDefault(); reset() }
    }
    document.addEventListener("keydown", handler)
    return () => document.removeEventListener("keydown", handler)
  }, [isTitled, handleQuickSave, openSaveModal, reset])

  const handleSend = useCallback(
    async (text: string, attachments: Attachment[]) => {
      if (morphicSearchEnabled || researchEnabled) {
        if (attachments.length > 0) return
      }

      if (studyEnabled && attachments.some(a => a.mime === "application/pdf")) {
        await handleStudyPdf(text, attachments, chatId, config.selectedModel, setMessages, activeProject)
        toggleStudy()
        return
      }

      if (attachments.length > 0) {
        setMessages(prev => [
          ...prev,
          { role: "user" as const, content: text, attachments },
          { role: "assistant" as const, content: "" },
        ])

        const needsParsing = attachments.filter(a => !a.mime.startsWith("image/") && !a.content && !a.parsedMd)
        let enriched = attachments

        if (needsParsing.length > 0) {
          try {
            const parsed = await parseFiles(chatId, needsParsing.map(a => a.name), activeProject)
            enriched = attachments.map(a => {
              if (a.content || a.parsedMd || a.mime.startsWith("image/")) return a
              const p = parsed.find(r => r.name === a.name)
              return { ...a, parsedMd: p?.parsedMd ?? undefined }
            })
          } catch {
            updateLastAssistant(setMessages, m => ({ ...m, content: "Failed to parse attached documents. You can try again or re-upload the file." }))
            return
          }
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

        const fallbackPrompt = text.trim() ? text : "Please review the attached document and provide a summary."
        const llmMsg = hiddenContent ? `${hiddenContent}\n\n${fallbackPrompt}` : fallbackPrompt
        send({ ...defaultSendParams(config, prompts), user_msg: llmMsg, img_base64: imgBase64, downscale_images: config.downscaleImages }, true)
        return
      }

      if (morphicSearchEnabled) {
        morphicSearch({ query: text, searchDepth: settings.searchDepth, model: config.selectedModel || undefined })
      } else if (researchEnabled) {
        research({
          query: text, fastModel: config.researchFastModel, smartModel: config.researchSmartModel, strategicModel: config.researchStrategicModel,
          depth: config.researchDepth, breadth: config.researchBreadth, reasoningEffort: config.researchReasoning, reportType: config.researchReportType,
        })
      } else {
        send({ ...defaultSendParams(config, prompts), user_msg: text })
      }
    },
    [morphicSearchEnabled, researchEnabled, studyEnabled, chatId, activeProject, config, send, research, morphicSearch, setMessages, toggleStudy],
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

  const handleCascadeDelete = useCallback(async (filename: string) => {
    await doCascadeDelete(filename)
    const { slug, filename: fn } = parseHistoryFile(filename)
    if (slug) {
      await deleteProjectTab(slug, fn).catch(() => {})
    }
    await refreshAll()
  }, [doCascadeDelete, refreshAll])

  const handleBranch = useCallback((qaIndex: number) => {
    if (!tab.historyFile) return
    doCreateBranch(tab.historyFile, qaIndex).then(({ childFile }) => {
      onOpenChat(childFile)
    })
  }, [tab.historyFile, doCreateBranch, onOpenChat])

  return (
    <div ref={containerRef} className="flex h-full overflow-hidden">
      <Sidebar
        onOpenChat={onOpenChat}
        onRefreshAll={refreshAll}
        onSave={openSaveModal}
        onReset={reset}
        onMerge={(childFile) => doMergeBranch(childFile)}
        onCascadeDelete={handleCascadeDelete}
      />
      <main className="flex-1 min-w-0 flex flex-col relative bg-paper">
        <header className="h-[60px] shrink-0 flex items-center px-4 gap-4 bg-paper z-40 border-b border-divider/50">
          {researchEnabled ? (
            <ResearchModelsBar
              models={models}
              fastModel={config.researchFastModel}
              smartModel={config.researchSmartModel}
              strategicModel={config.researchStrategicModel}
              onFastModelChange={(m) => onConfigChange({ researchFastModel: m })}
              onSmartModelChange={(m) => onConfigChange({ researchSmartModel: m })}
              onStrategicModelChange={(m) => onConfigChange({ researchStrategicModel: m })}
            />
          ) : (
            <div className="flex items-center gap-4">
              <ModelDropdown models={models} selectedModel={config.selectedModel} onSelect={(m) => onConfigChange({ selectedModel: m })} />
              <div className="w-px h-4 bg-surface-elevated" />
              <ReasoningSelector reasoningEffort={config.reasoningEffort} onReasoningChange={(v) => onConfigChange({ reasoningEffort: v })} />
            </div>
          )}

          <div className="flex-1" />

          <div className="flex items-center gap-2">
            <TokenCounter usage={totalUsage} />
            <ThemeToggle />
            <MoreOptionsMenu prompts={prompts} config={config} onConfigChange={onConfigChange} searchDepth={settings.searchDepth} onSearchDepthChange={settings.setSearchDepth} />
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
            onBranch={tab.historyFile ? handleBranch : undefined}
            focusQaIndex={focusQaIndex}
            focusKey={focusKey}
            isActive={isActive}
            branchMessageIdx={branchMessageIdx}
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
  const [focusQaIndex, setFocusQaIndex] = useState<number | null>(null)
  const [focusKey, setFocusKey] = useState(0)
  const { activeProject, projectData, dashboardOpen, syncTabs, setDashboardOpen, closeProject } = useProject()
  const settings = useSettings()
  const pendingHistoryRef = useRef<string | null>(null)

  const handleHistoryFileChanged = useCallback((tabId: string, historyFile: string) => {
    tabManagerRef.current?.updateTabHistoryFile(tabId, historyFile)
  }, [])

  const handleTitleLoaded = useCallback((tabId: string, title: string | null) => {
    tabManagerRef.current?.updateTabTitle(tabId, title)
  }, [])

  const handleOpenChat = useCallback((historyFile: string, qaIndex?: number) => {
    if (qaIndex != null) {
      setFocusQaIndex(qaIndex)
      setFocusKey((k) => k + 1)
    } else {
      setFocusQaIndex(null)
    }
    const tabs = tabManagerRef.current?.getTabs() ?? []
    const existing = tabs.find((t) => t.historyFile === historyFile)
    if (existing) {
      tabManagerRef.current?.switchToTab(existing.id)
      return
    }
    if (activeProject) {
      const belongsToActiveProject = historyFile.startsWith(`${activeProject}/`)
      if (!belongsToActiveProject) {
        pendingHistoryRef.current = historyFile
        closeProject()
        return
      }
    }
    const activeTabId = tabManagerRef.current?.getActiveTabId()
    if (activeTabId) {
      tabManagerRef.current?.updateTabHistoryFile(activeTabId, historyFile)
    }
  }, [activeProject, closeProject])

  const handleTabClose = useCallback((_tab: Tab) => {
  }, [])

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
        tabManagerRef.current?.initTabs([nextTab(label, pending, null, settingsToTabConfig(settings))])
      } else {
        tabManagerRef.current?.initTabs([nextTab(null, null, null, settingsToTabConfig(settings))])
      }
    } else {
      const newTabs: Tab[] = projectData.tabs.map((t) =>
        nextTab(t.name, `${activeProject}/${t.filename}`, t.title, settingsToTabConfig(settings))
      )
      if (newTabs.length === 0) newTabs.push(nextTab(null, null, null, settingsToTabConfig(settings)))
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
    <BranchProvider>
      <SidebarProvider>
        <TabManager
        ref={tabManagerRef}
        onCloseTab={handleTabClose}
        onTabsChange={handleTabsChange}
        defaultConfig={settingsToTabConfig(settings)}
        renderContent={(tab, onModeLabel, isActive, onConfigChange) => (
          <ModeProvider>
            <TabContent
              tab={tab}
              isActive={isActive}
              onModeLabel={onModeLabel}
              onHistoryFileChanged={handleHistoryFileChanged}
              onTitleLoaded={handleTitleLoaded}
              onOpenChat={handleOpenChat}
              activeProject={activeProject}
              focusQaIndex={focusQaIndex}
              focusKey={focusKey}
              onConfigChange={onConfigChange}
            />
          </ModeProvider>
        )}
      />
        {dashboardOpen && <ProjectDashboard />}
      </SidebarProvider>
    </BranchProvider>
  )
}

export default function Home() {
  return (
    <SettingsProvider>
      <ProjectProvider>
        <div id="main-content" className="h-dvh">
          <AppContent />
        </div>
      </ProjectProvider>
    </SettingsProvider>
  )
}
