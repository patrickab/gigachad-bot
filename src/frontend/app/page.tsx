"use client"

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react"
import { createPortal } from "react-dom"
import dynamic from "next/dynamic"
import { Maximize2, Minimize2 } from "lucide-react"
import { Sidebar } from "@/components/Sidebar"
import { CommandMenu, type CommandMenuItem } from "@/components/CommandMenu"
import { ChatContainer, ChatSidebarProvider, type ChatSidebarContextValue } from "@/components/ChatContainer"
import type { ChatInputHandle } from "@/components/ChatInput"
import { MAX_SIDEBAR_WIDTH } from "@/components/ChatSidebar"
import { ModelDropdown } from "@/components/ModelDropdown"
import { MoreOptionsMenu } from "@/components/MoreOptionsMenu"
import { PromptEditor } from "@/components/PromptEditor"
import { ReasoningSelector } from "@/components/ReasoningSelector"
import { ResearchModelsBar } from "@/components/ResearchModelsBar"
import { ThemeToggle } from "@/components/ThemeToggle"
import { OCRPanel } from "@/components/OCRPanel"
import { SaveChatModal } from "@/components/SaveChatModal"
import { MindmapModal } from "@/components/MindmapModal"
import { MemoryPanel } from "@/components/MemoryPanel"
import { FileViewer } from "@/components/FileViewer"
import { DocumentPicker } from "@/components/DocumentPicker"
import { DocumentEditor } from "@/components/DocumentEditor"
import { CreateDocumentPanel } from "@/components/CreateDocumentPanel"
import { CanvasWorkspace, type CanvasSelection } from "@/components/CanvasWorkspace"
import { useCommandBar } from "@/hooks/useCommandBar"
import { useMemoryCategories } from "@/hooks/useMemoryCategories"
import { useVaultPicker } from "@/hooks/useVaultPicker"
import { useProjectDocuments } from "@/hooks/useProjectDocuments"
import { useChatModals } from "@/hooks/useChatModals"
import { TabManager, nextTab, settingsToTabConfig, type TabConfig } from "@/components/TabManager"
import type { Tab, TabManagerHandle } from "@/components/TabManager"
import { ProjectDashboard } from "@/components/ProjectDashboard"
import { TokenCounter } from "@/components/TokenCounter"
import { useChat } from "@/hooks/useChat"
import { useModeState, ModeProvider } from "@/hooks/useModeState"
import { useSettings, SettingsProvider } from "@/contexts/SettingsContext"
import { useProject, ProjectProvider } from "@/contexts/ProjectContext"
import { useBranches } from "@/contexts/BranchContext"
import { BranchProvider } from "@/contexts/BranchContext"
import { SidebarProvider, type AppSurface } from "@/contexts/SidebarContext"
import { MemoryViewerProvider } from "@/contexts/MemoryViewerContext"
import { MemoryViewer } from "@/components/MemoryViewer"
import { handleStudyPdf, updateLastMsg as updateLastAssistant } from "@/hooks/useStudyHandler"
import {
  loadChatHistory as apiLoadChatHistory,
  parseFiles,
  deleteAttachment,
  loadProjectTab,
  parseHistoryFile,
  attachFileVaultFile,
  writeFileVaultFile,
  fileViewerRawUrl,
} from "@/lib/api"

const PdfViewer = dynamic(() => import("@/components/PdfViewer").then((m) => ({ default: m.PdfViewer })), { ssr: false })
import type { Attachment, Message } from "@/lib/types"
import {
  buildAttachedSend,
  buildHiddenContent,
  collectActiveImagePaths,
  normalizeMessageAttachments,
} from "@/lib/attachments"

function defaultSendParams(config: TabConfig, prompts: Record<string, string>, overrides: Record<string, unknown> = {}) {
  return {
    model: config.selectedModel,
    system_prompt: config.selectedPrompt ? (prompts[config.selectedPrompt] ?? "") : "",
    temperature: config.temperature,
    reasoning_effort: config.reasoningEffort === "none" ? null : config.reasoningEffort,
    downscale_images: config.downscaleImages,
    img_paths: [] as string[],
    ...overrides,
  }
}

const COMMANDS: CommandMenuItem[] = [
  { command: "/mindmap", keywords: ["mind", "map", "overview", "visualize"] },
  { command: "/memorize", keywords: ["memory", "both"] },
  { command: "/memorize-global", keywords: ["memory", "global", "profile"] },
  { command: "/memorize-project", keywords: ["memory", "project"] },
  { command: "/vault-load" },
]

function TabContent({ tab, isActive, onModeLabel, onHistoryFileChanged, onTitleLoaded, onOpenChat, activeProject, focusQaIndex, focusKey, onConfigChange, onAppModeChange }: {
  tab: Tab
  isActive: boolean
  onModeLabel: (label: string, loading?: boolean) => void
  onHistoryFileChanged: (tabId: string, historyFile: string) => void
  onTitleLoaded: (tabId: string, title: string | null) => void
  onOpenChat: (historyFile: string, qaIndex?: number) => void
  activeProject: string | null
  focusQaIndex: number | null
  focusKey: number
  onConfigChange: (config: Partial<TabConfig>) => void
  onAppModeChange: (mode: AppSurface) => void
}) {
  const {
    messages,
    isStreaming,
    send,
    regenerateAt,
    cancel,
    research,
    webSearch,
    reset,
    models,
    prompts,
    setPrompts,
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
    searchEnabled,
    ocrEnabled,
    studyEnabled,
    toggleResearch,
    toggleSearch,
    toggleOCR,
    toggleStudy,
  } = useModeState()

  const settings = useSettings()
  const config = tab.config
  const hasUsage = totalUsage.total_tokens > 0 ? totalUsage : undefined
  const appMode = tab.appMode
  const setTabAppMode = onAppModeChange
  const [canvasSel, setCanvasSel] = useState<CanvasSelection | null>(null)
  const [canvasToolbarSlot, setCanvasToolbarSlot] = useState<HTMLElement | null>(null)
  const [canvasFullscreen, setCanvasFullscreen] = useState(false)

  // Leaving canvas mode (or switching tabs) drops fullscreen so the sidebar
  // doesn't stay hidden when the user returns to chat.
  useEffect(() => {
    if (appMode !== "canvas") setCanvasFullscreen(false)
  }, [appMode, isActive])

  useEffect(() => {
    if (!canvasFullscreen) return
    const esc = (e: KeyboardEvent) => {
      if (e.key === "Escape") { e.preventDefault(); setCanvasFullscreen(false) }
    }
    document.addEventListener("keydown", esc)
    return () => document.removeEventListener("keydown", esc)
  }, [canvasFullscreen])

  const commandBar = useCommandBar()
  const { globalCategories, projectCategories } = useMemoryCategories(activeProject)
  const commandMemoryCount = commandBar.state.phase === "review" || commandBar.state.phase === "composing"
    ? commandBar.state.globalMemories.length + (commandBar.state.projectMemories?.length ?? 0)
    : 0
  const memoryActions = commandBar.bindProject(activeProject)
  const [commandMenuSlot, setCommandMenuSlot] = useState<HTMLElement | null>(null)
  const [tabLabelSlot, setTabLabelSlot] = useState<HTMLElement | null>(null)
  const [commandQuery, setCommandQuery] = useState("")
  const commandInputRef = useRef<HTMLInputElement>(null)
  const chatInputRef = useRef<ChatInputHandle>(null)
  const liveCanvasRef = useRef<{ path: string; content: string } | null>(null)
  const [ocrImage, setOCRImage] = useState<string | null>(null)
  const [chatId, setChatId] = useState(() => tab.chatId)
  const [extracting, setExtracting] = useState(0)

  const vault = useVaultPicker({
    isActive,
    hasMessages: messages.length > 0,
    chatInputRef,
    setExtracting,
  })

  const docs = useProjectDocuments({
    isActive,
    appMode,
    activeProject,
    chatId,
    chatInputRef,
    liveCanvasRef,
    setExtracting,
    setMessages,
  })

  // branchMessageIdx is read/written from the memory keydown shortcut, the
  // command bar, handleSend's memorize dispatch, and the history-load effect
  // below — it stays here rather than in one of the domain hooks.
  const [branchMessageIdx, setBranchMessageIdx] = useState<number | null>(null)

  const modals = useChatModals({
    tab,
    activeProject,
    messages,
    chatId,
    hasUsage,
    selectedModel: config.selectedModel,
    refreshAll,
    onHistoryFileChanged,
    setMessages,
  })

  const containerRef = useRef<HTMLDivElement>(null)
  const loadIdRef = useRef(0)
  const [measuredWidth, setMeasuredWidth] = useState(0)

  useLayoutEffect(() => {
    if (!isActive) return
    const el = containerRef.current
    if (!el) return
    setMeasuredWidth(el.clientWidth)
  }, [isActive])

  useLayoutEffect(() => {
    setCommandMenuSlot(document.getElementById(`tab-command-menu-${tab.id}`))
    setTabLabelSlot(document.getElementById(`tab-label-${tab.id}`))
  }, [tab.id])

  useEffect(() => {
    if (isActive) {
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
      if (data.messages.length > 0) {
        setMessages(data.messages.map(normalizeMessageAttachments))
      }
      if (data.chat_id) setChatId(data.chat_id)
      if (data.title) onTitleLoaded(tab.id, data.title)
      if (data.usage) setTotalUsage(data.usage)
      setBranchMessageIdx(data.branch_message_idx ?? null)
    }).catch(() => { })
  }, [tab.historyFile, activeProject, setMessages, onTitleLoaded, setTotalUsage])

  useEffect(() => {
    if (commandBar.state.phase === "input") {
      setCommandQuery("")
      setTimeout(() => commandInputRef.current?.focus(), 0)
    }
  }, [commandBar.state.phase])

  const docReviewLoading = commandBar.state.phase === "doc-review" && commandBar.state.loadingScopes.length > 0
  useEffect(() => {
    if (commandBar.state.phase === "input") onModeLabel("\0")
    else if (commandBar.state.phase === "extracting") onModeLabel("Memory Management", true)
    else if (commandBar.state.phase === "review") onModeLabel(`${commandMemoryCount} ${commandMemoryCount === 1 ? "memory" : "memories"} proposed`)
    else if (commandBar.state.phase === "composing") onModeLabel("Updating memory docs", true)
    else if (commandBar.state.phase === "doc-review") onModeLabel("Review memory docs", docReviewLoading)
    else if (commandBar.state.phase === "error") onModeLabel("Memory error")
    else if (researchEnabled) onModeLabel("Deep Research")
    else if (searchEnabled) onModeLabel("Search")
    else if (ocrEnabled) onModeLabel("LaTeX OCR")
    else if (studyEnabled) onModeLabel("PDF Study")
    else onModeLabel("Chat")
  }, [commandBar.state.phase, docReviewLoading, commandMemoryCount, researchEnabled, searchEnabled, ocrEnabled, studyEnabled, onModeLabel])

  useEffect(() => {
    if (!isActive || appMode === "canvas") return

    function handler(e: KeyboardEvent) {
      if (e.altKey && e.key === "s") { e.preventDefault(); modals.isTitled ? modals.handleQuickSave() : modals.setSaveModalOpen(true) }
      if (e.altKey && e.key === "r") { e.preventDefault(); reset() }
      if (e.altKey && e.key.toLowerCase() === "m") {
        e.preventDefault()
        // Memory is extracted from the main branch only — branches don't influence memories.
        if (commandBar.state.phase === "idle" && branchMessageIdx == null) {
          commandBar.submitCommand(
            messages.map((m) => ({ role: m.role, content: m.content })),
            activeProject,
            chatId,
            "/memorize",
          )
        }
        return
      }
      if (e.altKey && e.key.toLowerCase() === "o") {
        e.preventDefault()
        if (vault.vaultEnabled) vault.openVaultPicker()
        return
      }

      const isSlash = e.key === "/" && !e.altKey && !e.ctrlKey && !e.metaKey
      const isCtrlK = (e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k"

      if (isSlash || isCtrlK) {
        const tag = (e.target as HTMLElement).tagName
        if (isSlash && (tag === "TEXTAREA" || tag === "INPUT" || (e.target as HTMLElement).isContentEditable)) return
        e.preventDefault()
        e.stopPropagation()
        e.stopImmediatePropagation()
        if (commandBar.state.phase === "idle") { commandBar.open() }
      }
    }
    window.addEventListener("keydown", handler, true)
    return () => {
      window.removeEventListener("keydown", handler, true)
    }
  }, [isActive, appMode, modals.isTitled, modals.handleQuickSave, modals.setSaveModalOpen, reset, commandBar.open, commandBar.submitCommand, commandBar.state.phase, messages, activeProject, chatId, branchMessageIdx, vault.vaultEnabled, vault.openVaultPicker])

  const runCommand = useCallback(async (command: string) => {
    if (command === "/vault-load") {
      commandBar.close()
      vault.openVaultPicker()
      return
    }
    if (command === "/mindmap") {
      commandBar.close()
      if (messages.length === 0) return
      modals.setMindmapModalOpen(true)
      return
    }
    // Memory is extracted from the main branch only — branches don't influence memories.
    if (command.startsWith("/memorize") && branchMessageIdx != null) {
      commandBar.close()
      return
    }
    commandBar.submitCommand(
      messages.map((m) => ({ role: m.role, content: m.content })),
      activeProject,
      chatId,
      command,
    )
  }, [commandBar.submitCommand, commandBar.close, messages, activeProject, chatId, branchMessageIdx, vault.openVaultPicker, modals.setMindmapModalOpen])

  const commandItems = COMMANDS.filter((cmd) =>
    (cmd.command !== "/vault-load" || vault.vaultEnabled) &&
    (cmd.command !== "/memorize-project" || activeProject != null)
  )

  const handleSend = useCallback(
    async (text: string, attachments: Attachment[]) => {
      const trimmed = text.trim()
      if (trimmed === "/vault-load") {
        vault.openVaultPicker()
        return
      }
      if (trimmed === "/mindmap" || trimmed.startsWith("/mindmap ")) {
        if (messages.length === 0) return
        const userPrompt = trimmed === "/mindmap" ? "" : trimmed.slice("/mindmap ".length).trim()
        if (userPrompt) {
          await modals.handleMindmapSubmit(userPrompt, attachments)
        } else {
          modals.setMindmapAttachments(attachments)
          modals.setMindmapModalOpen(true)
        }
        return
      }
      if (trimmed === "/memorize" || trimmed.startsWith("/memorize ") || trimmed.startsWith("/memorize-")) {
        // Memory is extracted from the main branch only — branches don't influence memories.
        if (branchMessageIdx == null) {
          await commandBar.submitCommand(
            messages.map((m) => ({ role: m.role, content: m.content })),
            activeProject,
            chatId,
            trimmed,
          )
        }
        return
      }

      if (searchEnabled || researchEnabled) {
        if (attachments.length > 0) return
      }

      if (studyEnabled && attachments.some(a => a.mime === "application/pdf")) {
        await handleStudyPdf(text, attachments, chatId, config.selectedModel, setMessages, activeProject)
        toggleStudy()
        return
      }

      if (attachments.length > 0) {
        const activeAttachments = attachments.map((a) => ({ ...a, active: true }))
        setMessages(prev => [
          ...prev,
          { role: "user" as const, content: text, attachments: activeAttachments },
          { role: "assistant" as const, content: "" },
        ])

        let result
        try {
          result = await buildAttachedSend({
            text,
            attachments: activeAttachments,
            priorMessages: messages,
            chatId,
            slug: activeProject,
            downscaleImages: config.downscaleImages,
            baseParams: defaultSendParams(config, prompts),
          }, parseFiles, attachFileVaultFile)
        } catch {
          updateLastAssistant(setMessages, m => ({ ...m, content: "Failed to parse attached documents. You can try again or re-upload the file." }))
          return
        }

        const { enriched, hiddenContent, request } = result
        setMessages(prev => {
          const copy = [...prev]
          const idx = copy.length - 2
          if (idx >= 0 && copy[idx].role === "user") copy[idx] = { ...copy[idx], attachments: enriched, hiddenContent }
          return copy
        })

        send(request, true)
        return
      }

      if (searchEnabled) {
        webSearch({
          query: text,
          focusMode: config.searchFocusMode,
          optimizationMode: config.searchOptimization,
          systemInstructions: config.searchSystemInstructions,
          domain: config.searchDomain,
          images: config.searchImages,
          videos: config.searchVideos,
          model: config.selectedModel || undefined,
        })
      } else if (researchEnabled) {
        research({
          query: text, fastModel: config.researchFastModel, smartModel: config.researchSmartModel, strategicModel: config.researchStrategicModel,
          depth: config.researchDepth, breadth: config.researchBreadth, reasoningEffort: config.researchReasoning, reportType: config.researchReportType,
        })
      } else {
        send({
          ...defaultSendParams(config, prompts),
          chat_id: chatId,
          user_msg: text,
          img_paths: collectActiveImagePaths(messages),
          project_slug: activeProject,
        })
      }
    },
    [searchEnabled, researchEnabled, studyEnabled, chatId, branchMessageIdx, activeProject, config, send, research, webSearch, setMessages, toggleStudy, commandBar.submitCommand, messages, vault.openVaultPicker, modals.handleMindmapSubmit, modals.setMindmapAttachments, modals.setMindmapModalOpen],
  )

  const handleRegenerate = useCallback(
    async (globalIndex: number) => {
      if (isStreaming) return
      const userMsg = messages[globalIndex]
      if (!userMsg || userMsg.role !== "user") return
      const assistantMsg = messages[globalIndex + 1]
      if (assistantMsg?.search_result || assistantMsg?.research_steps?.length) return

      const attachments = userMsg.attachments ?? []
      if (attachments.length > 0) {
        const hidden = userMsg.hiddenContent ?? buildHiddenContent(attachments)
        const fallbackPrompt = userMsg.content.trim() ? userMsg.content : "Please review the attached document and provide a summary."
        const llmMsg = hidden ? `${hidden}\n\n${fallbackPrompt}` : fallbackPrompt
        await regenerateAt(globalIndex, {
          ...defaultSendParams(config, prompts),
          chat_id: chatId,
          user_msg: llmMsg,
          img_paths: collectActiveImagePaths(messages, { userIndex: globalIndex }),
          downscale_images: config.downscaleImages,
          project_slug: activeProject,
        })
        return
      }

      await regenerateAt(globalIndex, {
        ...defaultSendParams(config, prompts),
        chat_id: chatId,
        user_msg: userMsg.content,
        img_paths: collectActiveImagePaths(messages, { userIndex: globalIndex }),
        project_slug: activeProject,
      })
    },
    [isStreaming, messages, regenerateAt, config, prompts, activeProject, chatId],
  )

  const handleToggleAttachmentActive = useCallback((messageIndex: number, attachmentName: string) => {
    setMessages(prev => {
      const copy = [...prev]
      const msg = copy[messageIndex]
      if (!msg?.attachments) return prev
      const attachments = msg.attachments.map(a =>
        a.name === attachmentName ? { ...a, active: !a.active } : a
      )
      const hiddenContent = buildHiddenContent(attachments) || undefined
      copy[messageIndex] = { ...msg, attachments, hiddenContent }
      return copy
    })
  }, [setMessages])

  const handleRemoveAttachment = useCallback((messageIndex: number, attachmentName: string) => {
    setMessages(prev => {
      const copy = [...prev]
      const msg = copy[messageIndex]
      if (msg?.attachments) {
        const att = msg.attachments.find(a => a.name === attachmentName)
        // Vault refs have no upload copy — and the source file must never be deleted.
        if (att && !att.vaultPath) deleteAttachment(chatId, attachmentName, activeProject).catch(() => { })
        const remaining = msg.attachments.filter(a => a.name !== attachmentName)
        const hiddenContent = buildHiddenContent(remaining) || undefined
        copy[messageIndex] = { ...msg, attachments: remaining, hiddenContent }
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

  const sidebarContext = useMemo<ChatSidebarContextValue>(() => ({
    onRemoveAttachment: handleRemoveAttachment,
    onToggleAttachmentActive: handleToggleAttachmentActive,
    onAttachmentContentChange: handleAttachmentContentChange,
    vaultEnabled: vault.vaultEnabled,
    onOpenVault: vault.openVaultPicker,
    documents: docs.mergedDocuments,
    onSelectDocument: docs.handleDocumentSelect,
    onOpenDocuments: activeProject ? docs.openDocuments : undefined,
    onCreateDocument: activeProject ? () => docs.setCreateDocOpen(true) : undefined,
    onDeleteDocument: docs.handleDeleteDocument,
    onDocumentSaved: docs.handleDocumentSaved,
    liveCanvasRef,
    vaultPaths: docs.vaultDocPaths,
    vaultEditingPath: vault.vaultEditPath,
    onEditVaultDocument: vault.setVaultEditPath,
  }), [
    handleRemoveAttachment, handleToggleAttachmentActive, handleAttachmentContentChange,
    vault.vaultEnabled, vault.openVaultPicker, vault.vaultEditPath, vault.setVaultEditPath,
    docs.mergedDocuments, docs.handleDocumentSelect, docs.openDocuments, docs.setCreateDocOpen,
    docs.handleDeleteDocument, docs.handleDocumentSaved, docs.vaultDocPaths,
    activeProject,
  ])

  const handleCascadeDelete = useCallback(async (filename: string) => {
    await doCascadeDelete(filename)
    await refreshAll()
  }, [doCascadeDelete, refreshAll])

  const handleBranch = useCallback((qaIndex: number) => {
    if (!tab.historyFile) return
    doCreateBranch(tab.historyFile, qaIndex).then(({ childFile }) => {
      onOpenChat(childFile)
    })
  }, [tab.historyFile, doCreateBranch, onOpenChat])

  return (
    <div ref={containerRef} className="relative flex h-full overflow-hidden">
      {isActive && commandBar.state.phase === "input" && tabLabelSlot && createPortal(
        <input
          ref={commandInputRef}
          autoFocus
          value={commandQuery}
          onChange={(e) => setCommandQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Escape") { e.preventDefault(); commandBar.close() }
          }}
          onClick={(e) => e.stopPropagation()}
          spellCheck={false}
          className="flex-1 min-w-0 bg-transparent font-mono text-xs text-ink outline-none"
        />,
        tabLabelSlot,
      )}
      <CommandMenu
        open={isActive && commandBar.state.phase === "input"}
        items={commandItems}
        query={commandQuery}
        onRun={runCommand}
        onClose={commandBar.close}
        container={commandMenuSlot}
      />
      {!canvasFullscreen && (
        <Sidebar
          onOpenChat={onOpenChat}
          onRefreshAll={refreshAll}
          onSave={() => modals.setSaveModalOpen(true)}
          onReset={reset}
          onMerge={(childFile) => doMergeBranch(childFile)}
          onCascadeDelete={handleCascadeDelete}
          onVaultSelect={vault.handleVaultSelect}
          onVaultsChanged={vault.refreshVaultList}
          activeCanvasPath={canvasSel?.path ?? null}
          onCanvasSelect={(path, scope) => setCanvasSel({ path, scope })}
          onCanvasDeleted={(path) => setCanvasSel((s) => (s?.path === path ? null : s))}
          appMode={appMode}
          onAppModeChange={setTabAppMode}
        />
      )}
      <main className="flex-1 min-w-0 flex flex-col relative bg-paper">
        <header className="h-[60px] shrink-0 flex items-center px-4 gap-4 z-40 border-b border-divider/50 bg-paper/80 backdrop-blur-xl">
          {appMode === "canvas" ? (
            <div ref={setCanvasToolbarSlot} className="flex-1 min-w-0 flex items-center" />
          ) : researchEnabled ? (
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

          {appMode !== "canvas" && <div className="flex-1" />}

          <div className="flex items-center gap-2">
            {appMode === "canvas" && (
              <button
                onClick={() => setCanvasFullscreen((f) => !f)}
                className="rounded p-1 text-ink-subtle hover:text-ink hover:bg-hover transition-colors"
              >
                {canvasFullscreen ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
              </button>
            )}
            {appMode !== "canvas" && <TokenCounter usage={totalUsage} />}
            <ThemeToggle />
            <MoreOptionsMenu prompts={prompts} config={config} onConfigChange={onConfigChange} onEditPrompts={() => modals.setPromptEditorOpen(true)} />
          </div>
        </header>
        <div className="flex-1 overflow-hidden relative transition-opacity duration-200">
          {appMode === "canvas" ? (
            <CanvasWorkspace
              selected={canvasSel}
              slug={activeProject}
              toolbarSlot={canvasToolbarSlot}
              onCloseEditor={() => setCanvasSel(null)}
              onCreated={setCanvasSel}
              onModeLabel={onModeLabel}
            />
          ) : (
          <ChatSidebarProvider value={sidebarContext}>
            <ChatContainer
              chatId={chatId}
              messages={messages}
              isStreaming={isStreaming}
              extracting={extracting > 0}
              onSend={handleSend}
              onCancel={cancel}
              onDeletePair={deleteMessagePair}
              onRegenerate={handleRegenerate}
              onBranch={tab.historyFile ? handleBranch : undefined}
              focusQaIndex={focusQaIndex}
              focusKey={focusKey}
              isActive={isActive}
              branchMessageIdx={branchMessageIdx}
              onOCRRequest={setOCRImage}
              chatInputRef={chatInputRef}
              slug={activeProject}
              chatMaxWidth={chatMaxWidth}
            />
          </ChatSidebarProvider>
          )}
          {ocrEnabled && ocrImage && (
            <OCRPanel
              image={ocrImage}
              model={settings.ocrModel}
              onComplete={(output) => { addMessagePair("OCR Request", output); setOCRImage(null) }}
              onClose={() => { setOCRImage(null); toggleOCR() }}
            />
          )}
          {isActive && vault.vaultEditPath && (
            <DocumentEditor
              path={vault.vaultEditPath}
              slug=""
              overlay
              persistOverride={(content) => writeFileVaultFile(vault.vaultEditPath!, content)}
              onClose={() => vault.setVaultEditPath(null)}
              onModeLabel={onModeLabel}
              onNavigate={vault.setVaultEditPath}
              model={config.selectedModel}
            />
          )}
          {isActive && vault.vaultPdfPath && (
            <div className="absolute inset-0 z-50 flex flex-col bg-paper">
              <div className="flex h-9 shrink-0 items-center border-b border-divider/50 px-4">
                <span className="truncate text-xs font-medium text-ink">{vault.vaultPdfPath.split("/").pop()}</span>
              </div>
              <div className="min-h-0 flex-1">
                <PdfViewer url={fileViewerRawUrl(vault.vaultPdfPath)} />
              </div>
            </div>
          )}
        </div>
      </main>
      <SaveChatModal open={modals.saveModalOpen} onClose={() => modals.setSaveModalOpen(false)} onSave={modals.handleSaveSubmit} />
      <CreateDocumentPanel open={docs.createDocOpen} onClose={() => docs.setCreateDocOpen(false)} onCreate={docs.handleCreateDocument} />
      <MindmapModal open={modals.mindmapModalOpen} onClose={() => modals.setMindmapModalOpen(false)} onGenerate={(prompt) => modals.handleMindmapSubmit(prompt, modals.mindmapAttachments)} />
      <PromptEditor open={modals.promptEditorOpen} onClose={() => modals.setPromptEditorOpen(false)} onPromptsChanged={setPrompts} />
      {isActive && vault.vaultPickerOpen && vault.vaultEnabled && (
        <FileViewer
          files={vault.vaultFiles.map((f) => f.path)}
          onSelect={vault.handleVaultSelect}
          onClose={() => vault.setVaultPickerOpen(false)}
          placeholder="Search vaults…"
          emptyLabel="No files"
        />
      )}
      {isActive && docs.documentOpen && (
        <DocumentPicker
          projectSlug={activeProject}
          projectDocuments={docs.projectDocuments}
          allDocuments={docs.allDocuments}
          onAttach={docs.handleDocumentSelect}
          onAddToProject={docs.handleAddDocToProject}
          onUpload={docs.handleDocumentUpload}
          onClose={() => docs.setDocumentOpen(false)}
        />
      )}
      {commandBar.state.phase === "input" && (
        <div
          className="absolute inset-0 z-[70]"
          onClick={commandBar.close}
        />
      )}
      {isActive && commandBar.state.phase !== "idle" && commandBar.state.phase !== "input" && (
        <MemoryPanel
          state={commandBar.state}
          projectEnabled={!!activeProject}
          projectSlug={activeProject}
          actions={memoryActions}
          globalCategories={globalCategories}
          projectCategories={projectCategories}
        />
      )}
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
    const activeTabId = tabManagerRef.current?.getActiveTabId()
    const activeTab = tabs.find((t) => t.id === activeTabId)

    // toggle: clicking the active element deactivates it (blank chat)
    if (activeTab && activeTab.historyFile === historyFile && qaIndex == null) {
      tabManagerRef.current?.initTabs([nextTab(null, null, null, settingsToTabConfig(settings))])
      return
    }

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
    if (activeTabId) {
      tabManagerRef.current?.updateTabHistoryFile(activeTabId, historyFile)
    }
  }, [activeProject, closeProject, settings])

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

    const pending = pendingHistoryRef.current
    if (pending) {
      // a chat was clicked while another project was active — open exactly it
      pendingHistoryRef.current = null
      tabManagerRef.current?.initTabs([nextTab(pending.replace(".json", ""), pending, null, settingsToTabConfig(settings))])
    } else {
      // entering or leaving a project lands on a fresh chat. Saved chats are
      // opened explicitly from the sidebar, never auto-restored — selecting a
      // project must not silently activate a conversation.
      tabManagerRef.current?.initTabs([nextTab(null, null, null, settingsToTabConfig(settings))])
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
        <MemoryViewerProvider>
        <TabManager
        ref={tabManagerRef}
        onTabsChange={handleTabsChange}
        defaultConfig={settingsToTabConfig(settings)}
        renderContent={(tab, onModeLabel, isActive, onConfigChange, onAppModeChange) => (
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
              onAppModeChange={onAppModeChange}
            />
          </ModeProvider>
        )}
      />
        {dashboardOpen && <ProjectDashboard />}
        <MemoryViewer />
        </MemoryViewerProvider>
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
