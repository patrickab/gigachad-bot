"use client"

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react"
import { createPortal } from "react-dom"
import { Sidebar } from "@/components/Sidebar"
import { CommandMenu, type CommandMenuItem } from "@/components/CommandMenu"
import { ChatContainer } from "@/components/ChatContainer"
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
import { CreateDocumentPanel } from "@/components/CreateDocumentPanel"
import { useCommandBar } from "@/hooks/useCommandBar"
import { useMemoryCategories } from "@/hooks/useMemoryCategories"
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
import { SidebarProvider } from "@/contexts/SidebarContext"
import { MemoryViewerProvider } from "@/contexts/MemoryViewerContext"
import { MemoryViewer } from "@/components/MemoryViewer"
import { handleStudyPdf, updateLastMsg as updateLastAssistant } from "@/hooks/useStudyHandler"
import {
  saveChatHistory as apiSaveChatHistory,
  loadChatHistory as apiLoadChatHistory,
  parseFiles,
  deleteAttachment,
  saveProjectTab,
  loadProjectTab,
  parseHistoryFile,
  buildHistoryFile,
  listObsidianFiles,
  attachObsidianFile,
  listProjectDocuments,
  listAllDocuments,
  uploadDocument,
  addDocument,
  attachDocument,
  removeDocument,
  writeDocument,
  uploadFile,
  loadFileViewerText,
  generateMindmap,
} from "@/lib/api"
import { renderCanvasToJpeg } from "@/lib/drawing"
import { DEFAULT_VISION_MODEL } from "@/lib/config"
import type { Attachment, Message, ObsidianFile, ProjectDocument, Usage } from "@/lib/types"
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
  { command: "/obsidian-load" },
]

function TabContent({ tab, isActive, onModeLabel, onHistoryFileChanged, onTitleLoaded, onOpenChat, activeProject, focusQaIndex, focusKey, onConfigChange }: {
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

  const commandBar = useCommandBar()
  const { globalCategories, projectCategories } = useMemoryCategories(activeProject)
  const [obsidianEnabled, setObsidianEnabled] = useState(false)
  const [obsidianFiles, setObsidianFiles] = useState<ObsidianFile[]>([])
  const [obsidianOpen, setObsidianOpen] = useState(false)
  const [documentOpen, setDocumentOpen] = useState(false)
  const [createDocOpen, setCreateDocOpen] = useState(false)
  const [projectDocuments, setProjectDocuments] = useState<ProjectDocument[]>([])
  const [allDocuments, setAllDocuments] = useState<ProjectDocument[]>([])

  useEffect(() => {
    let cancelled = false
    listObsidianFiles()
      .then((r) => { if (!cancelled) { setObsidianEnabled(r.enabled); setObsidianFiles(r.files) } })
      .catch(() => {})
    return () => { cancelled = true }
  }, [])

  const openObsidian = useCallback(() => {
    setObsidianOpen(true)
    // Refresh the listing on open so vault changes are reflected.
    listObsidianFiles()
      .then((r) => { setObsidianEnabled(r.enabled); setObsidianFiles(r.files) })
      .catch(() => {})
  }, [])
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

  const appendAttachment = useCallback((att: Attachment) => {
    setMessages((prev) => {
      const copy = [...prev]
      let idx = -1
      for (let i = copy.length - 1; i >= 0; i--) {
        if (copy[i].role === "user") { idx = i; break }
      }
      if (idx === -1) {
        // Empty chat: seed a complete pair so display stays strictly alternating.
        const attachments = [att]
        return [
          { role: "user" as const, content: "", attachments, hiddenContent: buildHiddenContent(attachments) || undefined },
          { role: "assistant" as const, content: "" },
        ]
      }
      const msg = copy[idx]
      const attachments = [...(msg.attachments ?? []).filter((a) => a.name !== att.name), att]
      copy[idx] = { ...msg, attachments, hiddenContent: buildHiddenContent(attachments) || undefined }
      return copy
    })
  }, [setMessages])

  const attachToChat = useCallback(
    async (attach: (chatId: string, path: string, slug: string | null) => Promise<Attachment>, path: string) => {
      try { appendAttachment(await attach(chatId, path, activeProject)) } catch { }
    },
    [chatId, activeProject, appendAttachment],
  )

  const handleObsidianSelect = useCallback((path: string) => {
    setObsidianOpen(false)
    attachToChat(attachObsidianFile, path)
  }, [attachToChat])

  const handleDocumentSelect = useCallback(async (path: string) => {
    setDocumentOpen(false)
    if (path.endsWith(".canvas")) {
      try {
        const live = liveCanvasRef.current
        const text = live?.path === path ? live.content : await loadFileViewerText(path)
        const doc = text.trim() ? JSON.parse(text) : null
        if (!doc?.strokes?.length) return
        const blob = await renderCanvasToJpeg(doc.strokes)
        const name = path.split("/").pop()!.replace(/\.canvas$/, ".jpg")
        const file = new File([blob], name, { type: "image/jpeg" })
        const att = await uploadFile(chatId, file, activeProject, true)
        chatInputRef.current?.addAttachment(att)
      } catch { /* */ }
    } else {
      attachToChat(attachDocument, path)
    }
  }, [attachToChat, chatId, activeProject])

  const loadProjectDocs = useCallback(() => {
    if (activeProject) listProjectDocuments(activeProject).then(setProjectDocuments).catch(() => {})
    else setProjectDocuments([])
  }, [activeProject])

  const refreshDocuments = useCallback(() => {
    loadProjectDocs()
    listAllDocuments().then(setAllDocuments).catch(() => {})
  }, [loadProjectDocs])

  useEffect(() => { loadProjectDocs() }, [loadProjectDocs])

  const handleCreateDocument = useCallback(async (name: string) => {
    if (!activeProject) return
    try {
      await writeDocument(activeProject, name)
      setCreateDocOpen(false)
      refreshDocuments()
    } catch { /* */ }
  }, [activeProject, refreshDocuments])

  const handleDeleteDocument = useCallback(async (path: string) => {
    if (!activeProject) return
    try {
      await removeDocument(activeProject, path)
      refreshDocuments()
    } catch { /* */ }
  }, [activeProject, refreshDocuments])

  const handleDocumentSaved = useCallback((filename?: string, content?: string) => {
    refreshDocuments()
    if (filename && content !== undefined) {
      setMessages(prev => prev.map(msg => {
        if (!msg.attachments?.some(a => a.name === filename)) return msg
        const attachments = msg.attachments!.map(a => a.name === filename ? { ...a, content } : a)
        return { ...msg, attachments, hiddenContent: buildHiddenContent(attachments) || undefined }
      }))
    }
  }, [refreshDocuments, setMessages])

  const openDocuments = useCallback(() => {
    refreshDocuments()
    setDocumentOpen(true)
  }, [refreshDocuments])

  const handleDocumentUpload = useCallback(async (files: File[]) => {
    if (!activeProject) return
    // Sequential: each PDF spawns a MinerU server, so avoid parallel contention.
    for (const file of files) {
      try {
        await uploadDocument(activeProject, file)
      } catch { }
    }
    refreshDocuments()
  }, [activeProject, refreshDocuments])

  const handleAddDocToProject = useCallback(async (path: string) => {
    if (!activeProject) return
    try {
      await addDocument(activeProject, path)
      refreshDocuments()
    } catch { }
  }, [activeProject, refreshDocuments])

  useEffect(() => {
    if (!isActive) return
    const onKey = (e: KeyboardEvent) => {
      if (e.altKey && (e.key === "x" || e.key === "X")) {
        e.preventDefault()
        setDocumentOpen((open) => {
          if (!open) refreshDocuments()
          return !open
        })
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [isActive, refreshDocuments])

  const [branchMessageIdx, setBranchMessageIdx] = useState<number | null>(null)
  const [saveModalOpen, setSaveModalOpen] = useState(false)
  const [mindmapModalOpen, setMindmapModalOpen] = useState(false)
  const [promptEditorOpen, setPromptEditorOpen] = useState(false)
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

  const handleMindmapSubmit = useCallback(async (prompt: string) => {
    if (messages.length === 0) return
    const userMsg = prompt ? `Provide a mindmap. ${prompt}` : "Provide a mindmap."
    setMessages(prev => [
      ...prev,
      { role: "user" as const, content: userMsg },
      { role: "assistant" as const, content: "Generating mind map…" },
    ])
    setMindmapModalOpen(false)
    try {
      const mindmap = await generateMindmap(
        messages.map((m) => ({ role: m.role, content: m.content })),
        config.selectedModel,
        prompt,
      )
      updateLastAssistant(setMessages, m => ({ ...m, content: mindmap }))
    } catch {
      updateLastAssistant(setMessages, m => ({ ...m, content: "Mind map generation failed." }))
    }
  }, [messages, config.selectedModel, setMessages])

  useEffect(() => {
    if (!isActive) return

    function handler(e: KeyboardEvent) {
      if (e.altKey && e.key === "s") { e.preventDefault(); isTitled ? handleQuickSave() : openSaveModal() }
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
        if (obsidianEnabled) openObsidian()
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
  }, [isActive, isTitled, handleQuickSave, openSaveModal, reset, commandBar.open, commandBar.submitCommand, commandBar.state.phase, messages, activeProject, chatId, branchMessageIdx, obsidianEnabled, openObsidian])

  const runCommand = useCallback(async (command: string) => {
    if (command === "/obsidian-load") {
      commandBar.close()
      openObsidian()
      return
    }
    if (command === "/mindmap") {
      commandBar.close()
      if (messages.length === 0) return
      setMindmapModalOpen(true)
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
  }, [commandBar.submitCommand, commandBar.close, messages, activeProject, chatId, branchMessageIdx, openObsidian, setMindmapModalOpen])

  const commandItems = COMMANDS.filter((cmd) =>
    (cmd.command !== "/obsidian-load" || obsidianEnabled) &&
    (cmd.command !== "/memorize-project" || activeProject != null)
  )

  const handleSend = useCallback(
    async (text: string, attachments: Attachment[]) => {
      const trimmed = text.trim()
      if (trimmed === "/obsidian-load") {
        openObsidian()
        return
      }
      if (trimmed === "/mindmap" || trimmed.startsWith("/mindmap ")) {
        if (messages.length === 0) return
        const userPrompt = trimmed === "/mindmap" ? "" : trimmed.slice("/mindmap ".length).trim()
        if (userPrompt) {
          await handleMindmapSubmit(userPrompt)
        } else {
          setMindmapModalOpen(true)
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
          }, parseFiles)
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
    [searchEnabled, researchEnabled, studyEnabled, chatId, branchMessageIdx, activeProject, config, send, research, webSearch, setMessages, toggleStudy, commandBar.submitCommand, messages, openObsidian, handleMindmapSubmit, setMindmapModalOpen],
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
        if (att) deleteAttachment(chatId, attachmentName, activeProject).catch(() => { })
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
      <Sidebar
        onOpenChat={onOpenChat}
        onRefreshAll={refreshAll}
        onSave={openSaveModal}
        onReset={reset}
        onMerge={(childFile) => doMergeBranch(childFile)}
        onCascadeDelete={handleCascadeDelete}
      />
      <main className="flex-1 min-w-0 flex flex-col relative bg-paper">
        <header className="h-[60px] shrink-0 flex items-center px-4 gap-4 z-40 border-b border-divider/50 bg-paper/80 backdrop-blur-xl">
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
            <MoreOptionsMenu prompts={prompts} config={config} onConfigChange={onConfigChange} onEditPrompts={() => setPromptEditorOpen(true)} />
          </div>
        </header>
        <div className="flex-1 overflow-hidden relative transition-opacity duration-200">
          <ChatContainer
            chatId={chatId}
            messages={messages}
            isStreaming={isStreaming}
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
            onRemoveAttachment={handleRemoveAttachment}
            onToggleAttachmentActive={handleToggleAttachmentActive}
            onAttachmentContentChange={handleAttachmentContentChange}
            obsidianEnabled={obsidianEnabled}
            onOpenObsidian={openObsidian}
            documents={projectDocuments}
            onSelectDocument={handleDocumentSelect}
            onOpenDocuments={activeProject ? openDocuments : undefined}
            onCreateDocument={activeProject ? () => setCreateDocOpen(true) : undefined}
            onDeleteDocument={handleDeleteDocument}
            onDocumentSaved={handleDocumentSaved}
            chatInputRef={chatInputRef}
            liveCanvasRef={liveCanvasRef}
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
      <CreateDocumentPanel open={createDocOpen} onClose={() => setCreateDocOpen(false)} onCreate={handleCreateDocument} />
      <MindmapModal open={mindmapModalOpen} onClose={() => setMindmapModalOpen(false)} onGenerate={handleMindmapSubmit} />
      <PromptEditor open={promptEditorOpen} onClose={() => setPromptEditorOpen(false)} onPromptsChanged={setPrompts} />
      {isActive && obsidianOpen && obsidianEnabled && (
        <FileViewer
          files={obsidianFiles.map((f) => f.path)}
          onSelect={handleObsidianSelect}
          onClose={() => setObsidianOpen(false)}
          placeholder="Search vault…"
          emptyLabel="No notes"
        />
      )}
      {isActive && documentOpen && (
        <DocumentPicker
          projectSlug={activeProject}
          projectDocuments={projectDocuments}
          allDocuments={allDocuments}
          onAttach={handleDocumentSelect}
          onAddToProject={handleAddDocToProject}
          onUpload={handleDocumentUpload}
          onClose={() => setDocumentOpen(false)}
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
