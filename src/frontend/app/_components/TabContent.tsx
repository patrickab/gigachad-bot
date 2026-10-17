"use client"

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react"
import { PanelLeft, PanelLeftClose } from "lucide-react"
import { Sidebar } from "@/components/Sidebar"
import { ChatContainer, type ChatFocus } from "@/components/ChatContainer"
import { ChatSidebarProvider, type ChatSidebarContextValue } from "@/components/ChatContextPanel"
import type { ChatInputHandle } from "@/components/ChatInput"
import { MAX_SIDEBAR_WIDTH } from "@/components/SidePanel"
import { ModelDropdown } from "@/components/ModelDropdown"
import { SettingsMenu } from "@/components/SettingsMenu"
import { PromptEditor } from "@/components/PromptEditor"
import { ReasoningSelector, effectiveReasoningEffort, useReasoningLevels } from "@/components/ReasoningSelector"
import { ThemeToggle } from "@/components/ThemeToggle"
import { SaveAsModal } from "@/components/SaveAsModal"
import { FileViewer } from "@/components/FileViewer"
import { DocumentPicker } from "@/components/DocumentPicker"
import { DocumentEditor } from "@/components/DocumentEditor"
import { CreateDocumentPanel } from "@/components/CreateDocumentPanel"
import { CanvasWorkspace } from "@/components/CanvasWorkspace"
import { ArchitectureWindow } from "@/components/ArchitectureWindow"
import { SidebarTabs } from "@/components/TabManager"
import { TokenCounter } from "@/components/TokenCounter"
import { TopBar, TopBarPill, TopBarTransparent } from "@/components/TopBar"
import { LazyPdfViewer } from "@/components/LazyPdfViewer"
import { useBranches } from "@/contexts/BranchContext"
import { useEnabledTools } from "@/contexts/EnabledToolsContext"
import { useProject } from "@/contexts/ProjectContext"
import { useSidebar } from "@/contexts/SidebarContext"
import { useTab } from "@/contexts/TabContext"
import { useCanvasView } from "@/hooks/useCanvasView"
import { useChat } from "@/hooks/useChat"
import { useChatPersistence } from "@/hooks/useChatPersistence"
import { useChatTurns } from "@/hooks/useChatTurns"
import { useFilePicker } from "@/hooks/useFilePicker"
import { useMemoryCommandBar } from "@/hooks/useMemoryCommandBar"
import { useProjectDocuments } from "@/hooks/useProjectDocuments"
import { useSlashCommands } from "@/hooks/useSlashCommands"
import { loadChatHistory, writeRootFile } from "@/lib/api"
import { normalizeMessageAttachments } from "@/lib/attachments"
import { parseHistoryFile } from "@/lib/historyFile"
import { fileViewerRawUrl } from "@/lib/urls"
import { MemoryCommandLayer } from "./MemoryCommandLayer"

/** One tab's workspace: the chat (or canvas) with its sidebar, top bar, modals and keyboard shortcuts. */
export function TabContent({ onOpenChat, focus }: {
  onOpenChat: (historyFile: string, qaIndex?: number) => void
  focus: ChatFocus
}) {
  const { tab, isActive, setModeLabel, updateConfig, setAppMode, setHistoryFile, setTitle } = useTab()
  const { activeProject } = useProject()
  const chat = useChat()
  const { messages, isStreaming, cancel, reset, models, saveModelProviders, saveModelTabOrder, saveModelDefaults, prompts, setPrompts, setMessages, totalUsage, setTotalUsage } = chat
  const { refreshAll, createBranch, setActiveFile, setActiveQaIndex } = useBranches()
  const { researchEnabled } = useEnabledTools()
  const { collapsed: sidebarCollapsed, toggleCollapsed: toggleSidebar } = useSidebar()

  const config = tab.config
  const appMode = tab.appMode
  const reasoningLevels = useReasoningLevels(config.selectedModel)
  const reasoningEffort = effectiveReasoningEffort(reasoningLevels, config.reasoningEffort)
  const usage = totalUsage.total_tokens > 0 ? totalUsage : undefined

  const [chatId, setChatId] = useState(() => tab.chatId)
  // Index where this chat branched off its parent, null for a main chat.
  const [branchMessageIdx, setBranchMessageIdx] = useState<number | null>(null)
  const [extracting, setExtracting] = useState(0)
  const [promptEditorOpen, setPromptEditorOpen] = useState(false)
  const [contextToggleSlot, setContextToggleSlot] = useState<HTMLElement | null>(null)
  const [wideToggleSlot, setWideToggleSlot] = useState<HTMLElement | null>(null)
  const chatInputRef = useRef<ChatInputHandle>(null)
  const liveCanvasRef = useRef<{ path: string; content: string } | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const loadIdRef = useRef(0)
  const [measuredWidth, setMeasuredWidth] = useState(0)

  const canvasView = useCanvasView()
  const filePicker = useFilePicker({ isActive, hasMessages: messages.length > 0, chatInputRef, setExtracting })
  const docs = useProjectDocuments({ isActive, appMode, activeProject, chatId, chatInputRef, liveCanvasRef, setExtracting, setMessages })
  const persistence = useChatPersistence({ tab, activeProject, messages, chatId, usage, refreshAll, setHistoryFile })
  const memory = useMemoryCommandBar({ messages, chatId, branchMessageIdx })
  const { menuItems, runSlashCommand } = useSlashCommands({
    filesEnabled: filePicker.filesEnabled,
    hasProject: activeProject != null,
    openFilePicker: filePicker.openFilePicker,
    memorize: memory.memorize,
  })
  const turns = useChatTurns({
    chat,
    chatId,
    activeProject,
    config,
    reasoningEffort,
    prompts,
    autosave: persistence.handleAutosave,
    refreshDocuments: docs.refreshDocuments,
    runSlashCommand,
  })

  // A new tab has a blank selectedModel (DEFAULT_TAB_CONFIG). Seed it from the
  // configured default chat model. Never hardcode a fallback name here.
  useEffect(() => {
    if (!config.selectedModel && models?.defaults.default_model) updateConfig({ selectedModel: models.defaults.default_model })
  }, [config.selectedModel, models, updateConfig])

  useLayoutEffect(() => {
    if (!isActive) return
    const el = containerRef.current
    if (!el) return
    setMeasuredWidth(el.clientWidth)
  }, [isActive])

  useEffect(() => {
    if (isActive) setActiveFile(tab.historyFile)
  }, [isActive, tab.historyFile, setActiveFile])

  useEffect(() => {
    if (isActive) setActiveQaIndex(focus.qaIndex)
  }, [isActive, focus.qaIndex, setActiveQaIndex])

  // Leave room for the side panel when the chat column is capped.
  const chatMaxWidth = measuredWidth > 0 ? Math.max(0, measuredWidth - MAX_SIDEBAR_WIDTH) : undefined

  useEffect(() => {
    if (!tab.historyFile) return
    const loadId = ++loadIdRef.current
    setMessages([])
    setTotalUsage({ total_tokens: 0, prompt_tokens: 0, completion_tokens: 0 })
    setBranchMessageIdx(null)
    // Project chats load by their own slug, so switching projects never reloads open tabs.
    const { slug } = parseHistoryFile(tab.historyFile)
    loadChatHistory(tab.historyFile).then((data) => {
      if (loadIdRef.current !== loadId) return
      if (data.messages.length > 0) {
        setMessages(data.messages.map((m) => normalizeMessageAttachments(m, data.chat_id, slug)))
      }
      if (data.chat_id) setChatId(data.chat_id)
      if (data.title) setTitle(data.title)
      if (data.usage) setTotalUsage(data.usage)
      setBranchMessageIdx(data.branch_message_idx ?? null)
    }).catch(() => { })
  }, [tab.historyFile, setMessages, setTitle, setTotalUsage])

  useEffect(() => {
    if (!isActive || appMode === "canvas") return

    function handler(e: KeyboardEvent) {
      if (e.altKey && e.key === "s") { e.preventDefault(); persistence.isTitled ? persistence.handleQuickSave() : persistence.setSaveModalOpen(true) }
      if (e.altKey && e.key === "r") { e.preventDefault(); reset() }
      if (e.altKey && e.key.toLowerCase() === "m") {
        e.preventDefault()
        if (memory.isIdle) void memory.memorize(null)
        return
      }
      if (e.altKey && e.key.toLowerCase() === "o") {
        e.preventDefault()
        if (filePicker.filesEnabled) filePicker.openFilePicker()
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
        // The command input lives in this tab's sidebar row.
        if (memory.isIdle) {
          if (sidebarCollapsed) toggleSidebar()
          memory.openMenu()
        }
      }
    }
    // Capture phase so the shortcuts win over handlers inside focused editors.
    window.addEventListener("keydown", handler, true)
    return () => {
      window.removeEventListener("keydown", handler, true)
    }
  }, [isActive, appMode, persistence.isTitled, persistence.handleQuickSave, persistence.setSaveModalOpen, reset, memory.isIdle, memory.memorize, memory.openMenu, filePicker.filesEnabled, filePicker.openFilePicker, sidebarCollapsed, toggleSidebar])

  // Picking a menu entry always closes the menu first; memorizing then replaces it with the review panel.
  const runMenuCommand = useCallback((command: string) => {
    memory.closeMenu()
    void runSlashCommand(command)
  }, [memory.closeMenu, runSlashCommand])

  const sidebarContext = useMemo<ChatSidebarContextValue>(() => ({
    onRemoveAttachment: turns.handleRemoveAttachment,
    onToggleAttachmentActive: turns.handleToggleAttachmentActive,
    onAttachmentContentChange: turns.handleAttachmentContentChange,
    filesEnabled: filePicker.filesEnabled,
    onOpenFilePicker: filePicker.openFilePicker,
    documents: docs.visibleDocuments,
    onSelectDocument: docs.handleDocumentSelect,
    onAttachArchitecture: docs.handleAttachArchitecture,
    architectureError: docs.architectureError,
    onOpenDocuments: activeProject ? docs.openDocuments : undefined,
    onCreateDocument: activeProject ? () => docs.setCreateDocOpen(true) : undefined,
    onDeleteDocument: docs.handleDeleteDocument,
    onRenameDocument: docs.handleRenameDocument,
    onDocumentSaved: docs.handleDocumentSaved,
    liveCanvasRef,
    editorModel: config.selectedModel,
  }), [
    turns.handleRemoveAttachment, turns.handleToggleAttachmentActive, turns.handleAttachmentContentChange,
    filePicker.filesEnabled, filePicker.openFilePicker,
    docs.visibleDocuments, docs.handleDocumentSelect, docs.openDocuments, docs.setCreateDocOpen,
    docs.handleDeleteDocument, docs.handleRenameDocument, docs.handleDocumentSaved,
    docs.handleAttachArchitecture, docs.architectureError,
    activeProject, config.selectedModel,
  ])

  const handleBranch = useCallback((qaIndex: number) => {
    if (!tab.historyFile) return
    createBranch(tab.historyFile, qaIndex).then(({ childFile }) => {
      onOpenChat(childFile)
    })
  }, [tab.historyFile, createBranch, onOpenChat])

  const sidebarToggle = (
    <button
      aria-label={sidebarCollapsed ? "Open sidebar" : "Close sidebar"}
      onClick={toggleSidebar}
      className="shrink-0 rounded p-1 text-ink-subtle hover:text-ink hover:bg-hover transition-colors"
    >
      {sidebarCollapsed ? <PanelLeft className="h-4 w-4" /> : <PanelLeftClose className="h-4 w-4" />}
    </button>
  )

  return (
    <div ref={containerRef} className="relative flex h-full overflow-hidden">
      <Sidebar
        onFileSelect={filePicker.handleFileSelect}
        onRootsChanged={filePicker.refreshFileList}
        onDocumentsChanged={docs.refreshDocuments}
        {...canvasView.sidebarProps}
        tabs={
          <SidebarTabs
            ownerTabId={tab.id}
            commandOpen={isActive && memory.menuOpen}
            onLabelSlot={memory.setLabelSlot}
            onMenuSlot={memory.setMenuSlot}
          />
        }
        renderSettings={() => (
          <SettingsMenu prompts={prompts} config={config} onConfigChange={updateConfig} onEditPrompts={() => setPromptEditorOpen(true)} models={models} researchEnabled={researchEnabled} />
        )}
        appMode={appMode}
        onAppModeChange={setAppMode}
      />
      <main className="flex-1 min-w-0 flex flex-col relative bg-paper">
        {appMode === "canvas" ? (
          // The canvas spans the full height, the bar floats over it. Architecture keeps its own header strip visible.
          <TopBar
            overlay={canvasView.view.kind !== "architecture"}
            left={<>
              <TopBarTransparent>{sidebarToggle}</TopBarTransparent>
              <TopBarPill ref={canvasView.setDrawingSlot} />
            </>}
            right={<TopBarPill>
              <div ref={canvasView.setViewSlot} className="contents" />
              <ThemeToggle />
            </TopBarPill>}
          />
        ) : (
          <TopBar
            left={<>
              <TopBarTransparent>{sidebarToggle}</TopBarTransparent>
              <TopBarPill>
                <ReasoningSelector levels={reasoningLevels} reasoningEffort={reasoningEffort} onReasoningChange={(v) => updateConfig({ reasoningEffort: v })} placement="below" />
                <ModelDropdown models={models} selectedModel={config.selectedModel} onSelect={(m) => updateConfig({ selectedModel: m })} onProvidersChange={saveModelProviders} onTabOrderChange={saveModelTabOrder} onDefaultsChange={saveModelDefaults} placement="below" />
              </TopBarPill>
              <TopBarTransparent ref={setWideToggleSlot} />
            </>}
            right={<>
              <TopBarPill>
                <TokenCounter usage={totalUsage} />
                <ThemeToggle />
              </TopBarPill>
              <TopBarTransparent ref={setContextToggleSlot} />
            </>}
          />
        )}
        <div className={`flex-1 min-h-0 relative transition-opacity duration-200 ${appMode === "canvas" ? "overflow-hidden" : ""}`}>
          {appMode === "canvas" && canvasView.view.kind === "architecture" ? (
            <ArchitectureWindow path={canvasView.view.path} maximized hostScale={1} edgeStyle={canvasView.architectureEdgeStyle} onEdgeStyleChange={canvasView.setArchitectureEdgeStyle} onNavigate={canvasView.navigateArchitecture} />
          ) : appMode === "canvas" ? (
            <CanvasWorkspace
              selected={canvasView.view.kind === "canvas" ? canvasView.view.selection : null}
              slug={activeProject}
              toolbarSlots={canvasView.toolbarSlots}
              onCloseEditor={canvasView.closeCanvas}
              onCreated={canvasView.openCreatedCanvas}
              onModeLabel={setModeLabel}
              editorModel={config.selectedModel}
            />
          ) : (
            <ChatSidebarProvider value={sidebarContext}>
              <ChatContainer
                chatId={chatId}
                messages={messages}
                isStreaming={isStreaming}
                onDeletePair={turns.handleDeletePair}
                extracting={extracting > 0}
                onSend={turns.handleSend}
                onCancel={cancel}
                onRegenerate={turns.handleRegenerate}
                onBranch={tab.historyFile ? handleBranch : undefined}
                focus={focus}
                branchMessageIdx={branchMessageIdx}
                chatInputRef={chatInputRef}
                slug={activeProject}
                chatMaxWidth={chatMaxWidth}
                contextToggleSlot={contextToggleSlot}
                wideToggleSlot={wideToggleSlot}
              />
            </ChatSidebarProvider>
          )}
          {isActive && filePicker.fileEditPath && (
            <DocumentEditor
              path={filePicker.fileEditPath}
              slug=""
              overlay
              persistOverride={(content) => writeRootFile(filePicker.fileEditPath!, content)}
              onClose={() => filePicker.setFileEditPath(null)}
              onModeLabel={setModeLabel}
              onNavigate={filePicker.setFileEditPath}
              model={config.selectedModel}
            />
          )}
          {isActive && filePicker.filePdfPath && (
            <div className="absolute inset-0 z-50 flex flex-col bg-paper">
              <div className="flex h-9 shrink-0 items-center border-b border-divider/50 px-4">
                <span className="truncate text-xs font-medium text-ink">{filePicker.filePdfPath.split("/").pop()}</span>
              </div>
              <div className="min-h-0 flex-1">
                <LazyPdfViewer url={fileViewerRawUrl(filePicker.filePdfPath)} />
              </div>
            </div>
          )}
        </div>
      </main>
      <SaveAsModal open={persistence.saveModalOpen} onClose={() => persistence.setSaveModalOpen(false)} onSave={persistence.handleSaveSubmit} />
      <CreateDocumentPanel open={docs.createDocOpen} onClose={() => docs.setCreateDocOpen(false)} onCreate={docs.handleCreateDocument} />
      <PromptEditor open={promptEditorOpen} onClose={() => setPromptEditorOpen(false)} onPromptsChanged={setPrompts} />
      {isActive && filePicker.filePickerOpen && filePicker.filesEnabled && (
        <FileViewer
          files={filePicker.pickerFiles.map((f) => f.path)}
          onSelect={filePicker.handleFileSelect}
          onClose={() => filePicker.setFilePickerOpen(false)}
          placeholder="Search files…"
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
      <MemoryCommandLayer bar={memory} items={menuItems} onRun={runMenuCommand} />
    </div>
  )
}
