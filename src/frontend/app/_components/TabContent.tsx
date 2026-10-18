"use client"

import { useCallback, useEffect, useEffectEvent, useMemo, useRef, useState } from "react"
import { PanelLeft, PanelLeftClose, X } from "lucide-react"
import { Sidebar } from "@/features/navigation/Sidebar"
import { ChatContainer, type ChatFocus } from "@/features/chat/ChatContainer"
import { ChatSidebarProvider, type ChatSidebarContextValue } from "@/features/chat/context-panel/ChatContextPanel"
import type { ChatInputHandle } from "@/features/chat/composer/ChatInput"
import { ModelDropdown } from "@/features/settings/ModelDropdown"
import { SettingsMenu } from "@/features/settings/SettingsMenu"
import { PromptEditor } from "@/features/settings/PromptEditor"
import { ReasoningSelector, effectiveReasoningEffort, useReasoningLevels } from "@/features/settings/ReasoningSelector"
import { ThemeToggle } from "@/features/settings/ThemeToggle"
import { SaveAsModal } from "@/components/ui/SaveAsModal"
import { FileViewer } from "@/features/editor/FileViewer"
import { DocumentPicker } from "@/features/editor/DocumentPicker"
import { DocumentEditor } from "@/features/editor/DocumentEditor"
import { DiffViewer } from "@/features/editor/diff/DiffViewer"
import { CreateDocumentPanel } from "@/features/editor/CreateDocumentPanel"
import { CanvasWorkspace } from "@/features/editor/canvas/CanvasWorkspace"
import { ArchitectureWindow } from "@/features/editor/architecture/ArchitectureWindow"
import { OmpModelSelector } from "@/features/omp/OmpComposer"
import { OmpSurface } from "@/features/omp/OmpSurface"
import { OmpSystemPrompt } from "@/features/omp/OmpSystemPrompt"
import { TabSwitcher } from "@/features/navigation/TabManager"
import { TokenCounter } from "@/features/chat/TokenCounter"
import { TopBar, TopBarPill, TopBarTransparent } from "@/features/navigation/TopBar"
import { LazyPdfViewer } from "@/components/content/LazyPdfViewer"
import { useBranches } from "@/features/navigation/BranchContext"
import { useEnabledTools } from "@/features/chat/EnabledToolsContext"
import { useSidebar } from "@/features/navigation/SidebarContext"
import { useTab, type WorkspaceItem } from "@/contexts/TabContext"
import { useCanvasView } from "@/features/editor/canvas/useCanvasView"
import { useChat } from "@/features/chat/useChat"
import { useChatPersistence } from "@/features/chat/useChatPersistence"
import { useChatTurns } from "@/features/chat/useChatTurns"
import { useFilePicker } from "@/features/chat/useFilePicker"
import { useMemoryCommandBar } from "@/features/memory/useMemoryCommandBar"
import { useProjectDocuments } from "@/features/chat/useProjectDocuments"
import { useSlashCommands } from "@/features/navigation/useSlashCommands"
import { loadChatHistory, writeRootFile } from "@/lib/backend/api"
import { normalizeMessageAttachments } from "@/features/chat/attachments"
import { parseHistoryFile } from "@/lib/domain/historyFile"
import { fileViewerRawUrl } from "@/lib/backend/urls"
import { isNeovimKeyEvent } from "@/lib/utils"
import { MemoryCommandLayer } from "./MemoryCommandLayer"

/** One tab's workspace: the chat (or canvas) with its sidebar, top bar, modals and keyboard shortcuts. */
export function TabContent({ onOpenChat, onOpenProject, onNewChat, onOpenItem, focus }: {
  onOpenChat: (historyFile: string, qaIndex?: number) => void
  onOpenProject: (slug: string) => void
  onNewChat: (slug: string | null) => void
  onOpenItem: (item: WorkspaceItem, inNewTab?: boolean) => void
  focus: ChatFocus
}) {
  const { tab, isActive, setModeLabel, updateConfig, setAppMode, setWorkspaceItem, setHistoryFile, setTitle } = useTab()
  const projectSlug = tab.projectSlug
  const documentItem = tab.workspaceItem?.kind === "document" ? tab.workspaceItem : null
  const diffItem = tab.workspaceItem?.kind === "diff" ? tab.workspaceItem : null
  const ompItem = tab.workspaceItem?.kind === "omp" ? tab.workspaceItem : null
  const ompView = ompItem?.omp ?? null
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
  const [loadedBranch, setLoadedBranch] = useState<{ historyFile: string | undefined; messageIdx: number | null }>({ historyFile: undefined, messageIdx: null })
  // Index where this chat branched off its parent, null for a main chat or while its history loads.
  const branchMessageIdx = loadedBranch.historyFile === tab.historyFile ? loadedBranch.messageIdx : null
  const [extracting, setExtracting] = useState(0)
  const [promptEditorOpen, setPromptEditorOpen] = useState(false)
  const [contextToggleSlot, setContextToggleSlot] = useState<HTMLElement | null>(null)
  const [wideToggleSlot, setWideToggleSlot] = useState<HTMLElement | null>(null)
  const [chatColumn, setChatColumn] = useState<HTMLDivElement | null>(null)
  const chatInputRef = useRef<ChatInputHandle>(null)
  const liveCanvasRef = useRef<{ path: string; content: string } | null>(null)
  const loadIdRef = useRef(0)

  const { view: canvasView, sidebarProps: canvasSidebarProps, navigateArchitecture, closeCanvas, openCreatedCanvas, architectureEdgeStyle, setArchitectureEdgeStyle, toolbarSlots, setDrawingSlot, setViewSlot } = useCanvasView()
  const filePicker = useFilePicker({ isActive, hasMessages: messages.length > 0, chatInputRef, setExtracting })
  const docs = useProjectDocuments({ isActive, appMode, projectSlug, chatId, chatInputRef, liveCanvasRef, setExtracting, setMessages })
  const persistence = useChatPersistence({ tab, projectSlug, messages, chatId, usage, refreshAll, setHistoryFile })
  const memory = useMemoryCommandBar({ messages, chatId, branchMessageIdx })
  const { menuItems, runSlashCommand } = useSlashCommands({
    filesEnabled: filePicker.filesEnabled,
    hasProject: projectSlug != null,
    openFilePicker: filePicker.openFilePicker,
    memorize: memory.memorize,
  })
  const turns = useChatTurns({
    chat,
    chatId,
    projectSlug,
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

  useEffect(() => {
    if (isActive) setActiveFile(tab.historyFile)
  }, [isActive, tab.historyFile, setActiveFile])

  useEffect(() => {
    if (isActive) setActiveQaIndex(focus.qaIndex)
  }, [isActive, focus.qaIndex, setActiveQaIndex])
  useEffect(() => cancel, [cancel])


  useEffect(() => {
    if (!tab.historyFile) return
    const historyFile = tab.historyFile
    const loadId = ++loadIdRef.current
    setMessages([])
    setTotalUsage({ total_tokens: 0, prompt_tokens: 0, completion_tokens: 0 })
    // Project chats load by their own slug, so switching projects never reloads open tabs.
    const { slug } = parseHistoryFile(historyFile)
    loadChatHistory(historyFile).then((data) => {
      if (loadIdRef.current !== loadId) return
      if (data.messages.length > 0) {
        setMessages(data.messages.map((m) => normalizeMessageAttachments(m, data.chat_id, slug)))
      }
      if (data.chat_id) setChatId(data.chat_id)
      if (data.title) setTitle(data.title)
      if (data.usage) setTotalUsage(data.usage)
      setLoadedBranch({ historyFile, messageIdx: data.branch_message_idx ?? null })
    }).catch(() => { })
    return () => { loadIdRef.current++ }
  }, [tab.historyFile, setMessages, setTitle, setTotalUsage])

  const onKeyDown = useEffectEvent((e: KeyboardEvent) => {
    if (isNeovimKeyEvent(e)) return
    if (e.altKey && e.key === "s") {
      e.preventDefault()
      if (persistence.isTitled) persistence.handleQuickSave()
      else persistence.setSaveModalOpen(true)
    }
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
      if (memory.isIdle) {
        memory.openMenu()
      }
    }
  })

  useEffect(() => {
    if (!isActive || appMode !== "chat") return
    // Capture phase so the shortcuts win over handlers inside focused editors.
    window.addEventListener("keydown", onKeyDown, true)
    return () => {
      window.removeEventListener("keydown", onKeyDown, true)
    }
  }, [isActive, appMode])

  // Picking a menu entry always closes the menu first; memorizing then replaces it with the review panel.
  const { closeMenu: closeMemoryMenu } = memory
  const runMenuCommand = useCallback((command: string) => {
    closeMemoryMenu()
    void runSlashCommand(command)
  }, [closeMemoryMenu, runSlashCommand])

  const { setCreateDocOpen } = docs
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
    onOpenDocuments: projectSlug ? docs.openDocuments : undefined,
    onCreateDocument: projectSlug ? () => setCreateDocOpen(true) : undefined,
    onDeleteDocument: docs.handleDeleteDocument,
    onRenameDocument: docs.handleRenameDocument,
    onDocumentSaved: docs.handleDocumentSaved,
    liveCanvasRef,
    editorModel: config.selectedModel,
  }), [
    turns.handleRemoveAttachment, turns.handleToggleAttachmentActive, turns.handleAttachmentContentChange,
    filePicker.filesEnabled, filePicker.openFilePicker,
    docs.visibleDocuments, docs.handleDocumentSelect, docs.openDocuments, setCreateDocOpen,
    docs.handleDeleteDocument, docs.handleRenameDocument, docs.handleDocumentSaved,
    docs.handleAttachArchitecture, docs.architectureError,
    projectSlug, config.selectedModel,
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
    <div className="relative flex h-full overflow-hidden">
      <Sidebar
        onFileSelect={(path) => onOpenItem({ kind: "document", path, projectSlug: null, rootFile: true })}
        onDiffSelect={(path) => onOpenItem({ kind: "diff", path, projectSlug: null })}
        onRootsChanged={filePicker.refreshFileList}
        onDocumentsChanged={docs.refreshDocuments}
        {...canvasSidebarProps}
        onProjectSelect={onOpenProject}
        onNewChat={onNewChat}
        onDocumentSelect={(path, scope) => onOpenItem({ kind: "document", path, projectSlug: scope || null })}
        activeDocumentPath={appMode === "document" ? documentItem?.path : null}
        onCanvasSelect={(path, scope) => onOpenItem({ kind: "canvas", path, projectSlug: scope || null })}
        onArchitectureSelect={(slug) => onOpenItem({ kind: "architecture", path: `graph/${slug}/`, projectSlug: slug })}
        onOmpOpen={onOpenItem}
        renderSettings={() => (
          <SettingsMenu prompts={prompts} config={config} onConfigChange={updateConfig} onEditPrompts={() => setPromptEditorOpen(true)} models={models} researchEnabled={researchEnabled} />
        )}
        appMode={appMode}
      />
      <main className="flex-1 min-w-0 flex flex-col relative bg-paper">
        {appMode === "canvas" ? (
          // The canvas spans the full height, the bar floats over it. Architecture keeps its own header strip visible.
          <TopBar
            overlay={canvasView.kind !== "architecture"}
            center={<TabSwitcher />}
            left={<>
              <TopBarTransparent>{sidebarToggle}</TopBarTransparent>
              <TopBarPill ref={setDrawingSlot} />
            </>}
            right={<>
              <TopBarPill>
                <div ref={setViewSlot} className="contents" />
                <ThemeToggle />
              </TopBarPill>
            </>}
          />
        ) : appMode === "omp" ? (
          <TopBar
            center={<TabSwitcher />}
            left={<>
              <TopBarTransparent>{sidebarToggle}</TopBarTransparent>
              {ompView && <TopBarTransparent>
                <OmpModelSelector view={ompView} models={models} onProvidersChange={saveModelProviders} onTabOrderChange={saveModelTabOrder} onDefaultsChange={saveModelDefaults} placement="below" />
              </TopBarTransparent>}
            </>}
            right={<TopBarPill>
              {ompView && <OmpSystemPrompt view={ompView} />}
              <ThemeToggle />
            </TopBarPill>}
          />
        ) : appMode === "document" ? (
          <TopBar
            center={<TabSwitcher />}
            left={<>
              <TopBarTransparent>{sidebarToggle}</TopBarTransparent>
              <TopBarTransparent>
                <ModelDropdown models={models} selectedModel={config.selectedModel} onSelect={(model) => updateConfig({ selectedModel: model })} onProvidersChange={saveModelProviders} onTabOrderChange={saveModelTabOrder} onDefaultsChange={saveModelDefaults} placement="below" />
              </TopBarTransparent>
            </>}
            right={<>
              <TopBarTransparent>
                <ThemeToggle />
                <button aria-label="Close file" onClick={() => setAppMode("chat")} className="rounded p-1 text-ink-subtle hover:text-ink hover:bg-hover">
                  <X className="h-4 w-4" />
                </button>
              </TopBarTransparent>
            </>}
          />
        ) : (
          <TopBar
            center={<TabSwitcher />}
            left={<>
              <TopBarTransparent>{sidebarToggle}</TopBarTransparent>
              <TopBarPill>
                <ReasoningSelector levels={reasoningLevels} reasoningEffort={reasoningEffort} onReasoningChange={(v) => updateConfig({ reasoningEffort: v })} placement="below" />
              </TopBarPill>
              <TopBarTransparent>
                <ModelDropdown models={models} selectedModel={config.selectedModel} onSelect={(m) => updateConfig({ selectedModel: m })} onProvidersChange={saveModelProviders} onTabOrderChange={saveModelTabOrder} onDefaultsChange={saveModelDefaults} placement="below" chatColumn={chatColumn} />
              </TopBarTransparent>
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
        <div data-workspace-surface className={`flex-1 min-h-0 relative ${appMode === "canvas" ? "overflow-hidden" : ""}`}>
          {appMode === "omp" && ompItem && ompView ? (
            <OmpSurface view={ompView} onViewChange={(omp) => setWorkspaceItem({ ...ompItem, omp })} />
          ) : appMode === "document" && diffItem ? (
            <DiffViewer key={diffItem.path} root={diffItem.path} />
          ) : appMode === "document" && documentItem ? (
            documentItem.path.toLowerCase().endsWith(".pdf") ? (
              <LazyPdfViewer url={fileViewerRawUrl(documentItem.path)} />
            ) : (
              <DocumentEditor
                path={documentItem.path}
                slug={documentItem.projectSlug ?? ""}
                overlay
                persistOverride={documentItem.rootFile ? (content) => writeRootFile(documentItem.path, content) : undefined}
                onClose={() => setAppMode("chat")}
                onSaved={docs.handleDocumentSaved}
                onModeLabel={setModeLabel}
                onNavigate={(path) => setWorkspaceItem({ ...documentItem, path })}
                model={config.selectedModel}
              />
            )
          ) : appMode === "canvas" && canvasView.kind === "architecture" ? (
            <ArchitectureWindow path={canvasView.path} maximized hostScale={1} edgeStyle={architectureEdgeStyle} onEdgeStyleChange={setArchitectureEdgeStyle} onNavigate={navigateArchitecture} toolbarSlots={toolbarSlots} />
          ) : appMode === "canvas" ? (
            <CanvasWorkspace
              selected={canvasView.kind === "canvas" ? canvasView.selection : null}
              slug={projectSlug}
              toolbarSlots={toolbarSlots}
              onCloseEditor={closeCanvas}
              onCreated={openCreatedCanvas}
              onModeLabel={setModeLabel}
              editorModel={config.selectedModel}
            />
          ) : (
            <ChatSidebarProvider value={sidebarContext}>
              <ChatContainer
                chatId={chatId}
                onChatColumnChange={setChatColumn}
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
                slug={projectSlug}
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
          projectSlug={projectSlug}
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
