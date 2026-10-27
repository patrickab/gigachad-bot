"use client"

import { useCallback, useEffect, useEffectEvent, useMemo, useRef, useState } from "react"
import { FolderOpen, Library, PanelLeft, PanelLeftClose, X } from "lucide-react"
import { Sidebar } from "@/features/navigation/Sidebar"
import { ChatContainer, type ChatFocus } from "@/features/chat/ChatContainer"
import type { AttachMenuItem, ChatInputHandle } from "@/features/chat/composer/ChatInput"
import { ArchitectureIcon } from "@/components/ui/ArchitectureIcon"
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
import { FilePanelSlot, FilePanelToggle } from "@/features/editor/FilePanel"
import { PdfAssistant } from "@/features/editor/pdf/PdfAssistant"
import { CanvasWorkspace } from "@/features/editor/canvas/CanvasWorkspace"
import { ArchitectureWindow } from "@/features/editor/architecture/ArchitectureWindow"
import { FactorySurface } from "@/features/factory/FactorySurface"
import { OmpModelSelector, OmpReasoningSelector } from "@/features/omp/OmpComposer"
import { OmpSurface } from "@/features/omp/OmpSurface"
import { OmpSystemPrompt } from "@/features/omp/OmpSystemPrompt"
import { TabSwitcher } from "@/features/navigation/TabManager"
import { TokenCounter } from "@/features/chat/TokenCounter"
import { TopBar, TopBarPill, TopBarTransparent } from "@/features/navigation/TopBar"
import { LazyPdfViewer } from "@/components/content/LazyPdfViewer"
import { LazyAnnotatedPdfViewer } from "@/features/editor/pdf/LazyAnnotatedPdfViewer"
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
import { liveReferencePaths, normalizeMessageAttachments } from "@/features/chat/attachments"
import { parseHistoryFile } from "@/lib/domain/historyFile"
import { fileViewerRawUrl } from "@/lib/backend/urls"
import { isNeovimKeyEvent } from "@/lib/utils"
import { MemoryCommandLayer } from "./MemoryCommandLayer"

/** One tab's workspace: the chat (or canvas) with its sidebar, top bar, modals and keyboard shortcuts. */
export function TabContent({ onOpenChat, onOpenProject, onProjectRenamed, onNewChat, onOpenItem, focus }: {
  onOpenChat: (historyFile: string, qaIndex?: number) => void
  onOpenProject: (slug: string) => void
  onProjectRenamed: (oldSlug: string, newSlug: string) => void
  onNewChat: (slug: string | null) => void
  onOpenItem: (item: WorkspaceItem, inNewTab?: boolean) => void
  focus: ChatFocus
}) {
  const { tab, isActive, setModeLabel, updateConfig, setAppMode, setWorkspaceItem, setHistoryFile, setTitle } = useTab()
  const projectSlug = tab.projectSlug
  const documentItem = tab.workspaceItem?.kind === "document" ? tab.workspaceItem : null
  const diffItem = tab.workspaceItem?.kind === "diff" ? tab.workspaceItem : null
  const ompItem = tab.workspaceItem?.kind === "omp" ? tab.workspaceItem : null
  const factoryItem = tab.workspaceItem?.kind === "factory" ? tab.workspaceItem : null
  const ompView = ompItem?.omp ?? null
  const chat = useChat()
  const { messages, isStreaming, cancel, reset, models, modelLoadError, saveModelProviders, saveModelTabOrder, saveModelDefaults, prompts, setPrompts, setMessages, totalUsage, setTotalUsage } = chat
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
  const [wideToggleSlot, setWideToggleSlot] = useState<HTMLElement | null>(null)
  const [pdfToolbarSlot, setPdfToolbarSlot] = useState<HTMLElement | null>(null)
  const [pdfChatOpen, setPdfChatOpen] = useState(false)
  const [chatColumn, setChatColumn] = useState<HTMLDivElement | null>(null)
  const chatInputRef = useRef<ChatInputHandle>(null)
  const loadIdRef = useRef(0)

  const { view: canvasView, sidebarProps: canvasSidebarProps, navigateArchitecture, closeCanvas, openCreatedCanvas, architectureEdgeStyle, setArchitectureEdgeStyle, toolbarSlots, setDrawingSlot, setViewSlot } = useCanvasView()
  const filePicker = useFilePicker({ isActive, hasMessages: messages.length > 0, chatInputRef, setExtracting })
  const docs = useProjectDocuments({ isActive, appMode, projectSlug, chatId, chatInputRef, setExtracting })
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
  const { syncReferences } = turns

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
        const loaded = data.messages.map((m) => normalizeMessageAttachments(m, data.chat_id, slug))
        setMessages(loaded)
        void syncReferences(liveReferencePaths(loaded.flatMap((m) => m.attachments ?? [])))
      }
      if (data.chat_id) setChatId(data.chat_id)
      if (data.title) setTitle(data.title)
      if (data.usage) setTotalUsage(data.usage)
      setLoadedBranch({ historyFile, messageIdx: data.branch_message_idx ?? null })
    }).catch(() => { })
    return () => { loadIdRef.current++ }
  }, [tab.historyFile, setMessages, setTitle, setTotalUsage, syncReferences])

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

  const attachmentActions = useMemo(() => ({
    onToggleActive: turns.handleToggleAttachmentActive,
    onRemove: turns.handleRemoveAttachment,
    onContentChange: turns.handleAttachmentContentChange,
    onPreviewOpen: (path: string) => syncReferences([path]),
  }), [turns.handleToggleAttachmentActive, turns.handleRemoveAttachment, turns.handleAttachmentContentChange, syncReferences])

  const { handleAttachArchitecture } = docs
  const attachMenuItems = useMemo<AttachMenuItem[]>(() => [
    ...(filePicker.filesEnabled ? [{ label: "Mounted file", icon: FolderOpen, onClick: filePicker.openFilePicker }] : []),
    ...(projectSlug ? [{ label: "Project document", icon: Library, onClick: docs.openDocuments }] : []),
    ...(handleAttachArchitecture ? [{ label: "Architecture", icon: ArchitectureIcon, onClick: () => void handleAttachArchitecture() }] : []),
  ], [filePicker.filesEnabled, filePicker.openFilePicker, projectSlug, docs.openDocuments, handleAttachArchitecture])

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
  const filePanelToggle = <TopBarTransparent><FilePanelToggle /></TopBarTransparent>
  const isPdfDocument = appMode === "document" && !diffItem && !!documentItem?.path.toLowerCase().endsWith(".pdf")
  // Scrolling surfaces run on under the floating TopBar. The rest start below it.
  const surfaceRunsUnderTopBar = appMode === "canvas" ? canvasView.kind !== "architecture"
    : appMode === "document" ? isPdfDocument
    : appMode !== "omp" && appMode !== "factory"

  return (
    <div className="relative flex h-full overflow-hidden">
      <Sidebar
        onFileSelect={(path) => onOpenItem({ kind: "document", path, projectSlug: null, rootFile: true })}
        onDiffSelect={(path) => onOpenItem({ kind: "diff", path, projectSlug: null })}
        onRootsChanged={filePicker.refreshFileList}
        onDocumentsChanged={docs.refreshDocuments}
        {...canvasSidebarProps}
        onProjectSelect={onOpenProject}
        onProjectRenamed={onProjectRenamed}
        onNewChat={onNewChat}
        onDocumentSelect={(path, scope) => onOpenItem({ kind: "document", path, projectSlug: scope || null })}
        activeDocumentPath={appMode === "document" ? documentItem?.path : null}
        onCanvasSelect={(path, scope) => onOpenItem({ kind: "canvas", path, projectSlug: scope || null })}
        onArchitectureSelect={(slug) => onOpenItem({ kind: "architecture", path: `graph/${slug}/`, projectSlug: slug })}
        onItemOpen={onOpenItem}
        renderSettings={() => (
          <SettingsMenu prompts={prompts} config={config} onConfigChange={updateConfig} onEditPrompts={() => setPromptEditorOpen(true)} models={models} researchEnabled={researchEnabled} />
        )}
        appMode={appMode}
      />
      <main className="flex-1 min-w-0 flex flex-col relative bg-paper">
        {appMode === "canvas" ? (
          <TopBar
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
              {filePanelToggle}
            </>}
          />
        ) : appMode === "omp" ? (
          <TopBar
            center={<TabSwitcher />}
            left={<>
              <TopBarTransparent>{sidebarToggle}</TopBarTransparent>
              {ompView && <TopBarPill>
                <OmpReasoningSelector view={ompView} />
              </TopBarPill>}
              {ompView && <TopBarTransparent>
                <OmpModelSelector view={ompView} models={models} loadError={modelLoadError} onProvidersChange={saveModelProviders} onTabOrderChange={saveModelTabOrder} onDefaultsChange={saveModelDefaults} placement="below" />
              </TopBarTransparent>}
            </>}
            right={<>
              <TopBarPill>
                {ompView && <OmpSystemPrompt view={ompView} />}
                <ThemeToggle />
              </TopBarPill>
              {filePanelToggle}
            </>}
          />
        ) : appMode === "factory" ? (
          <TopBar
            center={<TabSwitcher />}
            left={<TopBarTransparent>{sidebarToggle}</TopBarTransparent>}
            right={<><TopBarPill><ThemeToggle /></TopBarPill>{filePanelToggle}</>}
          />
        ) : appMode === "document" ? (
          <TopBar
            center={<TabSwitcher />}
            left={<>
              <TopBarTransparent>{sidebarToggle}</TopBarTransparent>
              <TopBarTransparent>
                <ModelDropdown models={models} loadError={modelLoadError} selectedModel={config.selectedModel} onSelect={(model) => updateConfig({ selectedModel: model })} onProvidersChange={saveModelProviders} onTabOrderChange={saveModelTabOrder} onDefaultsChange={saveModelDefaults} placement="below" />
              </TopBarTransparent>
            </>}
            right={<>
              {isPdfDocument && <TopBarPill ref={setPdfToolbarSlot} />}
              <TopBarTransparent>
                <ThemeToggle />
                <button aria-label="Close file" onClick={() => setAppMode("chat")} className="rounded p-1 text-ink-subtle hover:text-ink hover:bg-hover">
                  <X className="h-4 w-4" />
                </button>
              </TopBarTransparent>
              {!isPdfDocument && filePanelToggle}
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
                <ModelDropdown models={models} loadError={modelLoadError} selectedModel={config.selectedModel} onSelect={(m) => updateConfig({ selectedModel: m })} onProvidersChange={saveModelProviders} onTabOrderChange={saveModelTabOrder} onDefaultsChange={saveModelDefaults} placement="below" chatColumn={chatColumn} />
              </TopBarTransparent>
              <TopBarTransparent ref={setWideToggleSlot} />
            </>}
            right={<>
              <TopBarPill>
                <TokenCounter usage={totalUsage} />
                <ThemeToggle />
              </TopBarPill>
              {filePanelToggle}
            </>}
          />
        )}
        <div data-workspace-surface className={`absolute inset-x-0 bottom-0 ${surfaceRunsUnderTopBar ? "top-0" : "top-11"} ${appMode === "canvas" ? "overflow-hidden" : ""}`}>
          {appMode === "omp" && ompItem && ompView ? (
            <OmpSurface view={ompView} onViewChange={(omp) => setWorkspaceItem({ ...ompItem, omp })} />
          ) : appMode === "factory" && factoryItem ? (
            <FactorySurface item={factoryItem} />
          ) : appMode === "document" && diffItem ? (
            <DiffViewer key={diffItem.path} root={diffItem.path} />
          ) : appMode === "document" && documentItem ? (
            isPdfDocument ? (
              <PdfAssistant key={documentItem.path} path={documentItem.path} model={config.selectedModel} chatOpen={pdfChatOpen} onChatOpenChange={setPdfChatOpen} topInset={44}>
                <LazyAnnotatedPdfViewer path={documentItem.path} topInset={44} toolbarSlot={pdfToolbarSlot} />
              </PdfAssistant>
            ) : (
              <DocumentEditor
                path={documentItem.path}
                slug={documentItem.projectSlug ?? ""}
                overlay
                persistOverride={documentItem.rootFile ? (content) => writeRootFile(documentItem.path, content) : undefined}
                onClose={() => setAppMode("chat")}
                onSaved={docs.refreshDocuments}
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
              attachmentActions={attachmentActions}
              attachMenuItems={attachMenuItems}
              composerNotice={docs.architectureError || undefined}
              wideToggleSlot={wideToggleSlot}
            />
          )}
        </div>
        {isActive && (filePicker.fileEditPath || filePicker.filePdfPath) && (
          // File-picker overlays sit below the bar, over whatever surface is open.
          <div className="absolute inset-x-0 top-11 bottom-0">
            {filePicker.fileEditPath && (
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
            {filePicker.filePdfPath && (
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
        )}
      </main>
      <FilePanelSlot />
      <SaveAsModal open={persistence.saveModalOpen} onClose={() => persistence.setSaveModalOpen(false)} onSave={persistence.handleSaveSubmit} />
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
          onClose={() => docs.setDocumentOpen(false)}
        />
      )}
      <MemoryCommandLayer bar={memory} items={menuItems} onRun={runMenuCommand} />
    </div>
  )
}
