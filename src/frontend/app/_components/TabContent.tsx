"use client"

import { useCallback, useEffect, useState, type RefCallback } from "react"
import { anchorSceneFallback } from "deepfocus-ambient"
import { ErrorBoundary } from "@/components/ui/ErrorBoundary"
import { PanelToggleButton } from "@/components/ui/PanelToggleButton"
import { useSidebar } from "@/features/navigation/SidebarContext"
import { ChatContainer, type ChatFocus } from "@/features/chat/ChatContainer"
import { newChatDraftKey } from "@/features/chat/composer/useComposerDraft"
import { useContentAreaRef } from "@/components/ui/ContentArea"
import { PromptEditor } from "@/features/settings/PromptEditor"
import { SaveAsModal } from "@/components/ui/SaveAsModal"
import { FileViewer } from "@/features/editor/FileViewer"
import { DocumentPicker } from "@/features/editor/DocumentPicker"
import { TabPicker } from "@/features/navigation/TabPicker"
import { DocumentEditor } from "@/features/editor/DocumentEditor"
import { FilePanelSlot } from "@/features/editor/FilePanel"
import { TopBarSlotProvider } from "@/features/navigation/TopBarSlots"
import { RawFileView } from "@/features/editor/RawFileView"
import { newPiItem, useTab, type WorkspaceItem } from "@/contexts/TabContext"
import type { Attachment } from "@/lib/domain/types"
import { clampThinking, useNewSessionModels, type PiLaunch } from "@/features/pi/PiTab"
import { piGatewayId, type PiModelControls } from "@/features/pi/PiTopBar"
import type { PiModel } from "@/features/pi/types"
import { PiWorkspacePicker } from "@/features/pi/PiWorkspacePicker"
import { piSessionItem } from "@/features/pi/usePiSidebar"
import { useCanvasView } from "@/features/editor/canvas/useCanvasView"
import { writeRootFile } from "@/lib/backend/api"
import { TabSidebar } from "./TabSidebar"
import { useTabChat } from "./useTabChat"
import { WorkspaceSurface, type ModelProps } from "./WorkspaceSurface"

interface TabContentProps {
  onOpenChat: (historyFile: string, qaIndex?: number) => void
  onOpenProject: (slug: string) => void
  onProjectRenamed: (oldSlug: string, newSlug: string) => void
  onDocumentRenamed: (oldPath: string, newPath: string) => void
  onNewChat: (slug: string | null) => void
  onOpenItem: (item: WorkspaceItem, inNewTab?: boolean) => void
  focus: ChatFocus
}

/**
 * Keep `--under-left` and `--under-right` on the tab's main area: the px between it and the tab's edges, the left sidebar's and the file panel's share with their gutters.
 * A surface running under the sidebars (`runsUnderSidebars`) extends its body by them, the cards float over it.
 */
function measureSidebarInsets(main: HTMLElement): (() => void) | undefined {
  if (typeof ResizeObserver === "undefined") return
  const tab = main.parentElement!
  const update = () => {
    main.style.setProperty("--under-left", `${main.offsetLeft}px`)
    main.style.setProperty("--under-right", `${tab.clientWidth - main.offsetLeft - main.offsetWidth}px`)
  }
  const observer = new ResizeObserver(update)
  observer.observe(main)
  observer.observe(tab)
  return () => observer.disconnect()
}

/**
 * The checkout an empty chat's first message starts a Pi session in, picked by path.
 *
 * Any folder may hold a Pi session. `folderError` stays for callers, it is always null.
 */
export function usePiFolder(): { folder: string | null; folderError: string | null; pickFolder: (picked: string | null) => void } {
  const [folder, setFolder] = useState<string | null>(null)
  const pickFolder = useCallback((picked: string | null) => {
    setFolder(picked === null ? null : picked.replace(/\/+$/, "") || "/")
  }, [])
  return { folder, folderError: null, pickFolder }
}

/** One tab's workspace: the chat (or canvas) with its sidebar, top bar, modals and keyboard shortcuts. */
export function TabContent(props: TabContentProps) {
  return <ErrorBoundary><TopBarSlotProvider><TabWorkspace {...props} /></TopBarSlotProvider></ErrorBoundary>
}

/** The tab's chat surfaces and the top bar whose slots they render their controls into, beside the sidebar it never re-renders. */
function TabWorkspace({ onOpenChat, onOpenProject, onProjectRenamed, onDocumentRenamed, onNewChat, onOpenItem, focus }: TabContentProps) {
  const { tab, isActive, setModeLabel } = useTab()
  const { projectSlug, config } = tab
  const { collapsed: sidebarCollapsed, toggleCollapsed: toggleSidebar } = useSidebar()
  const registerContentArea = useContentAreaRef()
  const [mainArea, setMainArea] = useState<HTMLElement | null>(null)
  const contentAreaRef = useCallback<RefCallback<HTMLElement>>((main) => {
    if (!main) return
    setMainArea(main)
    const unregister = registerContentArea?.(main)
    const stopMeasuring = measureSidebarInsets(main)
    return () => {
      setMainArea(null)
      unregister?.()
      stopMeasuring?.()
    }
  }, [registerContentArea])
  // Without a composer anchor the scene sits at the middle of the area between the sidebars, not of the window
  useEffect(() => {
    if (!isActive || !mainArea) return
    return anchorSceneFallback(() => {
      const box = mainArea.getBoundingClientRect()
      if (box.height === 0 || box.width === 0) return null
      return { horizonY: box.top + box.height * 0.6, lightX: box.left + box.width / 2, lightY: box.top + box.height * 0.52 }
    })
  }, [isActive, mainArea])
  const [promptEditorOpen, setPromptEditorOpen] = useState(false)
  const openPromptEditor = useCallback(() => setPromptEditorOpen(true), [])
  const [chatColumn, setChatColumn] = useState<HTMLDivElement | null>(null)
  const canvas = useCanvasView()
  const {
    chat, chatId, branchMessageIdx, historyLoading, extracting, chatInputRef, reasoningLevels, reasoningEffort,
    filePicker, docs, persistence, turns, attachmentActions, attachMenuItems, onBranch,
  } = useTabChat({ focus, onOpenChat, onOpenItem })
  const { models } = chat
  const { standaloneFile } = filePicker
  const modelProps: ModelProps = { models, loadError: chat.modelLoadError, onProvidersChange: chat.saveModelProviders, onTabOrderChange: chat.saveModelTabOrder, onDefaultsChange: chat.saveModelDefaults, placement: "below" }

  // A folder picked in an empty chat makes its first message start a Pi session there, the Pi tab sends it once the session is ready.
  // The picker shows only in an empty chat, so a folder left over from one never redirects another chat's message
  const { folder, folderError, pickFolder } = usePiFolder()
  const [launch, setLaunch] = useState<PiLaunch | null>(null)
  const isEmptyChat = chat.messages.length === 0 && !historyLoading && !chat.isStreaming
  // With a folder picked the model and thinking pickers offer Pi's catalog, the first message starts the session on them
  const piStartActive = folder !== null && isEmptyChat
  const piCatalog = useNewSessionModels(piStartActive)
  const [piChoice, setPiChoice] = useState<{ model: PiModel | null; thinking: string }>({ model: null, thinking: "medium" })
  const piStart: PiModelControls | null = piStartActive ? {
    model: piChoice.model,
    models: piCatalog.models,
    thinkingLevel: piChoice.thinking,
    thinkingLevels: piChoice.model ? piCatalog.levels.get(piGatewayId(piChoice.model)) ?? [] : piCatalog.defaultLevels,
    onModel: (model) => setPiChoice((current) => ({ model, thinking: clampThinking(current.thinking, piCatalog.levels.get(piGatewayId(model)) ?? []) })),
    onThinking: (thinking) => setPiChoice((current) => ({ ...current, thinking })),
  } : null
  const handleSend = (text: string, attachments: Attachment[]) => {
    if (folder === null || !isEmptyChat) {
      turns.handleSend(text, attachments)
      return
    }
    setLaunch({ text, attachments, model: piChoice.model, thinking: piChoice.thinking })
    pickFolder(null)
    onOpenItem(newPiItem(folder, projectSlug))
  }

  const chatBody = (
    <ChatContainer
      chatId={chatId}
      draftKey={tab.historyFile ? `chat:${projectSlug ?? ""}:${tab.historyFile}` : newChatDraftKey(tab.id)}
      onChatColumnChange={setChatColumn}
      messages={chat.messages}
      historyLoading={historyLoading}
      isStreaming={chat.isStreaming}
      onDeletePair={turns.handleDeletePair}
      extracting={extracting}
      onSend={handleSend}
      onCancel={chat.cancel}
      onRegenerate={turns.handleRegenerate}
      onBranch={onBranch}
      focus={focus}
      branchMessageIdx={branchMessageIdx}
      chatInputRef={chatInputRef}
      slug={projectSlug}
      attachmentActions={attachmentActions}
      attachMenuItems={attachMenuItems}
      composerLeading={isEmptyChat ? <PiWorkspacePicker folder={folder} onChange={pickFolder} onOpenSession={(session, repo) => onOpenItem(piSessionItem(session, repo ?? newPiItem(session.cwd, projectSlug).pi!.repo, projectSlug))} /> : undefined}
      composerNotice={(isEmptyChat && folderError) || docs.architectureError || undefined}
    />
  )

  return (
    <div className="relative flex h-full overflow-hidden">
      <TabSidebar
        onOpenProject={onOpenProject}
        onProjectRenamed={onProjectRenamed}
        onDocumentRenamed={onDocumentRenamed}
        onNewChat={onNewChat}
        onOpenItem={onOpenItem}
        onEditPrompts={openPromptEditor}
      />
      {/* No paper of its own: the ocean backdrop, or a shared frame laid under this tab, shows through */}
      <main ref={contentAreaRef} className="flex-1 min-w-0 flex flex-col relative">
        <WorkspaceSurface
          sidebarToggle={sidebarCollapsed && <PanelToggleButton side="left" open={false} onClick={toggleSidebar} openLabel="Open sidebar" closeLabel="Close sidebar" />}
          canvas={canvas}
          modelProps={modelProps}
          reasoningLevels={reasoningLevels}
          reasoningEffort={reasoningEffort}
          totalUsage={chat.totalUsage}
          chatColumn={chatColumn}
          chatBody={chatBody}
          launch={launch}
          piStart={piStart}
          onLaunched={() => setLaunch(null)}
          onOpenItem={onOpenItem}
        />
        {isActive && standaloneFile && (
          // File-picker overlays sit below the bar, over whatever surface is open.
          <div className="absolute inset-x-0 top-11 bottom-0">
            {/* Its own scope: a canvas opened here keeps its toolbar inline, the top bar belongs to the surface underneath. */}
            {standaloneFile.kind === "edit" && (
              <TopBarSlotProvider>
                <DocumentEditor
                  path={standaloneFile.path}
                  slug=""
                  overlay
                  persistOverride={(content) => writeRootFile(standaloneFile.path, content)}
                  onClose={() => filePicker.setStandaloneFile(null)}
                  onModeLabel={setModeLabel}
                  onNavigate={filePicker.openFile}
                  model={config.selectedModel}
                />
              </TopBarSlotProvider>
            )}
            {standaloneFile.kind === "pdf" && (
              <div className="absolute inset-0 z-50 flex flex-col bg-paper">
                <div className="flex h-9 shrink-0 items-center border-b border-divider/50 px-4">
                  <span className="truncate text-xs font-medium text-ink">{standaloneFile.path.split("/").pop()}</span>
                </div>
                <RawFileView path={standaloneFile.path} kind="pdf" />
              </div>
            )}
          </div>
        )}
      </main>
      <FilePanelSlot />
      <SaveAsModal open={persistence.saveModalOpen} onClose={() => persistence.setSaveModalOpen(false)} onSave={persistence.handleSaveSubmit} />
      <PromptEditor open={promptEditorOpen} onClose={() => setPromptEditorOpen(false)} />
      {isActive && filePicker.filePickerOpen && filePicker.rootFiles.enabled && (
        <FileViewer
          files={(filePicker.rootFiles.files ?? []).map((f) => f.path)}
          onSelect={filePicker.handleFileSelect}
          onClose={() => filePicker.setFilePickerOpen(false)}
          placeholder="Search files…"
          emptyLabel={filePicker.rootFiles.files ? "No files" : "Loading…"}
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
      <TabPicker />
    </div>
  )
}
