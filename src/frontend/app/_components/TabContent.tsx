"use client"

import { memo, useCallback, useEffect, useState, type RefCallback } from "react"
import { anchorSceneFallback } from "deepfocus-ambient"
import { ErrorBoundary } from "@/components/ui/ErrorBoundary"
import { SaveAsModal } from "@/components/ui/SaveAsModal"
import { ChatContainer, type ChatContainerProps, type ChatReveal, type ChatRevealHit } from "@/features/chat/ChatContainer"
import type { ChatStore } from "@/features/chat/chatStore"
import { newChatDraftKey } from "@/features/chat/composer/useComposerDraft"
import { useContentAreaRef } from "@/components/ui/ContentArea"
import { FileViewer } from "@/features/editor/FileViewer"
import { DocumentPicker } from "@/features/editor/DocumentPicker"
import { DocumentEditor } from "@/features/editor/DocumentEditor"
import { FilePanelProvider, FilePanelSlot } from "@/features/editor/FilePanel"
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
import { useStore } from "@/hooks/useStore"
import { useTabChat } from "./useTabChat"
import { WorkspaceSurface, type ModelProps } from "./WorkspaceSurface"
// The tool renderers a feature's rows need from another feature, loaded where both are known: a chat memory call
// renders as Pi's memory row (`features/pi/MemoryTool.tsx`), without the chat importing Pi.
import "@/features/pi/MemoryTool"

interface TabContentProps {
  onOpenChat: (historyFile: string, qaIndex?: number, hit?: ChatRevealHit) => void
  onOpenItem: (item: WorkspaceItem, inNewTab?: boolean) => void
  reveal: ChatReveal
}

/**
 * Keep `--under-left` and `--under-right` on the tab's main area: the px between it and the app's left edge (the sidebar's share with its gutter, left of the tab stack)
 * and between it and the tab's right edge (the file panel's). A surface running under the sidebars (`runsUnderSidebars`) extends its body by them, the cards float over it.
 */
export function measureSidebarInsets(main: HTMLElement): (() => void) | undefined {
  if (typeof ResizeObserver === "undefined") return
  const tab = main.parentElement!
  const stack = main.closest<HTMLElement>("[data-tab-stack]")
  const update = () => {
    main.style.setProperty("--under-left", `${(stack?.offsetLeft ?? 0) + main.offsetLeft}px`)
    main.style.setProperty("--under-right", `${tab.clientWidth - main.offsetLeft - main.offsetWidth}px`)
  }
  const observer = new ResizeObserver(update)
  observer.observe(main)
  observer.observe(tab)
  return () => observer.disconnect()
}

/**
 * One tab's workspace: the chat (or canvas) with its top bar, file panel and keyboard shortcuts.
 * Memoized on this hook-free wrapper (see `AppSidebar`), so a tab switch or a history refresh re-renders only the tabs whose context changed.
 */
export const TabContent = memo(function TabContent(props: TabContentProps) {
  return <FilePanelProvider><ErrorBoundary><TopBarSlotProvider><TabWorkspace {...props} /></TopBarSlotProvider></ErrorBoundary></FilePanelProvider>
})

/** The tab's chat on its store: the only subscriber of the messages, so a token frame renders nothing above it. */
export function TabChat({ chat, ...props }: { chat: ChatStore } & Omit<ChatContainerProps, "messages" | "isStreaming" | "onCancel">) {
  const messages = useStore(chat, (state) => state.messages)
  const isStreaming = useStore(chat, (state) => state.isStreaming)
  return <ChatContainer {...props} messages={messages} isStreaming={isStreaming} onCancel={chat.cancel} />
}

/** The tab's chat surfaces and the top bar whose slots they render their controls into. */
function TabWorkspace({ onOpenChat, onOpenItem, reveal }: TabContentProps) {
  const { tab, isActive, setModeLabel } = useTab()
  const { projectSlug, config } = tab
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
  const [chatColumn, setChatColumn] = useState<HTMLDivElement | null>(null)
  const canvas = useCanvasView()
  const {
    chat, hasMessages, isStreaming, chatConfig, chatId, branchMessageIdx, historyLoading, extracting, chatInputRef, reasoningLevels, reasoningEffort,
    filePicker, docs, turns, attachmentActions, attachMenuItems, onEditPair, onBranch, saveAs,
  } = useTabChat({ onOpenChat, onOpenItem })
  const { models } = chatConfig
  const { standaloneFile } = filePicker
  const modelProps: ModelProps = { models, loadError: chatConfig.modelLoadError, onProvidersChange: chatConfig.saveModelProviders, onTabOrderChange: chatConfig.saveModelTabOrder, onDefaultsChange: chatConfig.saveModelDefaults, placement: "below" }

  // A folder picked in an empty chat makes its first message start a Pi session there, the Pi tab sends it once the session is ready.
  // The picker shows only in an empty chat, so a folder left over from one never redirects another chat's message
  const [folder, setFolder] = useState<string | null>(null)
  const pickFolder = (picked: string | null) => setFolder(picked === null ? null : picked.replace(/\/+$/, "") || "/")
  const [launch, setLaunch] = useState<PiLaunch | null>(null)
  const isEmptyChat = !hasMessages && !historyLoading && !isStreaming
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
    <TabChat
      chat={chat}
      chatId={chatId}
      historyFile={tab.historyFile ?? null}
      draftKey={tab.historyFile ? `chat:${projectSlug ?? ""}:${tab.historyFile}` : newChatDraftKey(tab.id)}
      onChatColumnChange={setChatColumn}
      historyLoading={historyLoading}
      onDeletePair={turns.handleDeletePair}
      onEditPair={onEditPair}
      extracting={extracting}
      onSend={handleSend}
      onBranch={onBranch}
      reveal={reveal}
      branchMessageIdx={branchMessageIdx}
      chatInputRef={chatInputRef}
      slug={projectSlug}
      attachmentActions={attachmentActions}
      attachMenuItems={attachMenuItems}
      composerLeading={isEmptyChat ? <PiWorkspacePicker folder={folder} onChange={pickFolder} onOpenSession={(session, repo) => onOpenItem(piSessionItem(session, repo ?? newPiItem(session.cwd, projectSlug).pi!.repo, projectSlug))} /> : undefined}
      composerNotice={turns.turnError || docs.architectureError || undefined}
    />
  )

  return (
    // Unclipped: a surface running under the sidebar reaches left of the tab stack, the app's chrome clips
    <div className="relative flex h-full">
      {/* No paper of its own: the ocean backdrop, or a shared frame laid under this tab, shows through */}
      <main ref={contentAreaRef} className="flex-1 min-w-0 flex flex-col relative">
        <WorkspaceSurface
          canvas={canvas}
          modelProps={modelProps}
          reasoningLevels={reasoningLevels}
          reasoningEffort={reasoningEffort}
          chat={chat}
          chatColumn={chatColumn}
          chatBody={chatBody}
          launch={launch}
          piStart={piStart}
          onLaunched={() => setLaunch(null)}
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
      <SaveAsModal {...saveAs} />
      {isActive && filePicker.filePickerOpen && filePicker.rootFiles.enabled && (
        <FileViewer
          files={(filePicker.rootFiles.files ?? []).map((f) => f.path)}
          onSelect={filePicker.handleFileSelect}
          onClose={() => filePicker.setFilePickerOpen(false)}
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
    </div>
  )
}
