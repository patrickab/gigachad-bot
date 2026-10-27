"use client"

import { useState } from "react"
import { PanelLeft, PanelLeftClose } from "lucide-react"
import { Sidebar } from "@/features/navigation/Sidebar"
import { useSidebar } from "@/features/navigation/SidebarContext"
import { ChatContainer, type ChatFocus } from "@/features/chat/ChatContainer"
import { useContentAreaRef } from "@/components/ui/ContentArea"
import { SettingsMenu } from "@/features/settings/SettingsMenu"
import { PromptEditor } from "@/features/settings/PromptEditor"
import { SaveAsModal } from "@/components/ui/SaveAsModal"
import { FileViewer } from "@/features/editor/FileViewer"
import { DocumentPicker } from "@/features/editor/DocumentPicker"
import { DocumentEditor } from "@/features/editor/DocumentEditor"
import { FilePanelSlot } from "@/features/editor/FilePanel"
import { TopBarSlotProvider } from "@/features/navigation/TopBarSlots"
import { RawFileView } from "@/features/editor/RawFileView"
import { useEnabledTools } from "@/features/chat/EnabledToolsContext"
import { itemOf, useTab, type WorkspaceItem } from "@/contexts/TabContext"
import { useCanvasView } from "@/features/editor/canvas/useCanvasView"
import { writeRootFile } from "@/lib/backend/api"
import { MemoryCommandLayer } from "./MemoryCommandLayer"
import { useTabChat } from "./useTabChat"
import { WorkspaceSurface, type ModelProps } from "./WorkspaceSurface"

interface TabContentProps {
  onOpenChat: (historyFile: string, qaIndex?: number) => void
  onOpenProject: (slug: string) => void
  onProjectRenamed: (oldSlug: string, newSlug: string) => void
  onNewChat: (slug: string | null) => void
  onOpenItem: (item: WorkspaceItem, inNewTab?: boolean) => void
  focus: ChatFocus
}

/** One tab's workspace: the chat (or canvas) with its sidebar, top bar, modals and keyboard shortcuts. */
export function TabContent(props: TabContentProps) {
  return <TopBarSlotProvider><TabWorkspace {...props} /></TopBarSlotProvider>
}

/** The tab's sidebar, surfaces and the top bar whose slots they render their controls into. */
function TabWorkspace({ onOpenChat, onOpenProject, onProjectRenamed, onNewChat, onOpenItem, focus }: TabContentProps) {
  const { tab, isActive, setModeLabel, updateConfig } = useTab()
  const { projectSlug, config, appMode } = tab
  const { researchEnabled } = useEnabledTools()
  const { collapsed: sidebarCollapsed, toggleCollapsed: toggleSidebar } = useSidebar()
  const contentAreaRef = useContentAreaRef()
  const [promptEditorOpen, setPromptEditorOpen] = useState(false)
  const [chatColumn, setChatColumn] = useState<HTMLDivElement | null>(null)
  const canvas = useCanvasView()
  const {
    chat, chatId, branchMessageIdx, extracting, chatInputRef, reasoningLevels, reasoningEffort,
    filePicker, docs, persistence, memory, menuItems, runMenuCommand, turns, attachmentActions, attachMenuItems, onBranch,
  } = useTabChat({ focus, onOpenChat })
  const { models, prompts, setPrompts } = chat
  const { standaloneFile } = filePicker
  const modelProps: ModelProps = { models, loadError: chat.modelLoadError, onProvidersChange: chat.saveModelProviders, onTabOrderChange: chat.saveModelTabOrder, onDefaultsChange: chat.saveModelDefaults, placement: "below" }

  const chatBody = (
    <ChatContainer
      chatId={chatId}
      onChatColumnChange={setChatColumn}
      messages={chat.messages}
      isStreaming={chat.isStreaming}
      onDeletePair={turns.handleDeletePair}
      extracting={extracting}
      onSend={turns.handleSend}
      onCancel={chat.cancel}
      onRegenerate={turns.handleRegenerate}
      onBranch={onBranch}
      focus={focus}
      branchMessageIdx={branchMessageIdx}
      chatInputRef={chatInputRef}
      slug={projectSlug}
      attachmentActions={attachmentActions}
      attachMenuItems={attachMenuItems}
      composerNotice={docs.architectureError || undefined}
    />
  )

  return (
    <div className="relative flex h-full overflow-hidden">
      <Sidebar
        onFileSelect={(path) => onOpenItem({ kind: "document", path, projectSlug: null, rootFile: true })}
        onDiffSelect={(path) => onOpenItem({ kind: "diff", path, projectSlug: null })}
        onRootsChanged={filePicker.refreshFileList}
        onDocumentsChanged={docs.refreshDocuments}
        {...canvas.sidebarProps}
        onProjectSelect={onOpenProject}
        onProjectRenamed={onProjectRenamed}
        onNewChat={onNewChat}
        onDocumentSelect={(path, scope) => onOpenItem({ kind: "document", path, projectSlug: scope || null })}
        activeDocumentPath={appMode === "document" ? itemOf(tab, "document")?.path : null}
        onCanvasSelect={(path, scope) => onOpenItem({ kind: "canvas", path, projectSlug: scope || null })}
        onArchitectureSelect={(slug) => onOpenItem({ kind: "architecture", path: `graph/${slug}/`, projectSlug: slug })}
        onItemOpen={onOpenItem}
        renderSettings={() => (
          <SettingsMenu prompts={prompts} config={config} onConfigChange={updateConfig} onEditPrompts={() => setPromptEditorOpen(true)} models={models} researchEnabled={researchEnabled} />
        )}
        appMode={appMode}
      />
      <main ref={contentAreaRef} className="flex-1 min-w-0 flex flex-col relative bg-paper">
        <WorkspaceSurface
          sidebarToggle={
            <button
              aria-label={sidebarCollapsed ? "Open sidebar" : "Close sidebar"}
              onClick={toggleSidebar}
              className="shrink-0 rounded p-1 text-ink-subtle hover:text-ink hover:bg-hover transition-colors"
            >
              {sidebarCollapsed ? <PanelLeft className="h-4 w-4" /> : <PanelLeftClose className="h-4 w-4" />}
            </button>
          }
          canvas={canvas}
          modelProps={modelProps}
          reasoningLevels={reasoningLevels}
          reasoningEffort={reasoningEffort}
          totalUsage={chat.totalUsage}
          chatColumn={chatColumn}
          chatBody={chatBody}
          refreshDocuments={docs.refreshDocuments}
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
                  onNavigate={(path) => filePicker.setStandaloneFile({ kind: "edit", path })}
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
      <PromptEditor open={promptEditorOpen} onClose={() => setPromptEditorOpen(false)} onPromptsChanged={setPrompts} />
      {isActive && filePicker.filePickerOpen && filePicker.rootFiles.enabled && (
        <FileViewer
          files={filePicker.rootFiles.files.map((f) => f.path)}
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
