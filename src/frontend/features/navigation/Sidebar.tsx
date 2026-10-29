"use client"

import { useMemo, useState } from "react"
import { motion } from "framer-motion"
import { Brain, Pencil } from "lucide-react"
import { PanelToggleButton } from "@/components/ui/PanelToggleButton"
import { SaveAsModal } from "@/components/ui/SaveAsModal"
import { ErrorText } from "@/components/ui/ErrorText"
import { FileTree } from "@/components/ui/file-tree/FileTree"
import { useFileTree, usePathTree } from "@/components/ui/file-tree/useFileTree"
import { useProject } from "@/contexts/ProjectContext"
import { useTab, type AppSurface, type WorkspaceItem } from "@/contexts/TabContext"
import { useUndoDelete } from "@/contexts/UndoDeleteContext"
import { useMemoryViewer } from "@/features/memory/MemoryViewerContext"
import { addFileRoot, cloneFileRoot } from "@/lib/backend/api"
import { cn } from "@/lib/utils"
import { ArchiveSelector } from "./ArchiveSelector"
import { useBranches } from "./BranchContext"
import { ChatBranchPreview } from "./ChatBranchPreview"
import { CreateDocumentPanel } from "./CreateDocumentPanel"
import { FileMountPicker } from "./FileMountPicker"
import { ProviderUsage } from "./ProviderUsage"
import { useSidebar } from "./SidebarContext"
import { TabList } from "./TabList"
import { useCodeMounts } from "./useCodeMounts"
import { useDeleteChat } from "./useDeleteChat"
import { useProjectActions } from "./useProjectActions"
import { useWorkspaceDocuments } from "./useWorkspaceDocuments"
import { codeItems, directoryEntries, projectItems } from "./workspaceTree"

interface SidebarProps {
  onFileSelect?: (path: string) => void
  onDiffSelect?: (path: string) => void
  onRootsChanged?: () => void
  onDocumentsChanged?: () => void
  onDocumentSelect?: (path: string, scope: string) => void
  onProjectSelect?: (slug: string) => void
  onProjectRenamed?: (oldSlug: string, newSlug: string) => void
  onNewChat?: (slug: string | null) => void
  activeCanvasPath?: string | null
  activeDocumentPath?: string | null
  onCanvasSelect?: (path: string, scope: string) => void
  onCanvasDeleted?: (path: string) => void
  onArchitectureSelect?: (slug: string) => void
  /** Show an omp-web session, new session or settings, or the factory board, in the current tab unless `inNewTab`. */
  onItemOpen?: (item: WorkspaceItem, inNewTab?: boolean) => void
  renderSettings?: () => React.ReactNode
  appMode: AppSurface
}

export function Sidebar(props: SidebarProps) {
  const { onNewChat, activeCanvasPath, renderSettings, appMode } = props
  const { collapsed, toggleCollapsed } = useSidebar()
  const { tab } = useTab()
  const branches = useBranches()
  const { projects, createProject } = useProject()
  const { openMemoryViewer } = useMemoryViewer()
  const { pending } = useUndoDelete()
  const deleteChat = useDeleteChat()
  const defaultExpanded = useMemo(() => [
    "shared", "shared:canvases",
    ...projects.flatMap((project) => ["chats", "canvases", "files"].map((category) => `project:${project.slug}:${category}`)),
  ], [projects])
  const controller = useFileTree({ storageKey: "expanded_workspace_projects", defaultExpanded })
  const fileMountController = usePathTree()
  const [fileMountPickerOpen, setFileMountPickerOpen] = useState(false)
  const projectActions = useProjectActions(props.onProjectRenamed)
  const documents = useWorkspaceDocuments({
    ...props, projects, moveHistory: branches.moveHistory, reloadKey: `${activeCanvasPath}\n${appMode}`,
  })
  const code = useCodeMounts(props)
  const { omp } = code

  const items = projectItems({
    ...props, ...branches, ...documents, ...projectActions, projects, pending, deleteChat, activeProjectSlug: tab.projectSlug,
  })
  const codeContext = { ...props, ...code, projects, pending }

  return <motion.aside
    initial={false} animate={{ width: collapsed ? 0 : 280 }} transition={{ duration: 0.25, ease: "easeInOut" }}
    className={cn("relative shrink-0 flex flex-col transition-[margin,background-color,box-shadow,border-color] duration-200",
      collapsed ? "border-0" : "m-3 mr-0 rounded-2xl border border-divider/50 bg-surface shadow-[var(--shadow-lg)]")}
  >
    {!collapsed && <>
      <div className="flex h-10 shrink-0 items-center justify-between pl-4 pr-2">
        <button onClick={openMemoryViewer} aria-label="Open memory"
          className="rounded-md p-1 text-ink-subtle hover:bg-surface-elevated hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted transition-colors">
          <Brain className="h-4 w-4" />
        </button>
        <PanelToggleButton side="left" open onClick={toggleCollapsed} openLabel="Open sidebar" closeLabel="Close sidebar" />
      </div>
      <nav aria-label="Workspace" className="min-h-0 flex-1 overflow-y-auto px-2 pb-3">
        <TabList />
        <div role="separator" className="my-4 border-t border-divider/50" />
        <FileTree label="Projects" items={items} controller={controller} loading={branches.historiesLoading}
          headerActions={<ArchiveSelector
            noun="projects"
            entries={projects.filter((project) => !pending(`project:${project.slug}`)).map((project) => ({ key: project.slug, name: project.name, archived: Boolean(project.archived) }))}
            onRestore={(slug) => projectActions.changeArchive(slug, false)}
          />}
          create={{ label: "New project", placeholder: "Project name", onSubmit: async (name) => {
            const slug = await createProject(name)
            controller.expand(slug)
            onNewChat?.(slug)
          } }}
          renderBeforeItem={(item) => item.data ? <ChatBranchPreview file={item.data} /> : null}
          onMoveToRoot={(path) => documents.move(path, "")}
        />
        <ErrorText className="px-2 py-1">{projectActions.archiveError}</ErrorText>
        <div role="separator" className="my-4 border-t border-divider/50" />
        <FileTree label="Code" items={codeItems(codeContext)} controller={fileMountController}
          headerActions={<ArchiveSelector noun="directories" entries={directoryEntries(codeContext)} onRestore={(path) => code.setArchived(path, false)} />}
          labelAction={omp.available ? { label: "Open omp settings", onClick: omp.openSettings } : undefined}
          create={code.hostAccess ? { label: "Add file mount", onClick: () => setFileMountPickerOpen(true) } : undefined}
        />
        <ErrorText className="px-2 py-1">{omp.error}</ErrorText>
      </nav>
      <ProviderUsage />
      {renderSettings && <div className="shrink-0 p-2">{renderSettings()}</div>}
    </>}
    <FileMountPicker open={fileMountPickerOpen} onClose={() => setFileMountPickerOpen(false)}
      onConfirm={(path) => code.mountThen(() => addFileRoot(path))} onClone={(url) => code.mountThen(() => cloneFileRoot(url))} />
    <SaveAsModal {...projectActions.renameDialog} title="Rename project" icon={Pencil} submitLabel="Rename" />
    <CreateDocumentPanel open={documents.createTarget != null} onClose={() => documents.setCreateTarget(null)}
      onCreate={(name) => void documents.createDocument(name)} />
  </motion.aside>
}
