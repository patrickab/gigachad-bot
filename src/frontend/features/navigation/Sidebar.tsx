"use client"

import { useMemo, useState } from "react"
import { Brain, FolderPlus } from "lucide-react"
import { PanelToggleButton } from "@/components/ui/PanelToggleButton"
import { ErrorText } from "@/components/ui/ErrorText"
import { FileTree } from "@/components/ui/file-tree/FileTree"
import { useFileTree, usePathTree } from "@/components/ui/file-tree/useFileTree"
import { useInlineRename } from "@/components/ui/file-tree/useInlineRename"
import { useProject } from "@/contexts/ProjectContext"
import { useTab, type AppSurface, type WorkspaceItem } from "@/contexts/TabContext"
import { useUndoDelete } from "@/contexts/UndoDeleteContext"
import { useMemoryViewer } from "@/features/memory/MemoryViewerContext"
import { LearnSection, useLearnSidebar } from "@/features/learn/LearnSidebar"
import { addFileRoot, cloneFileRoot, createTexProjectFile } from "@/lib/backend/api"
import { refreshDocumentLists, refreshRootTree } from "@/lib/backend/resources"
import { SidePanel, SidePanelScroll } from "@/components/ui/SidePanel"
import { ArchiveSelector } from "./ArchiveSelector"
import { useBranches } from "./BranchContext"
import { ChatBranchPreview } from "./ChatBranchPreview"
import { CreateDocumentPanel } from "./CreateDocumentPanel"
import { FileMountPicker } from "./FileMountPicker"
import { ProviderUsage } from "./ProviderUsage"
import { useSidebar } from "./SidebarContext"
import { useCodeMounts } from "./useCodeMounts"
import { useDeleteChat } from "./useDeleteChat"
import { useFileRename } from "./useFileRename"
import { useProjectActions } from "./useProjectActions"
import { useWorkspaceDocuments } from "./useWorkspaceDocuments"
import { directoryEntries, owningProject, projectItems, workspaceItems } from "./workspaceTree"

/** `name` as a LaTeX project file name: `.tex` appended unless it names its own extension. */
const texFileName = (name: string) => (/\.[^./]+$/.test(name) ? name : `${name}.tex`)

interface SidebarProps {
  /** Open the file `path` of a folder on disk, `scope` is the project it belongs to, null for none. */
  onFileSelect?: (path: string, scope: string | null) => void
  onDiffSelect?: (path: string) => void
  onDocumentSelect?: (path: string, scope: string) => void
  /** Opens `.tex .bib .sty .cls` and `.md` files in the workbench, absent where it is unavailable. */
  onWorkbenchSelect?: (path: string, scope: string) => void
  onProjectSelect?: (slug: string) => void
  onProjectRenamed?: (oldSlug: string, newSlug: string) => void
  /** A file was renamed: tabs showing `oldPath` follow it to `newPath`. */
  onDocumentRenamed?: (oldPath: string, newPath: string) => void
  onNewChat?: (slug: string | null) => void
  activeCanvasPath?: string | null
  activeDocumentPath?: string | null
  onCanvasSelect?: (path: string, scope: string) => void
  onCanvasDeleted?: (path: string) => void
  onArchitectureSelect?: (slug: string) => void
  /** Show a Pi session or new session, a learn view, or the factory board, in the current tab unless `inNewTab`. */
  onItemOpen?: (item: WorkspaceItem, inNewTab?: boolean) => void
  renderSettings?: () => React.ReactNode
  appMode: AppSurface
}

export function Sidebar(props: SidebarProps) {
  const { onNewChat, renderSettings } = props
  const { collapsed, toggleCollapsed, width, setWidth } = useSidebar()
  const { tab } = useTab()
  const branches = useBranches()
  const { projects, createProject } = useProject()
  const { openMemoryViewer } = useMemoryViewer()
  const { pending } = useUndoDelete()
  const deleteChat = useDeleteChat()
  const defaultExpanded = useMemo(() => [
    "shared", "shared:chats",
    ...projects.map((project) => `project:${project.slug}:chats`),
  ], [projects])
  const stored = useFileTree({ storageKey: "expanded_workspace_projects", defaultExpanded })
  // Folders on disk (ids are absolute paths) reopen collapsed, the rest keeps its saved expansion
  const paths = usePathTree()
  const pick = (id: string) => id.startsWith("/") ? paths : stored
  const controller = {
    isExpanded: (id: string) => pick(id).isExpanded(id),
    toggleBranch: (id: string) => pick(id).toggleBranch(id),
    expand: (id: string) => pick(id).expand(id),
    set: (changes: Record<string, boolean>) => {
      const entries = Object.entries(changes)
      stored.set(Object.fromEntries(entries.filter(([id]) => !id.startsWith("/"))))
      paths.set(Object.fromEntries(entries.filter(([id]) => id.startsWith("/"))))
    },
  }
  const [fileMountPickerOpen, setFileMountPickerOpen] = useState(false)
  const projectActions = useProjectActions(props.onProjectRenamed)
  const { renameOf } = useInlineRename()
  const documents = useWorkspaceDocuments({ ...props, projects, moveHistory: branches.moveHistory })
  const code = useCodeMounts(props)
  const fileRename = useFileRename({
    // Opening a stored file in the workbench moves it to disk, so the row may hold the key it had before
    currentPath: ({ path, name, scope }) => {
      const shown = documents.documents[scope] ?? []
      return shown.some((doc) => doc.path === path) ? path : shown.find((doc) => doc.name === name)?.path ?? path
    },
    onRenamed: async (oldPath, newPath) => {
      props.onDocumentRenamed?.(oldPath, newPath)
      await refreshDocumentLists()
    },
  })
  const selectFile = (path: string) => props.onFileSelect?.(path, owningProject(path, documents.documents, projects, code.mounts))
  const folderRows = {
    onFileSelect: selectFile,
    moveToFolder: async (path: string, folder: string) => {
      await documents.moveToFolder(path, folder)
      await refreshRootTree()
    },
    createLatexFile: async (folder: string, name: string) => {
      const path = await createTexProjectFile(folder, texFileName(name))
      await Promise.all([refreshDocumentLists(), refreshRootTree()])
      selectFile(path)
    },
  }

  const learn = useLearnSidebar({ open: props.onItemOpen ?? (() => {}), projects, sidebar: { collapsed, toggle: toggleCollapsed } })

  const projectTree = projectItems({
    ...props, ...branches, ...documents, ...projectActions, ...fileRename, ...folderRows, renameOf, projects, pending, deleteChat, activeProjectSlug: tab.projectSlug,
  })
  // Host logins see each project's learn sessions in its group.
  const items = code.hostAccess ? projectTree.map((root) => {
    const project = projects.find((candidate) => candidate.slug === root.id)
    return project ? learn.withProjectLearn(root, project) : root
  }) : projectTree
  const codeContext = { ...props, ...code, ...fileRename, ...folderRows, renameOf, projects, pending }
  const archiveEntries = [
    ...projects.filter((project) => !pending(`project:${project.slug}`)).map((project) => ({ key: `project:${project.slug}`, name: project.name, archived: Boolean(project.archived) })),
    ...directoryEntries(codeContext).map((entry) => ({ ...entry, key: `directory:${entry.key}` })),
  ]
  const restore = (key: string) => key.startsWith("project:")
    ? projectActions.changeArchive(key.slice("project:".length), false)
    : code.setArchived(key.slice("directory:".length), false)

  return <>
    <SidePanel side="left" open={!collapsed} width={width} onWidthChange={setWidth} label="Sidebar" className="z-10">
      <div className="flex h-10 shrink-0 items-center justify-between pl-4 pr-2">
        <button onClick={openMemoryViewer} aria-label="Open memory"
          className="rounded-md p-1 text-ink-subtle hover:bg-surface-elevated hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted transition-colors">
          <Brain className="h-4 w-4" />
        </button>
        <PanelToggleButton side="left" open onClick={toggleCollapsed} openLabel="Open sidebar" closeLabel="Close sidebar" />
      </div>
      <nav aria-label="Workspace" className="flex min-h-0 flex-1 flex-col">
        <SidePanelScroll className="px-2 pb-3">
          <FileTree label="Projects" items={workspaceItems(items, codeContext)} controller={controller} loading={branches.historiesLoading}
            headerActions={<>
              <ArchiveSelector noun="projects" entries={archiveEntries} onRestore={restore} />
              {code.hostAccess && <button aria-label="Add file mount" onClick={() => setFileMountPickerOpen(true)}
                className="rounded p-1 text-ink-faint opacity-0 hover:text-ink group-hover/row:opacity-100 focus-visible:opacity-100 focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted">
                <FolderPlus className="h-3 w-3" />
              </button>}
            </>}
            create={{ label: "New project", placeholder: "Project name", onSubmit: async (name) => {
              const slug = await createProject(name)
              controller.expand(slug)
              onNewChat?.(slug)
            } }}
            renderBeforeItem={(item) => item.data ? <ChatBranchPreview file={item.data} /> : null}
            onMoveToRoot={(path) => documents.move(path, "")}
          />
          <ErrorText className="px-2 py-1">{projectActions.archiveError}</ErrorText>
          <ErrorText className="px-2 py-1">{documents.fileError}</ErrorText>
          <ErrorText className="px-2 py-1">{code.sessions.error}</ErrorText>
          {code.hostAccess && <>
            <div role="separator" className="my-3 border-t border-divider/50" />
            <LearnSection section={learn.section} />
          </>}
        </SidePanelScroll>
      </nav>
      <ProviderUsage />
      {renderSettings && <div className="shrink-0 p-2">{renderSettings()}</div>}
    </SidePanel>
    <FileMountPicker open={fileMountPickerOpen} onClose={() => setFileMountPickerOpen(false)}
      onConfirm={(path) => code.mountThen(() => addFileRoot(path))} onClone={(url) => code.mountThen(() => cloneFileRoot(url))} />
    <CreateDocumentPanel open={documents.createTarget != null} onClose={() => documents.setCreateTarget(null)}
      onCreate={(name) => void documents.createDocument(name)} />
  </>
}
