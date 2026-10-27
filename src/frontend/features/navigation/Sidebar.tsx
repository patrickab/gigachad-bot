"use client"

import { useCallback, useEffect, useMemo, useState } from "react"
import { motion } from "framer-motion"
import { Archive, ArchiveRestore, Brain, Factory, FilePlus, GitCompare, Pencil } from "lucide-react"
import { ArchitectureIcon } from "@/components/ui/ArchitectureIcon"
import { PopoverSelector } from "@/components/ui/PopoverSelector"
import { SaveAsModal } from "@/components/ui/SaveAsModal"
import { CreateDocumentPanel } from "./CreateDocumentPanel"
import { ProviderUsage } from "./ProviderUsage"
import { ErrorText } from "@/components/ui/ErrorText"
import { cn, errorMessage } from "@/lib/utils"
import { useAsyncAction } from "@/hooks/useAsyncAction"
import { FileTree, type FileTreeItem } from "@/components/ui/file-tree/FileTree"
import { useFileTree, usePathTree } from "@/components/ui/file-tree/useFileTree"
import { useMemoryViewer } from "@/features/memory/MemoryViewerContext"
import { useProject } from "@/contexts/ProjectContext"
import { useBranches } from "@/features/navigation/BranchContext"
import { useDeleteChat } from "@/features/navigation/useDeleteChat"
import { useSidebar } from "@/features/navigation/SidebarContext"
import { useUndoDelete } from "@/contexts/UndoDeleteContext"
import { useStartupRead } from "@/contexts/StartupReadContext"
import type { BranchMeta, FileTreeNode, ProjectListItem, ProjectDocument } from "@/lib/domain/types"
import { ChatBranchPreview } from "./ChatBranchPreview"
import { FileMountPicker } from "./FileMountPicker"
import { useWorktrees } from "./useWorktrees"
import { useArchivedDirectories } from "./useArchivedDirectories"
import { addRootFilesToProject, addFileRoot, cloneFileRoot, createArchitectureSource, fileRootTree, removeFileRoot, listNotes, listProjectDocuments, moveDocument, removeDocument, writeDocument } from "@/lib/backend/api"
import { useTab, useTabActive, type AppSurface, type WorkspaceItem } from "@/contexts/TabContext"
import { useHostAccess } from "@/contexts/SettingsContext"
import { factoryItem } from "@/features/factory/items"
import { useFactoryRepos } from "@/features/factory/useFactoryRepos"
import { projectMount } from "@/features/omp/sessions"
import { useOmpSidebar } from "@/features/omp/useOmpSidebar"
import { isArchitecturePath } from "@/lib/domain/architecturePaths"
import { baseName } from "@/lib/paths"

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

export function Sidebar({
  onFileSelect, onDiffSelect, onRootsChanged, onDocumentsChanged, onDocumentSelect, onProjectSelect, onProjectRenamed, onNewChat,
  activeCanvasPath, onCanvasSelect, onCanvasDeleted, onArchitectureSelect, onItemOpen,
  activeDocumentPath, renderSettings, appMode,
}: SidebarProps) {
  const { collapsed } = useSidebar()
  const hostAccess = useHostAccess()
  const isActive = useTabActive()
  const { tab } = useTab()
  const { branchMeta, visibleRootFiles, historiesLoading, moveHistory, onFileClick, activeFile, activeQaIndex, refreshAll } = useBranches()
  const { projects, createProject, deleteProject, setProjectArchived, renameProject } = useProject()
  const { openMemoryViewer } = useMemoryViewer()
  const { schedule, pending } = useUndoDelete()
  const deleteChat = useDeleteChat()
  const defaultExpanded = useMemo(() => [
    "shared", "shared:canvases",
    ...projects.flatMap((project) => ["chats", "canvases", "files"].map((category) => `project:${project.slug}:${category}`)),
  ], [projects])
  const controller = useFileTree({ storageKey: "expanded_workspace_projects", defaultExpanded })
  const fileMountController = usePathTree()
  const [documents, setDocuments] = useState<Record<string, ProjectDocument[]>>({})
  const [notes, setNotes] = useState<ProjectDocument[]>([])
  const [rootTreeData, setRootTreeData] = useState<FileTreeNode[]>([])
  const [createDocumentSlug, setCreateDocumentSlug] = useState<string | null>(null)
  const [fileMountPickerOpen, setFileMountPickerOpen] = useState(false)
  const { archived: archivedDirectories, setArchived: setDirectoryArchived } = useArchivedDirectories()
  const { run: archiveProject, error: projectArchiveError } = useAsyncAction(setProjectArchived, "Could not update project archive")
  const [archivingProject, setArchivingProject] = useState<string | null>(null)
  const [renameTarget, setRenameTarget] = useState<ProjectListItem | null>(null)
  const { run: rename, pending: renamePending, error: renameError, setError: setRenameError } = useAsyncAction(async (target: ProjectListItem, name: string) => {
    const slug = await renameProject(target.slug, name)
    if (slug !== target.slug) {
      onProjectRenamed?.(target.slug, slug)
      void refreshAll().catch(() => {})
    }
    setRenameTarget(null)
  }, "Could not rename project")
  const submitRename = (name: string) => {
    if (renameTarget && !renamePending) void rename(renameTarget, name)
  }
  const changeProjectArchive = async (slug: string, archived: boolean) => {
    setArchivingProject(slug)
    await archiveProject(slug, archived)
    setArchivingProject(null)
  }

  const loadProjectDocuments = useCallback(async (signal?: AbortSignal) => {
    const entries = await Promise.all(projects.map(async (project) =>
      [project.slug, await listProjectDocuments(project.slug, signal)] as const))
    return Object.fromEntries(entries)
  }, [projects])

  const refreshDocuments = useCallback(async () => {
    const [loose, entries] = await Promise.all([listNotes(), loadProjectDocuments()])
    setNotes(loose)
    setDocuments(entries)
  }, [loadProjectDocuments])

  /** Create `name` in `scope`, refresh the lists and hand the new path to `open`. */
  const createAndOpen = async (scope: string, name: string, open: (path: string) => void) => {
    const { path } = await writeDocument(scope, name)
    await refreshDocuments()
    open(path)
  }

  const { retry: reloadNotes } = useStartupRead("Sidebar notes", listNotes, setNotes, isActive)
  const { retry: reloadDocuments } = useStartupRead("Sidebar project documents", loadProjectDocuments, setDocuments, isActive)
  useEffect(() => {
    if (!isActive) return
    reloadNotes()
    reloadDocuments()
  }, [activeCanvasPath, appMode, isActive, reloadNotes, reloadDocuments])

  const mounts = rootTreeData.map(mountOf)
  const openFactoryRepo = useFactoryRepos(hostAccess)
  const omp = useOmpSidebar({
    onOpen: onItemOpen,
    workspaceActions: (directory, label, reportError) => !hostAccess ? [] : [{
      label: `Open factory for ${label}`,
      icon: Factory,
      onClick: async () => {
        reportError("")
        try {
          onItemOpen?.(factoryItem((await openFactoryRepo(directory)).name))
        } catch (cause) {
          reportError(errorMessage(cause, "Could not open the factory"))
        }
      },
    }],
  })
  const worktrees = useWorktrees(mounts)
  const { reload: reloadWorktrees } = worktrees
  const activeWorktreesKey = worktrees.activeWorktrees.join("\n")

  const worktreeKeys = useMemo(() => activeWorktreesKey.split("\n").filter(Boolean), [activeWorktreesKey])
  const readRootTree = useCallback((signal?: AbortSignal) => fileRootTree(worktreeKeys, signal), [worktreeKeys])
  const loadRootTree = useCallback(() => readRootTree().then((result) => setRootTreeData(result.tree)), [readRootTree])
  useStartupRead("Sidebar file mounts", readRootTree, (result) => setRootTreeData(result.tree), isActive)
  const mountThen = async (add: () => Promise<unknown>) => {
    await add()
    await loadRootTree()
    onRootsChanged?.()
  }

  useEffect(() => {
    if (isActive) reloadWorktrees()
  }, [isActive, reloadWorktrees])

  const moveToProject = async (path: string, scope: string) => {
    if (isArchitecturePath(path)) return
    if (path.endsWith(".canvas")) {
      const from = Object.entries(documents).find(([, docs]) => docs.some((doc) => doc.path === path))?.[0] ?? ""
      if (from !== scope) {
        await moveDocument(path, from, scope)
        await refreshDocuments()
        onCanvasDeleted?.(path)
      }
    } else {
      await moveHistory(path, scope)
    }
  }

  const deleteDocument = (path: string, scope: string) => {
    schedule(`document:${scope}:${path}`, baseName(path), async () => {
      await removeDocument(scope, path)
      await refreshDocuments()
      onDocumentsChanged?.()
      onCanvasDeleted?.(path)
    })
  }

  const importFiles = async (scope: string, paths: string[]) => {
    await addRootFilesToProject(scope, paths)
    await refreshDocuments()
    onDocumentsChanged?.()
  }

  const buildRootFiles = (nodes: FileTreeNode[]): FileTreeItem<string>[] => nodes.map((node) => ({
    id: node.path,
    label: node.name,
    type: node.type === "file" ? "element" : "folder",
    multiSelect: true,
    isActive: node.path === activeDocumentPath,
    onSelect: node.type === "file" ? () => onFileSelect?.(node.path) : undefined,
    children: node.children ? buildRootFiles(node.children) : undefined,
  }))

  /** Mounts keep their registered path as id, so switching worktrees keeps them expanded. */
  const buildMounts = (nodes: FileTreeNode[]): FileTreeItem<string>[] => nodes
    .filter((node) => !pending(`root:${mountOf(node)}`) && !archivedDirectories.has(mountOf(node)))
    .map((node) => {
      const mount = mountOf(node)
      const folder = worktrees.folder(mount)
      const code = omp.directoryCode(mount, node.name, projects.find((project) => (project.linked_root ?? projectMount(project.slug, mounts)) === mount)?.slug ?? null)
      return {
        id: mount,
        label: node.name,
        type: "folder",
        // Dragging a mount adds its path, only unambiguous while it shows the clone itself.
        multiSelect: node.path === mount,
        actions: [
          { label: `Archive ${node.name}`, icon: Archive, onClick: () => setDirectoryArchived(mount, true) },
          { label: `Open diff for ${node.name}`, icon: GitCompare, onClick: () => onDiffSelect?.(node.path) },
          ...omp.directoryActions(mount, node.name),
        ],
        onDelete: () => schedule(`root:${mount}`, node.name, async () => {
          await removeFileRoot(mount)
          await loadRootTree()
          onRootsChanged?.()
        }),
        children: [...generatedRows([folder, code]), ...buildRootFiles(node.children ?? [])],
      }
    })

  /** Omp workspaces outside every mount: a directory row holding only its sessions. */
  const unmounted = omp.unmountedWorkspaces(mounts)
  const buildUnmounted = (): FileTreeItem<string>[] => unmounted
    .filter((workspace) => !archivedDirectories.has(workspace.path))
    .map((workspace) => {
      const code = omp.directoryCode(workspace.path, workspace.label, null, workspace.sessions)
      return {
        id: workspace.path,
        label: workspace.label,
        type: "folder",
        actions: [
          { label: `Archive ${workspace.label}`, icon: Archive, onClick: () => setDirectoryArchived(workspace.path, true) },
          ...omp.directoryActions(workspace.path, workspace.label),
        ],
        children: code ? [code] : [],
      }
    })

  const hasArchitecture = (slug: string) => (documents[slug] ?? []).some((doc) => isArchitecturePath(doc.path))

  const openArchitecture = async (slug: string) => {
    if (!hasArchitecture(slug)) {
      await createArchitectureSource(slug, "system.c4")
      await refreshDocuments()
    }
    onArchitectureSelect?.(slug)
  }

  /** Create a document in `slug` and open it: architecture sources in the architecture window, canvases as canvas, the rest in the editor. */
  const createDocument = async (slug: string, name: string) => {
    try {
      let open: () => void
      if (name.endsWith(".c4")) {
        await createArchitectureSource(slug, name)
        open = () => onArchitectureSelect?.(slug)
      } else {
        const { path } = await writeDocument(slug, name)
        open = path.endsWith(".canvas") ? () => onCanvasSelect?.(path, slug) : () => onDocumentSelect?.(path, slug)
      }
      setCreateDocumentSlug(null)
      await refreshDocuments()
      onDocumentsChanged?.()
      open()
    } catch { /* the panel stays open to retry */ }
  }

  const buildCategories = (project: ProjectListItem | null): FileTreeItem<string>[] => {
    const scope = project?.slug ?? ""
    const owner = project?.name ?? "general"
    const prefix = project ? `project:${scope}` : "shared"
    const docs = (project ? documents[scope] ?? [] : notes)
      .filter((doc) => !pending(`document:${scope}:${doc.path}`))
    const chats = project ? buildChats(project, branchMeta) : visibleRootFiles.map((path) => ({ path, label: path.replace(/\.json$/, "") }))
    return [
      {
        id: `${prefix}:chats`, label: "chats", type: "folder",
        create: { label: `New chat in ${owner}`, onClick: () => onNewChat?.(scope || null) },
        onMoveHere: (path) => moveToProject(path, scope),
        children: chats.filter((chat) => !pending(`history:${chat.path}`)).map((chat) => {
          const count = branchMeta[chat.path]?.qa_count ?? 0
          return {
            id: chat.path, label: chat.label, type: "element", data: chat.path, draggable: true,
            badge: count > 0 ? `@${count - 1}` : undefined,
            isActive: appMode === "chat" && activeFile === chat.path && activeQaIndex == null,
            onSelect: () => onFileClick(chat.path),
            onDelete: () => deleteChat(chat.path),
          }
        }),
      },
      {
        id: `${prefix}:canvases`, label: "canvases", type: "folder",
        onMoveHere: (path) => moveToProject(path, scope),
        create: {
          label: `New canvas in ${owner}`, placeholder: "Canvas name",
          onSubmit: (name) => createAndOpen(scope, name.endsWith(".canvas") ? name : `${name}.canvas`, (path) => onCanvasSelect?.(path, scope)),
        },
        children: docs.filter((doc) => doc.name.endsWith(".canvas")).map((doc) => ({
          id: doc.path, label: doc.name.replace(/\.canvas$/, ""), type: "element", draggable: true,
          isActive: appMode === "canvas" && activeCanvasPath === doc.path,
          onSelect: () => onCanvasSelect?.(doc.path, scope),
          onDelete: () => deleteDocument(doc.path, scope),
        })),
      },
      ...(project ? [{
        id: `${prefix}:files`, label: "files", type: "folder" as const,
        create: {
          label: `New file in ${owner}`, placeholder: "File name",
          onSubmit: (name: string) => createAndOpen(scope, name, (path) => onDocumentSelect?.(path, scope)),
        },
        onDropFiles: (paths: string[]) => importFiles(scope, paths),
        children: docs.filter((doc) => !doc.name.endsWith(".canvas") && !isArchitecturePath(doc.path)).map((doc): FileTreeItem<string> => ({
          id: doc.path, label: doc.name, type: "element", isActive: activeDocumentPath === doc.path,
          onSelect: () => onDocumentSelect?.(doc.path, scope),
          onDelete: () => deleteDocument(doc.path, scope),
        })),
      }] : []),
    ]
  }

  const [generalChats, ...generalCategories] = buildCategories(null)
  const items: FileTreeItem<string>[] = [
    ...projects.filter((project) => !project.archived && !pending(`project:${project.slug}`)).map((project): FileTreeItem<string> => ({
      id: project.slug, label: project.name, type: "root", isActive: tab.projectSlug === project.slug,
      onSelect: () => onProjectSelect?.(project.slug),
      onDoubleClick: () => { setRenameError(""); setRenameTarget(project) },
      onDelete: () => schedule(`project:${project.slug}`, project.name, () => deleteProject(project.slug)),
      actions: [
        { label: `Archive ${project.name}`, icon: Archive, onClick: () => { if (!archivingProject) void changeProjectArchive(project.slug, true) } },
        { label: `New document in ${project.name}`, icon: FilePlus, onClick: () => setCreateDocumentSlug(project.slug) },
        { label: `${hasArchitecture(project.slug) ? "Open" : "Create"} architecture for ${project.name}`, icon: ArchitectureIcon,
          onClick: () => openArchitecture(project.slug) },
      ],
      onMoveHere: (path) => moveToProject(path, project.slug),
      onDropFiles: (paths) => importFiles(project.slug, paths),
      children: buildCategories(project),
    })),
    {
      id: "shared", label: "general", type: "root",
      create: generalChats.create,
      onMoveHere: (path) => moveToProject(path, ""),
      children: [...generalCategories, ...(generalChats.children ?? [])],
    },
  ]
  const archivedSessions = omp.archivedSessions()
  const directoryEntries = [
    ...rootTreeData.filter((node) => !pending(`root:${mountOf(node)}`)).map((node) => ({ key: mountOf(node), name: node.name })),
    ...unmounted.map((workspace) => ({ key: workspace.path, name: workspace.label })),
  ].map((entry) => ({ ...entry, archived: archivedDirectories.has(entry.key) }))

  return <motion.aside
    initial={false} animate={{ width: collapsed ? 0 : 280 }} transition={{ duration: 0.25, ease: "easeInOut" }}
    className={cn("relative shrink-0 flex flex-col transition-[margin,background-color,box-shadow,border-color] duration-200",
      collapsed ? "border-0" : "m-3 mr-0 rounded-2xl border border-divider/50 bg-surface shadow-[var(--shadow-lg)]")}
  >
    {!collapsed && <>
      <div className="flex h-10 shrink-0 items-center px-4">
        <button onClick={openMemoryViewer} aria-label="Open memory"
          className="rounded-md p-1 text-ink-subtle hover:bg-surface-elevated hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted transition-colors">
          <Brain className="h-4 w-4" />
        </button>
      </div>
      <nav aria-label="Workspace" className="min-h-0 flex-1 overflow-y-auto px-2 pb-3">
        <FileTree label="Projects" items={items} controller={controller} loading={historiesLoading}
          headerActions={<ArchiveSelector
            noun="projects"
            entries={projects.filter((project) => !pending(`project:${project.slug}`)).map((project) => ({ key: project.slug, name: project.name, archived: Boolean(project.archived) }))}
            onRestore={(slug) => { if (!archivingProject) void changeProjectArchive(slug, false) }}
          />}
          create={{ label: "New project", placeholder: "Project name", onSubmit: async (name) => {
            const slug = await createProject(name)
            controller.expand(slug)
            onNewChat?.(slug)
          } }}
          renderBeforeItem={(item) => item.data ? <ChatBranchPreview file={item.data} /> : null}
          onMoveToRoot={(path) => moveToProject(path, "")}
        />
        <ErrorText className="px-2 py-1">{projectArchiveError}</ErrorText>
        <div role="separator" className="my-4 border-t border-divider/50" />
        <FileTree label="Code" items={[...buildMounts(rootTreeData), ...buildUnmounted(), ...(archivedSessions ? [archivedSessions] : [])]} controller={fileMountController}
          headerActions={<ArchiveSelector noun="directories" entries={directoryEntries} onRestore={(path) => setDirectoryArchived(path, false)} />}
          labelAction={omp.available ? { label: "Open omp settings", onClick: omp.openSettings } : undefined}
          create={hostAccess ? { label: "Add file mount", onClick: () => setFileMountPickerOpen(true) } : undefined}
        />
        <ErrorText className="px-2 py-1">{omp.error}</ErrorText>
      </nav>
      <ProviderUsage />
      {renderSettings && <div className="shrink-0 p-2">{renderSettings()}</div>}
    </>}
    <FileMountPicker open={fileMountPickerOpen} onClose={() => setFileMountPickerOpen(false)} onConfirm={(path) => mountThen(() => addFileRoot(path))} onClone={(url) => mountThen(() => cloneFileRoot(url))} />
    <SaveAsModal open={renameTarget != null} onClose={() => { if (!renamePending) setRenameTarget(null) }}
      title="Rename project" icon={Pencil} initialName={renameTarget?.name} pending={renamePending} error={renameError}
      submitLabel="Rename" onSave={submitRename} />
    <CreateDocumentPanel open={createDocumentSlug != null} onClose={() => setCreateDocumentSlug(null)}
      onCreate={(name) => { if (createDocumentSlug) void createDocument(createDocumentSlug, name) }} />
  </motion.aside>
}

/** Registered path of a file-root node: its mount when set, else its own path. */
const mountOf = (node: FileTreeNode): string => node.mount ?? node.path

interface ArchiveEntry {
  key: string
  name: string
  archived: boolean
}

/** Header popover of every entry of a section, selecting an archived one restores it. */
function ArchiveSelector({ noun, entries, onRestore }: { noun: string; entries: ArchiveEntry[]; onRestore: (key: string) => void }) {
  return <PopoverSelector
    triggerLabel={`Show all ${noun}`} triggerContent={<Archive className="h-3 w-3" />}
    triggerClassName="h-auto rounded p-1 text-ink-faint opacity-0 group-hover/row:opacity-100 group-focus-within/row:opacity-100 aria-expanded:opacity-100 focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"
    label={`All ${noun}`} placement="below" panelClassName="left-auto right-0 w-56"
    categories={[{ key: noun, label: `All ${noun}`, items: entries }]}
    itemKey={(entry) => entry.key} isSelected={(entry) => !entry.archived}
    renderItem={(entry) => <>
      <span className="min-w-0 flex-1 truncate">{entry.name}</span>
      {entry.archived ? <span className="flex shrink-0 items-center gap-1.5 text-xs text-ink-subtle"><ArchiveRestore className="h-3 w-3" />Restore</span>
        : <span className="text-xs text-ink-faint">Active</span>}
    </>}
    onSelect={(entry) => { if (entry.archived) onRestore(entry.key) }}
    emptyText={`No ${noun}`}
  />
}

function buildChats(project: ProjectListItem, branchMeta: Record<string, BranchMeta>): {
  path: string
  label: string
}[] {
  const prefix = `${project.slug}/`
  const tabs = new Map((project.tabs ?? []).map((tab) => [tab.filename, tab]))
  return Object.entries(branchMeta).flatMap(([path, metadata]) => {
    if (!path.startsWith(prefix) || metadata.parent_id) return []
    const filename = path.slice(prefix.length)
    if (filename === "project.json" || filename.startsWith("memory/")) return []
    const tab = tabs.get(filename)
    return [{ path, label: tab?.name ?? tab?.title ?? filename.replace(/\.json$/, "") }]
  })
}

/** Worktrees and Sessions rows that exist, the last one spaced from the files below it. */
function generatedRows(rows: (FileTreeItem<string> | null)[]): FileTreeItem<string>[] {
  const present = rows.filter((row) => row !== null)
  return present.map((row, index) => index === present.length - 1 ? { ...row, gapAfter: true } : row)
}
