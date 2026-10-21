"use client"

import { useCallback, useEffect, useMemo, useState } from "react"
import { motion } from "framer-motion"
import { Brain, Factory, FilePlus, GitCompare } from "lucide-react"
import { ArchitectureIcon } from "@/components/ui/ArchitectureIcon"
import { CreateDocumentPanel } from "./CreateDocumentPanel"
import { ProviderUsage } from "./ProviderUsage"
import { cn } from "@/lib/utils"
import { FileTree, type FileTreeItem } from "@/components/ui/file-tree/FileTree"
import { useFileTree, usePathTree } from "@/components/ui/file-tree/useFileTree"
import { useMemoryViewer } from "@/features/memory/MemoryViewerContext"
import { useProject } from "@/contexts/ProjectContext"
import { useBranches } from "@/features/navigation/BranchContext"
import { useSidebar } from "@/features/navigation/SidebarContext"
import { useUndoDelete } from "@/contexts/UndoDeleteContext"
import type { BranchMeta, FileTreeNode, ProjectListItem, ProjectDocument } from "@/lib/domain/types"
import { ChatBranchPreview } from "./ChatBranchPreview"
import { FileMountPicker } from "./FileMountPicker"
import { useWorktrees } from "./useWorktrees"
import { addRootFilesToProject, addFileRoot, createArchitectureSource, fileRootTree, removeFileRoot, listNotes, listProjectDocuments, moveDocument, removeDocument, writeDocument } from "@/lib/backend/api"
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
  onFileSelect, onDiffSelect, onRootsChanged, onDocumentsChanged, onDocumentSelect, onProjectSelect, onNewChat,
  activeCanvasPath, onCanvasSelect, onCanvasDeleted, onArchitectureSelect, onItemOpen,
  activeDocumentPath, renderSettings, appMode,
}: SidebarProps) {
  const { collapsed } = useSidebar()
  const hostAccess = useHostAccess()
  const isActive = useTabActive()
  const { tab } = useTab()
  const { branchMeta, visibleRootFiles, historiesLoading, moveHistory, onFileClick, activeFile, activeQaIndex, cascadeDelete } = useBranches()
  const { projects, createProject, deleteProject } = useProject()
  const { openMemoryViewer } = useMemoryViewer()
  const { schedule, pending } = useUndoDelete()
  const defaultExpanded = useMemo(() => [
    "shared:canvases",
    ...projects.flatMap((project) => ["chats", "canvases", "files", "code"].map((category) => `project:${project.slug}:${category}`)),
  ], [projects])
  const controller = useFileTree({ storageKey: "expanded_workspace_projects", defaultExpanded })
  const fileMountController = usePathTree()
  const [documents, setDocuments] = useState<Record<string, ProjectDocument[]>>({})
  const [notes, setNotes] = useState<ProjectDocument[]>([])
  const [rootTreeData, setRootTreeData] = useState<FileTreeNode[]>([])
  const [createDocumentSlug, setCreateDocumentSlug] = useState<string | null>(null)
  const [fileMountPickerOpen, setFileMountPickerOpen] = useState(false)
  const [ompWorkspacePickerOpen, setOmpWorkspacePickerOpen] = useState(false)

  const refreshDocuments = useCallback(async () => {
    const [loose, entries] = await Promise.all([
      listNotes(),
      Promise.all(projects.map(async (project) => [project.slug, await listProjectDocuments(project.slug)] as const)),
    ])
    setNotes(loose)
    setDocuments(Object.fromEntries(entries))
  }, [projects])

  useEffect(() => {
    if (!isActive) return
    let cancelled = false
    Promise.all([
      listNotes(),
      Promise.all(projects.map(async (project) => [project.slug, await listProjectDocuments(project.slug)] as const)),
    ]).then(([loose, entries]) => {
      if (cancelled) return
      setNotes(loose)
      setDocuments(Object.fromEntries(entries))
    }).catch(() => {})
    return () => { cancelled = true }
  }, [projects, activeCanvasPath, appMode, isActive])

  const mounts = rootTreeData.map((node) => node.mount ?? node.path)
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
          reportError(cause instanceof Error ? cause.message : "Could not open the factory")
        }
      },
    }],
  })
  const worktrees = useWorktrees(mounts)
  const { reload: reloadWorktrees } = worktrees
  const activeWorktreesKey = worktrees.activeWorktrees.join("\n")

  const loadRootTree = useCallback(() => fileRootTree(activeWorktreesKey.split("\n").filter(Boolean))
    .then((result) => setRootTreeData(result.tree)), [activeWorktreesKey])

  useEffect(() => {
    if (!isActive) return
    let cancelled = false
    fileRootTree(activeWorktreesKey.split("\n").filter(Boolean)).then((result) => { if (!cancelled) setRootTreeData(result.tree) }).catch(() => {})
    return () => { cancelled = true }
  }, [isActive, activeWorktreesKey])

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
    .filter((node) => !pending(`root:${node.mount ?? node.path}`))
    .map((node) => {
      const mount = node.mount ?? node.path
      const folder = worktrees.folder(mount)
      return {
        id: mount,
        label: node.name,
        type: "folder",
        // Dragging a mount adds its path, only unambiguous while it shows the clone itself.
        multiSelect: node.path === mount,
        actions: [{ label: `Open diff for ${node.name}`, icon: GitCompare, onClick: () => onDiffSelect?.(node.path) }],
        onDelete: () => schedule(`root:${mount}`, node.name, async () => {
          await removeFileRoot(mount)
          await loadRootTree()
          onRootsChanged?.()
        }),
        children: [...(folder ? [folder] : []), ...buildRootFiles(node.children ?? [])],
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
    const mount = project ? projectMount(project.slug, mounts) : null
    const code = project && mount ? omp.codeCategory(project, mount) : null
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
            onDelete: () => schedule(`history:${chat.path}`, baseName(chat.path), async () => { await cascadeDelete(chat.path) }),
          }
        }),
      },
      {
        id: `${prefix}:canvases`, label: "canvases", type: "folder",
        onMoveHere: (path) => moveToProject(path, scope),
        create: {
          label: `New canvas in ${owner}`, placeholder: "Canvas name",
          onSubmit: async (name) => {
            const doc = await writeDocument(scope, name.endsWith(".canvas") ? name : `${name}.canvas`)
            await refreshDocuments()
            onCanvasSelect?.(doc.path, scope)
          },
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
          onSubmit: async (name: string) => {
            const doc = await writeDocument(scope, name)
            await refreshDocuments()
            onDocumentSelect?.(doc.path, scope)
          },
        },
        onDropFiles: (paths: string[]) => importFiles(scope, paths),
        children: docs.filter((doc) => !doc.name.endsWith(".canvas") && !isArchitecturePath(doc.path)).map((doc): FileTreeItem<string> => ({
          id: doc.path, label: doc.name, type: "element", isActive: activeDocumentPath === doc.path,
          onSelect: () => onDocumentSelect?.(doc.path, scope),
          onDelete: () => deleteDocument(doc.path, scope),
        })),
      }] : []),
      ...(code ? [code] : []),
    ]
  }

  const items: FileTreeItem<string>[] = [
    ...projects.filter((project) => !pending(`project:${project.slug}`)).map((project): FileTreeItem<string> => ({
      id: project.slug, label: project.name, type: "root", isActive: tab.projectSlug === project.slug,
      onSelect: () => onProjectSelect?.(project.slug),
      onDelete: () => schedule(`project:${project.slug}`, project.name, () => deleteProject(project.slug)),
      actions: [
        { label: `New document in ${project.name}`, icon: FilePlus, onClick: () => setCreateDocumentSlug(project.slug) },
        { label: `${hasArchitecture(project.slug) ? "Open" : "Create"} architecture for ${project.name}`, icon: ArchitectureIcon,
          onClick: () => openArchitecture(project.slug) },
        { label: `Open memory for ${project.name}`, icon: Brain,
          onClick: () => openMemoryViewer({ scope: "project", projectSlug: project.slug }) },
      ],
      onMoveHere: (path) => moveToProject(path, project.slug),
      onDropFiles: (paths) => importFiles(project.slug, paths),
      children: buildCategories(project),
    })),
  ]
  const [generalChats, ...generalCategories] = buildCategories(null)
  const linkedMounts = projects.flatMap((project) => projectMount(project.slug, mounts) ?? [])

  return <motion.aside
    initial={false} animate={{ width: collapsed ? 0 : 280 }} transition={{ duration: 0.25, ease: "easeInOut" }}
    className={cn("relative shrink-0 flex flex-col transition-[margin,background-color,box-shadow,border-color] duration-200",
      collapsed ? "border-0" : "m-3 mr-0 rounded-2xl border border-divider/50 bg-surface shadow-[var(--shadow-lg)]")}
  >
    {!collapsed && <>
      <div className="flex h-10 shrink-0 items-center px-4">
        <button onClick={() => openMemoryViewer({ scope: "global" })} aria-label="Open global memory profile"
          className="rounded-md p-1 text-ink-subtle hover:bg-surface-elevated hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted transition-colors">
          <Brain className="h-4 w-4" />
        </button>
      </div>
      <nav aria-label="Workspace" className="min-h-0 flex-1 overflow-y-auto px-2 pb-3">
        <FileTree label="Projects" items={items} controller={controller} loading={historiesLoading}
          create={{ label: "New project", placeholder: "Project name", onSubmit: async (name) => {
            const slug = await createProject(name)
            controller.expand(slug)
            onNewChat?.(slug)
          } }}
          renderBeforeItem={(item) => item.data ? <ChatBranchPreview file={item.data} /> : null}
          onMoveToRoot={(path) => moveToProject(path, "")}
        />
        <div role="separator" className="my-4 border-t border-divider/50" />
        <FileTree label="General" items={[...(generalChats.children ?? []), ...generalCategories]} controller={controller} loading={historiesLoading}
          create={generalChats.create}
          renderBeforeItem={(item) => item.data ? <ChatBranchPreview file={item.data} /> : null}
          onMoveToRoot={(path) => moveToProject(path, "")}
        />
        <div role="separator" className="my-4 border-t border-divider/50" />
        <FileTree label="File mounts" items={buildMounts(rootTreeData)} controller={fileMountController}
          create={{ label: "Add file mount", onClick: () => setFileMountPickerOpen(true) }}
        />
        {omp.available && <>
          <div role="separator" className="my-4 border-t border-divider/50" />
          <FileTree label="OMP" items={omp.workspaceItems(linkedMounts)} controller={controller}
            labelAction={{ label: "Open omp settings", onClick: omp.openSettings }}
            create={{ label: "Add omp workspace", onClick: () => setOmpWorkspacePickerOpen(true) }}
          />
          {omp.error && <p role="alert" className="px-2 py-1 text-xs text-danger">{omp.error}</p>}
        </>}
      </nav>
      <ProviderUsage />
      {renderSettings && <div className="shrink-0 p-2">{renderSettings()}</div>}
    </>}
    <FileMountPicker open={fileMountPickerOpen} onClose={() => setFileMountPickerOpen(false)} onConfirm={async (path) => {
      await addFileRoot(path)
      await loadRootTree()
      onRootsChanged?.()
    }} />
    <FileMountPicker noun="omp workspace" open={ompWorkspacePickerOpen} onClose={() => setOmpWorkspacePickerOpen(false)} onConfirm={omp.addWorkspace} />
    <CreateDocumentPanel open={createDocumentSlug != null} onClose={() => setCreateDocumentSlug(null)}
      onCreate={(name) => { if (createDocumentSlug) void createDocument(createDocumentSlug, name) }} />
  </motion.aside>
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
