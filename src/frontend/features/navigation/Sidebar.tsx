"use client"

import { useCallback, useEffect, useMemo, useState } from "react"
import { motion } from "framer-motion"
import { Brain } from "lucide-react"
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
import { WorkspacePicker } from "./WorkspacePicker"
import { addRootFilesToProject, addFileRoot, createArchitectureSource, fileRootTree, removeFileRoot, listNotes, listProjectDocuments, moveDocument, removeDocument, writeDocument } from "@/lib/backend/api"
import { useTab, useTabActive, type AppSurface } from "@/contexts/TabContext"
import { isArchitecturePath } from "@/lib/domain/architecturePaths"
import { baseName } from "@/lib/paths"

interface SidebarProps {
  onFileSelect?: (path: string) => void
  onRootsChanged?: () => void
  onDocumentsChanged?: () => void
  onDocumentSelect?: (path: string, scope: string) => void
  onProjectSelect?: (slug: string) => void
  onNewChat?: (slug: string | null) => void
  activeCanvasPath?: string | null
  activeDocumentPath?: string | null
  onCanvasSelect?: (path: string, scope: string) => void
  onCanvasDeleted?: (path: string) => void
  activeArchitecture?: string | null
  onArchitectureSelect?: (slug: string) => void
  renderSettings?: () => React.ReactNode
  appMode: AppSurface
}

export function Sidebar({
  onFileSelect, onRootsChanged, onDocumentsChanged, onDocumentSelect, onProjectSelect, onNewChat,
  activeCanvasPath, onCanvasSelect, onCanvasDeleted, activeArchitecture, onArchitectureSelect,
  activeDocumentPath, renderSettings, appMode,
}: SidebarProps) {
  const { collapsed } = useSidebar()
  const isActive = useTabActive()
  const { tab } = useTab()
  const { branchMeta, visibleRootFiles, historiesLoading, moveHistory, onFileClick, activeFile, activeQaIndex, cascadeDelete } = useBranches()
  const { projects, createProject, deleteProject } = useProject()
  const { openMemoryViewer } = useMemoryViewer()
  const { schedule, pending } = useUndoDelete()
  const defaultExpanded = useMemo(() => [
    "shared:canvases",
    ...projects.flatMap((project) => ["chats", "canvases", "architecture", "files"].map((category) => `project:${project.slug}:${category}`)),
  ], [projects])
  const controller = useFileTree({ storageKey: "expanded_workspace_projects", defaultExpanded })
  const workspaceController = usePathTree()
  const [documents, setDocuments] = useState<Record<string, ProjectDocument[]>>({})
  const [notes, setNotes] = useState<ProjectDocument[]>([])
  const [rootTreeData, setRootTreeData] = useState<FileTreeNode[]>([])
  const [workspacePickerOpen, setWorkspacePickerOpen] = useState(false)

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

  useEffect(() => {
    if (!isActive) return
    let cancelled = false
    fileRootTree().then((result) => { if (!cancelled) setRootTreeData(result.tree) }).catch(() => {})
    return () => { cancelled = true }
  }, [isActive])

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

  const buildRootFiles = (nodes: FileTreeNode[]): FileTreeItem<string>[] => nodes
    .filter((node) => !pending(`root:${node.path}`))
    .map((node) => ({
      id: node.path,
      label: node.name,
      type: node.type === "file" ? "element" : "folder",
      multiSelect: true,
      isActive: node.path === activeDocumentPath,
      onSelect: node.type === "file" ? () => onFileSelect?.(node.path) : undefined,
      onDelete: node.type === "root" ? () => schedule(`root:${node.path}`, node.name, async () => {
        const result = await removeFileRoot(node.path)
        setRootTreeData(result.tree)
        onRootsChanged?.()
      }) : undefined,
      children: node.children ? buildRootFiles(node.children) : undefined,
    }))

  const buildCategories = (project: ProjectListItem | null): FileTreeItem<string>[] => {
    const scope = project?.slug ?? ""
    const owner = project?.name ?? "general"
    const prefix = project ? `project:${scope}` : "shared"
    const docs = (project ? documents[scope] ?? [] : notes)
      .filter((doc) => !pending(`document:${scope}:${doc.path}`))
    const chats = project ? buildChats(project, branchMeta) : visibleRootFiles.map((path) => ({ path, label: path.replace(/\.json$/, "") }))
    const hasArchitecture = docs.some((doc) => isArchitecturePath(doc.path))
    return [
      {
        id: `${prefix}:chats`, label: "sessions", type: "folder",
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
        id: `${prefix}:architecture`, label: "architecture", type: "folder" as const,
        create: hasArchitecture ? undefined : {
          label: `New architecture in ${owner}`,
          onClick: async () => {
            await createArchitectureSource(scope, "system.c4")
            await refreshDocuments()
            onArchitectureSelect?.(scope)
          },
        },
        children: hasArchitecture ? [{
          id: `graph/${scope}/`, label: "System overview", type: "element" as const,
          isActive: appMode === "canvas" && activeArchitecture === scope,
          onSelect: () => onArchitectureSelect?.(scope),
        }] : [],
      }] : []),
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
    ]
  }

  const items: FileTreeItem<string>[] = [
    ...projects.filter((project) => !pending(`project:${project.slug}`)).map((project): FileTreeItem<string> => ({
      id: project.slug, label: project.name, type: "root", isActive: tab.projectSlug === project.slug,
      onSelect: () => onProjectSelect?.(project.slug),
      onDelete: () => schedule(`project:${project.slug}`, project.name, () => deleteProject(project.slug)),
      actions: [{ label: `Open memory for ${project.name}`, icon: Brain,
        onClick: () => openMemoryViewer({ scope: "project", projectSlug: project.slug }) }],
      onMoveHere: (path) => moveToProject(path, project.slug),
      onDropFiles: (paths) => importFiles(project.slug, paths),
      children: buildCategories(project),
    })),
  ]
  const [generalChats, ...generalCategories] = buildCategories(null)

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
        <FileTree label="Workspaces" items={buildRootFiles(rootTreeData)} controller={workspaceController}
          create={{ label: "Add workspace", onClick: () => setWorkspacePickerOpen(true) }}
        />
      </nav>
      {renderSettings && <div className="shrink-0 p-2">{renderSettings()}</div>}
    </>}
    <WorkspacePicker open={workspacePickerOpen} onClose={() => setWorkspacePickerOpen(false)} onConfirm={async (path) => {
      const result = await addFileRoot(path)
      setRootTreeData(result.tree)
      onRootsChanged?.()
    }} />
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
