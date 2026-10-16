"use client"

import { useCallback, useEffect, useMemo, useState } from "react"
import { motion } from "framer-motion"
import { FileTree, type FileTreeItem } from "./FileTree"
import { useFileTree } from "@/hooks/useFileTree"
import { Boxes, Brain, Clock, FileText, FolderOpen, Lightbulb, LayoutDashboard, PanelLeftClose, PanelLeft, PenLine, Save, RotateCcw } from "lucide-react"
import { useMemoryViewer } from "@/contexts/MemoryViewerContext"
import { SidebarElement } from "./SidebarElement"
import { useProject } from "@/contexts/ProjectContext"
import { useBranches } from "@/contexts/BranchContext"
import { useSidebar, type AppSurface } from "@/contexts/SidebarContext"
import { useUndoDelete } from "@/contexts/UndoDeleteContext"
import type { BranchMeta, FileTreeNode, ProjectListItem, ProjectDocument } from "@/lib/types"
import { ChatBranchItem } from "./ChatBranchItem"
import { addRootFilesToProject, addFileRootMountpoint, addFileRoot, createDirectory, moveHistoryItem, fileRootTree, removeFileRoot, listNotes, listProjectDocuments, moveDocument, removeDocument, writeDocument, HistoryFolder } from "@/lib/api"
import { isArchitecturePath } from "@/lib/architecture"

const COLLAPSED_WIDTH = 50
const EXPANDED_WIDTH = 280

/** Drops tree rows whose deletion is pending, at any depth; a hidden folder hides its children. */
function hidePending<T>(items: FileTreeItem<T>[], hidden: (item: FileTreeItem<T>) => boolean): FileTreeItem<T>[] {
  return items.filter((item) => !hidden(item)).map((item) => item.children
    ? { ...item, children: hidePending(item.children, hidden) }
    : item)
}

interface SidebarProps {
  onOpenChat: (filename: string, qaIndex?: number) => void
  onRefreshAll: () => Promise<void>
  onSave: () => void
  onReset: () => void
  onMerge?: (childFile: string) => Promise<void>
  onCascadeDelete?: (filename: string) => Promise<void>
  onFileSelect?: (path: string) => void
  onRootsChanged?: () => void
  /** Project documents changed through the sidebar (e.g. files dropped onto a project). */
  onDocumentsChanged?: () => void
  activeCanvasPath?: string | null
  onCanvasSelect?: (path: string, scope: string) => void
  onCanvasDeleted?: (path: string) => void
  /** Project whose architecture is shown standalone in canvas mode, instead of a canvas. */
  activeArchitecture?: string | null
  onArchitectureSelect?: (slug: string) => void
  appMode: AppSurface
  onAppModeChange: (mode: AppSurface) => void
}

export function Sidebar({
  onOpenChat,
  onRefreshAll,
  onSave,
  onReset,
  onMerge,
  onCascadeDelete,
  onFileSelect,
  onRootsChanged,
  onDocumentsChanged,
  activeCanvasPath,
  onCanvasSelect,
  onCanvasDeleted,
  activeArchitecture,
  onArchitectureSelect,
  appMode,
  onAppModeChange,
}: SidebarProps) {
  const {
    collapsed,
    toggleCollapsed,
    projectsOpen,
    setProjectsOpen,
    historiesOpen,
    setHistoriesOpen,
  } = useSidebar()

  const {
    branchMeta,
    visibleRootFiles: rootFiles,
    visibleHistories: histories,
    historiesLoading,
    registerOnFileClick,
    registerOnMerge,
    registerOnDelete,
  } = useBranches()

  const {
    projects,
    activeProject,
    openProject,
    closeProject,
    createProject,
    deleteProject,
    setDashboardOpen,
  } = useProject()

  const { openMemoryViewer } = useMemoryViewer()
  const { schedule, pending } = useUndoDelete()

  const handleTimelineDelete = useCallback(async (file: string) => {
    schedule(`history:${file}`, file.split("/").pop() ?? file, async () => { await onCascadeDelete?.(file) })
  }, [onCascadeDelete, schedule])

  const expandIfCollapsed = () => { if (collapsed) toggleCollapsed() }

  const sidebarItems = [
    { id: "toggle-collapse", icon: collapsed ? PanelLeft : PanelLeftClose, onClick: toggleCollapsed },
    { id: "save-chat", icon: Save, title: "Save Chat", onClick: onSave },
    { id: "reset-chat", icon: RotateCcw, title: "Reset History", onClick: onReset },
  ]

  const toggleItem = sidebarItems.find(item => item.id === "toggle-collapse")
  const mainItems = sidebarItems.filter(
    item => item.id !== "toggle-collapse" && item.id !== "save-chat" && item.id !== "reset-chat",
  )

  const sidebarWidth = collapsed ? COLLAPSED_WIDTH : EXPANDED_WIDTH
  const animateWidth = useMemo(() => ({ width: sidebarWidth }), [sidebarWidth])


  // Selection primitives — one per tree, so the trees behave identically yet stay
  // independent (collapsing a project never touches a histories folder).
  // Projects: accordion sourced from activeProject; opening loads the project
  // context, closing tears it down. Histories: a persisted multi-open file browser.
  const projectsController = useFileTree({
    accordion: true,
    activeId: activeProject,
    onActivate: openProject,
    onDeactivate: closeProject,
    // ponytail: mounted-root ids are absolute paths; only project slugs join the accordion
    isAccordionId: (id) => !id.startsWith("/"),
  })
  const historiesController = useFileTree({ storageKey: "expanded_history_folders" })
  const filesController = useFileTree({ storageKey: "expanded_file_folders" })
  const canvasController = useFileTree({ storageKey: "expanded_canvas_folders" })
  const [canvasOpen, setCanvasOpen] = useState(true)
  const [canvasNotes, setCanvasNotes] = useState<ProjectDocument[]>([])
  const [projectCanvases, setProjectCanvases] = useState<Record<string, ProjectDocument[]>>({})
  // Projects with at least one architecture source.
  const [architectures, setArchitectures] = useState<ReadonlySet<string>>(new Set())

  const refreshCanvases = useCallback(async () => {
    const notes = await listNotes().catch(() => [])
    setCanvasNotes(notes.filter((d) => d.name.endsWith(".canvas")))
    const entries = await Promise.all(projects.map(async (p) => [p.slug, await listProjectDocuments(p.slug).catch(() => [])] as const))
    setProjectCanvases(Object.fromEntries(entries.map(([slug, docs]) => [slug, docs.filter((d) => d.name.endsWith(".canvas"))])))
    setArchitectures(new Set(entries.filter(([, docs]) => docs.some((d) => isArchitecturePath(d.path))).map(([slug]) => slug)))
  }, [projects])

  useEffect(() => {
    if (appMode === "canvas") refreshCanvases()
  }, [appMode, activeCanvasPath, refreshCanvases])

  const canvasItems = useMemo(() =>
    hidePending(buildCanvasItems(canvasNotes, projects, projectCanvases, architectures, activeCanvasPath ?? null, activeArchitecture ?? null),
      (item) => !!item.data && pending(`canvas:${item.data.scope}:${item.data.path}`)),
    [canvasNotes, projects, projectCanvases, architectures, activeCanvasPath, activeArchitecture, pending],
  )

  // File roots: maintained server-side in the database; rendered
  // as a file tree. Clicking a file attaches it to the active chat as a live reference.
  const [rootTreeData, setRootTreeData] = useState<FileTreeNode[]>([])
  const [filesOpen, setFilesOpen] = useState(false)

  useEffect(() => {
    let cancelled = false
    fileRootTree()
      .then((r) => { if (!cancelled) setRootTreeData(r.tree) })
      .catch(() => {})
    return () => { cancelled = true }
  }, [])

  // A file-root root, standalone (Files) or mounted to a project (Projects), is keyed by its absolute path.
  const deleteFileRoot = (path: string) => schedule(`root:${path}`, path.split("/").pop() ?? path, async () => {
    const result = await removeFileRoot(path)
    setRootTreeData(result.tree)
    onRootsChanged?.()
  })

  // Roots mounted to a project render inside that project's subtree, not here.
  const fileItems = useMemo(() => hidePending(buildFileItems(rootTreeData.filter((node) => !node.project)),
    (item) => pending(`root:${item.id}`)), [rootTreeData, pending])
  const projectItems = useMemo(() =>
    // Projects tree rows: absolute ids are mounted file roots, other file root rows are projects, elements are chats.
    hidePending(buildProjectItems(projects, activeProject, branchMeta, rootTreeData),
      (item) => pending(`${item.id.startsWith("/") ? "root" : item.type === "root" ? "project" : "history"}:${item.id}`)),
    [projects, activeProject, branchMeta, rootTreeData, pending],
  )
  const historyItems = useMemo(() =>
    hidePending(buildHistoryItems(rootFiles, histories, projects), (item) => pending(`history:${item.id}`)),
    [rootFiles, histories, projects, pending],
  )

  const renderChatElement = useCallback((item: FileTreeItem<string>, depth: number) => {
    const data = typeof item.data === "string" ? item.data : item.id
    // ponytail: absolute path = file inside a mounted file root; chat files are always relative
    if (typeof data === "string" && data.startsWith("/")) {
      return (
        <div style={{ paddingLeft: 8 + depth * 16, paddingRight: 8 }}>
          <button
            onClick={() => onFileSelect?.(data)}
            className="flex w-full items-center gap-1.5 truncate py-0.5 text-left text-[11px] text-ink-muted transition-colors hover:text-ink"
          >
            <FileText className="h-3.5 w-3.5 shrink-0 text-ink-muted" />
            {item.label}
          </button>
        </div>
      )
    }
    return (
      <ChatBranchItem
        file={data}
        label={item.label}
        depth={depth}
      />
    )
  }, [onFileSelect])

  useEffect(() => { registerOnFileClick(onOpenChat) }, [onOpenChat, registerOnFileClick])
  useEffect(() => { if (onMerge) registerOnMerge(onMerge) }, [onMerge, registerOnMerge])
  useEffect(() => { registerOnDelete(handleTimelineDelete) }, [handleTimelineDelete, registerOnDelete])

  return (
    <motion.aside
      initial={false}
      animate={animateWidth}
      transition={{ duration: 0.25, ease: "easeInOut" }}
      className="shrink-0 border-r border-divider bg-paper flex flex-col overflow-hidden shadow-[inset_-1px_0_0_0_oklch(100%_0_0/0.02)]"
    >
      <div className="group flex items-center justify-between px-2 py-2 border-b border-divider h-[60px] bg-paper/60 backdrop-blur-lg">
        {!collapsed && (
          <div className="flex items-center gap-1.5 px-2 min-w-0">
            <button
              onClick={() => onAppModeChange(appMode === "canvas" ? "chat" : "canvas")}
              className="text-base font-semibold tracking-tight text-ink truncate hover:opacity-80 transition-opacity"
            >
              {appMode === "canvas" ? "Canvas" : "Chat"}
            </button>
            {appMode !== "canvas" && (
              <button
                onClick={() => openMemoryViewer({ scope: "global" })}
                aria-label="Open global memory profile"
                className="shrink-0 rounded-md p-1 text-ink-subtle opacity-0 group-hover:opacity-100 hover:bg-surface hover:text-ink transition-all"
              >
                <Brain className="h-4 w-4" />
              </button>
            )}
          </div>
        )}
        <div className={collapsed ? "flex justify-center w-full" : "flex justify-end"}>
          {toggleItem && (
            <SidebarElement
              {...toggleItem}
              collapsed={collapsed}
              className="w-auto p-1.5 hover:bg-surface"
            />
          )}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto py-3 flex flex-col gap-2">
        {appMode === "canvas" ? (
          <div className="px-2">
            <FileTree<CanvasData>
              sectionIcon={Lightbulb}
              sectionTitle="Projects"
              items={canvasItems}
              collapsed={collapsed}
              open={canvasOpen}
              onOpenChange={setCanvasOpen}
              onExpand={expandIfCollapsed}
              controller={canvasController}
              plusTitle="New canvas"
              onElementClick={(item) => {
                if (!item.data) return
                if (item.data.architecture) onArchitectureSelect?.(item.data.scope)
                else onCanvasSelect?.(item.data.path, item.data.scope)
              }}
              folderPlaceholder="Canvas name"
              onElementDelete={(item) => {
                if (!item.data) return
                const { scope, path } = item.data
                schedule(`canvas:${scope}:${path}`, item.label, async () => {
                  await removeDocument(scope, path)
                  await refreshCanvases()
                  onCanvasDeleted?.(path)
                })
              }}
              onAddFolder={async (parentId, name) => {
                const scope = parentId ?? ""
                const filename = name.endsWith(".canvas") ? name : `${name}.canvas`
                const doc = await writeDocument(scope, filename)
                refreshCanvases()
                setCanvasOpen(true)
                onCanvasSelect?.(doc.path, scope)
              }}
              onMoveElement={async (elementId, targetId) => {
                if (elementId.startsWith("graph/")) return // a project's architecture stays with it
                const inProject = Object.entries(projectCanvases).find(([, docs]) => docs.some((d) => d.path === elementId))
                const fromSlug = inProject ? inProject[0] : ""
                const toSlug = targetId ?? ""
                if (fromSlug === toSlug) return
                try {
                  await moveDocument(elementId, fromSlug, toSlug)
                  refreshCanvases()
                } catch { /* conflict or invalid target — leave the tree as-is */ }
              }}
            />
          </div>
        ) : (
        <>
        <div className="px-2">
          <FileTree
            sectionIcon={Lightbulb}
            sectionTitle="Projects"
            items={projectItems}
            collapsed={collapsed}
            open={projectsOpen}
            onOpenChange={setProjectsOpen}
            onExpand={expandIfCollapsed}
            controller={projectsController}
            onRootDelete={(id) => {
              if (id.startsWith("/")) deleteFileRoot(id)
              else schedule(`project:${id}`, id, () => deleteProject(id))
            }}
            onAddRoot={(name) => createProject(name)}
            onAddMountpoint={async (rootId: string, path: string) => {
              // Project row → mount a FileRoots to the project; mounted file root row → nested mountpoint.
              const r = rootId.startsWith("/")
                ? await addFileRootMountpoint(rootId, path)
                : await addFileRoot(path, rootId)
              setRootTreeData(r.tree)
              onRootsChanged?.()
            }}
            mountpointPlaceholder="Folder path…"
            onDashboardClick={(rootId) => {
              openProject(rootId)
              setDashboardOpen(true)
            }}
            onMemoryClick={(rootId) => openMemoryViewer({ scope: "project", projectSlug: rootId })}
            onMoveElement={async (elementId: string, targetId: string | null) => {
              const slug = targetId && projects.find(p => p.slug === targetId)?.slug
              if (!slug) return
              await moveHistoryItem(elementId, slug)
              await onRefreshAll()
            }}
            renderElement={renderChatElement}
            onDropFiles={async (paths, targetId) => {
              const slug = projects.find((p) => p.slug === targetId)?.slug
              if (!slug) return
              try {
                await addRootFilesToProject(slug, paths)
                onDocumentsChanged?.()
              } catch { /* nothing valid dropped, or project gone — leave documents as-is */ }
            }}
          />
        </div>

        <div className="px-2">
          <FileTree
            sectionIcon={Clock}
            sectionTitle="Histories"
            items={historyItems}
            loading={historiesLoading}
            collapsed={collapsed}
            open={historiesOpen}
            onOpenChange={setHistoriesOpen}
            onExpand={expandIfCollapsed}
            plusTitle="New folder"
            controller={historiesController}
            renderElement={renderChatElement}
            onAddFolder={async (parentId: string | null, name: string) => {
              await createDirectory(parentId ?? "", name)
              await onRefreshAll()
              setHistoriesOpen(true)
            }}
            onRootDelete={(id: string) => {
              schedule(`history:${id}`, id.split("/").pop() ?? id, async () => {
                await new HistoryFolder(id).delete()
                await onRefreshAll()
              })
            }}
            onMoveElement={async (elementId: string, targetId: string | null) => {
              await moveHistoryItem(elementId, targetId ?? "")
              await onRefreshAll()
            }}
          />
        </div>

        <div className="px-2">
          <FileTree
            sectionIcon={FolderOpen}
            sectionTitle="Files"
            items={fileItems}
            collapsed={collapsed}
            open={filesOpen}
            onOpenChange={setFilesOpen}
            onExpand={expandIfCollapsed}
            plusTitle="Add folder"
            controller={filesController}
            rootPlaceholder="Folder path…"
            multiSelect
            onElementClick={(item) => onFileSelect?.((item.data as string) ?? item.id)}
            onAddRoot={async (path: string) => {
              const r = await addFileRoot(path)
              setRootTreeData(r.tree)
              setFilesOpen(true)
              onRootsChanged?.()
            }}
            onAddMountpoint={async (rootId: string, path: string) => {
              const r = await addFileRootMountpoint(rootId, path)
              setRootTreeData(r.tree)
              onRootsChanged?.()
            }}
            onRootDelete={deleteFileRoot}
          />
        </div>

        <div className="px-2 flex flex-col gap-1">
          {mainItems.map((item) => (
            <SidebarElement
              key={item.id}
              {...item}
              collapsed={collapsed}
            />
          ))}
        </div>
        </>
        )}
      </div>
    </motion.aside>
  )
}

interface CanvasData {
  path: string
  scope: string
  /** The project's architecture (shown on its own), not a canvas file. */
  architecture?: boolean
}

function buildCanvasItems(
  canvasNotes: ProjectDocument[],
  projects: ProjectListItem[],
  projectCanvases: Record<string, ProjectDocument[]>,
  architectures: ReadonlySet<string>,
  activeCanvasPath: string | null,
  activeArchitecture: string | null,
): FileTreeItem<CanvasData>[] {
  const canvasElement = (doc: ProjectDocument, scope: string): FileTreeItem<CanvasData> => ({
    id: doc.path,
    label: doc.name.replace(/\.canvas$/, ""),
    type: "element",
    icon: PenLine,
    data: { path: doc.path, scope },
    isActive: activeCanvasPath === doc.path,
  })
  // Not a file of its own, so it cannot be deleted or moved from here.
  const architectureElement = (slug: string): FileTreeItem<CanvasData> => ({
    id: `graph/${slug}/`,
    label: "Architecture",
    type: "element",
    icon: Boxes,
    data: { path: `graph/${slug}/`, scope: slug, architecture: true },
    isActive: activeArchitecture === slug,
    isSystem: true,
  })

  const projectItems: FileTreeItem<CanvasData>[] = projects
    .filter((p) => architectures.has(p.slug) || (projectCanvases[p.slug] ?? []).length > 0)
    .map((p) => ({
      id: p.slug,
      label: p.name,
      type: "root" as const,
      icon: Lightbulb,
      children: [
        ...(architectures.has(p.slug) ? [architectureElement(p.slug)] : []),
        ...(projectCanvases[p.slug] ?? []).map((doc) => canvasElement(doc, p.slug)),
      ],
    }))

  const rootItems = canvasNotes.map((doc) => canvasElement(doc, ""))
  return [...projectItems, ...rootItems]
}

function buildFileItems(nodes: FileTreeNode[]): FileTreeItem<string>[] {
  return nodes.map((n) => {
    if (n.type === "file") {
      return { id: n.path, label: n.name, type: "element" as const, data: n.path, icon: FileText }
    }
    const isRootNode = n.type === "root"
    return {
      id: n.path,
      label: n.name,
      type: n.type, // "root" | "folder"
      data: n.path,
      icon: isRootNode ? FolderOpen : undefined,
      mountable: isRootNode,
      mounted: isRootNode,
      children: buildFileItems(n.children ?? []),
    }
  })
}

function buildProjectItems(
  projects: ProjectListItem[],
  activeProject: string | null,
  branchMeta: Record<string, BranchMeta>,
  rootTree: FileTreeNode[],
): FileTreeItem<string>[] {
  return projects.map((project) => {
    const dirPrefix = `${project.slug}/`
    const tabByName = new Map((project.tabs ?? []).map(t => [t.filename, t]))

    const items: FileTreeItem<string>[] = []
    for (const [key, meta] of Object.entries(branchMeta)) {
      if (!key.startsWith(dirPrefix) || key === `${project.slug}/project.json`) continue
      if (meta.parent_id) continue
      const filename = key.slice(dirPrefix.length)
      if (filename.startsWith("memory/")) continue
      const tab = tabByName.get(filename)
      const label = tab ? (tab.name ?? tab.title ?? filename.replace(".json", "")) : filename.replace(".json", "")
      items.push({ id: key, label, type: "element", data: key })
    }

    // FileRoots roots mounted to this project appear as file root nodes in its subtree.
    const mountedRoots = buildFileItems(rootTree.filter((n) => n.project === project.slug))

    return {
      id: project.slug,
      label: project.name,
      type: "root",
      data: project.slug,
      icon: Lightbulb,
      mountable: true,
      isActive: activeProject === project.slug,
      children: [
        ...mountedRoots,
        {
          id: `${project.slug}/__dashboard__`,
          label: "Dashboard",
          type: "element" as const,
          icon: LayoutDashboard,
          isSystem: true,
          displayInRoot: false,
          data: project.slug,
        },
        ...items,
      ].filter((child: FileTreeItem<string>) => child.displayInRoot !== false),
    }
  })
}

function buildHistoryItems(
  rootFiles: string[],
  histories: Record<string, string[]>,
  projects: ProjectListItem[],
): FileTreeItem<string>[] {
  const projectSlugs = new Set(projects.map((p) => p.slug))

  const folderItems = Object.entries(histories)
    .filter(([dir]) => !projectSlugs.has(dir))
    .map(([dir, files]) => ({
      id: dir,
      label: dir,
      type: "folder" as const,
      data: dir,
      children: files.map((file) => {
        const key = `${dir}/${file}`
        return {
          id: key,
          label: file.replace(".json", ""),
          type: "element" as const,
          data: key,
          
        }
      }),
    }))

  const rootItems = rootFiles.map((file) => {
    return {
      id: file,
      label: file.replace(".json", ""),
      type: "element" as const,
      data: file,
      
    }
  })

  return [...folderItems, ...rootItems]
}
