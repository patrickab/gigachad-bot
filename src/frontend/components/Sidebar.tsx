"use client"

import { cn } from "@/lib/utils"
import { useCallback, useEffect, useMemo, useState } from "react"
import { motion } from "framer-motion"
import { FileTree, type FileTreeItem } from "./FileTree"
import { useFileTree } from "@/hooks/useFileTree"
import { Boxes, Brain, FileText, FolderOpen, Lightbulb, PenLine } from "lucide-react"
import { useMemoryViewer } from "@/contexts/MemoryViewerContext"
import { useProject } from "@/contexts/ProjectContext"
import { useBranches } from "@/contexts/BranchContext"
import { useSidebar } from "@/contexts/SidebarContext"
import { useUndoDelete } from "@/contexts/UndoDeleteContext"
import type { BranchMeta, FileTreeNode, ProjectListItem, ProjectDocument } from "@/lib/types"
import { ChatBranchItem } from "./ChatBranchItem"
import { addRootFilesToProject, addFileRoot, moveHistoryItem, fileRootTree, removeFileRoot, listNotes, listProjectDocuments, moveDocument, removeDocument, writeDocument } from "@/lib/api"
import type { AppSurface } from "@/contexts/TabContext"
import { isArchitecturePath } from "@/lib/architecturePaths"
import { baseName } from "@/lib/paths"

const COLLAPSED_WIDTH = 0
const EXPANDED_WIDTH = 280

/** Drops tree rows whose deletion is pending, at any depth; a hidden folder hides its children. */
function hidePending<T>(items: FileTreeItem<T>[], hidden: (item: FileTreeItem<T>) => boolean): FileTreeItem<T>[] {
  return items.filter((item) => !hidden(item)).map((item) => item.children
    ? { ...item, children: hidePending(item.children, hidden) }
    : item)
}

interface SidebarProps {
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
  /** Open tabs section, pinned above the settings entry in both app modes. */
  tabs?: React.ReactNode
  /** Settings entry pinned to the sidebar bottom; receives the collapsed state. */
  renderSettings?: () => React.ReactNode
  appMode: AppSurface
  onAppModeChange: (mode: AppSurface) => void
}

export function Sidebar({
  onFileSelect,
  onRootsChanged,
  onDocumentsChanged,
  activeCanvasPath,
  onCanvasSelect,
  onCanvasDeleted,
  activeArchitecture,
  onArchitectureSelect,
  tabs,
  renderSettings,
  appMode,
  onAppModeChange,
}: SidebarProps) {
  const { collapsed } = useSidebar()

  const {
    branchMeta,
    visibleRootFiles: rootFiles,
    historiesLoading,
    refreshAll,
  } = useBranches()

  const {
    projects,
    activeProject,
    openProject,
    closeProject,
    createProject,
    deleteProject,
  } = useProject()

  const { openMemoryViewer } = useMemoryViewer()
  const { schedule, pending } = useUndoDelete()

  const sidebarWidth = collapsed ? COLLAPSED_WIDTH : EXPANDED_WIDTH
  const animateWidth = useMemo(() => ({ width: sidebarWidth }), [sidebarWidth])


  // Selection primitives, one per tree, so the trees behave identically yet stay
  // independent (collapsing a project never touches a file folder).
  // Projects: accordion sourced from activeProject; opening loads the project
  // context, closing tears it down.
  const projectsController = useFileTree({
    accordion: true,
    activeId: activeProject,
    onActivate: openProject,
    onDeactivate: closeProject,
  })
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

  // Refetch on active canvas change to pick up newly created or renamed canvases
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

  // A file-root root is keyed by its absolute path.
  const deleteFileRoot = (path: string) => schedule(`root:${path}`, baseName(path), async () => {
    const result = await removeFileRoot(path)
    setRootTreeData(result.tree)
    onRootsChanged?.()
  })

  const fileItems = useMemo(() => hidePending(buildFileItems(rootTreeData),
    (item) => pending(`root:${item.id}`)), [rootTreeData, pending])
  const projectItems = useMemo(() =>
    // Projects tree rows: file root rows are projects, elements are chats.
    hidePending(buildProjectItems(projects, activeProject, branchMeta),
      (item) => pending(`${item.type === "root" ? "project" : "history"}:${item.id}`)),
    [projects, activeProject, branchMeta, pending],
  )
  const historyItems = useMemo(() =>
    hidePending(buildHistoryItems(rootFiles), (item) => pending(`history:${item.id}`)),
    [rootFiles, pending],
  )
  // A history is any chat that is not in a project, listed after the projects like loose canvases.
  const chatItems = useMemo(() => [...projectItems, ...historyItems], [projectItems, historyItems])

  const renderChatElement = useCallback((item: FileTreeItem<string>, depth: number) => (
    <ChatBranchItem
      file={typeof item.data === "string" ? item.data : item.id}
      label={item.label}
      depth={depth}
    />
  ), [])

  return (
    <motion.aside
      initial={false}
      animate={animateWidth}
      transition={{ duration: 0.25, ease: "easeInOut" }}
      className={cn(
        "relative shrink-0 flex flex-col transition-[margin,background-color,box-shadow,border-color] duration-200",
        collapsed
          ? "border-0"
          : "m-3 mr-0 rounded-2xl border border-divider/50 bg-surface shadow-[var(--shadow-lg)]",
      )}
    >
      {!collapsed && (
        <div className="group flex items-center px-2 h-12 shrink-0">
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
                className="shrink-0 rounded-md p-1 text-ink-subtle opacity-0 group-hover:opacity-100 hover:bg-surface-elevated hover:text-ink transition-all"
              >
                <Brain className="h-4 w-4" />
              </button>
            )}
          </div>
        </div>
      )}

      {!collapsed && (
      <>
      <div className="flex-1 overflow-y-auto py-3 flex flex-col gap-2">
        {appMode === "canvas" ? (
          <div className="px-2">
            <FileTree<CanvasData>
              sectionIcon={Lightbulb}
              sectionTitle="Projects"
              items={canvasItems}
              open={canvasOpen}
              onOpenChange={setCanvasOpen}
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
            items={chatItems}
            loading={historiesLoading}
            plusTitle="New project"
            controller={projectsController}
            onRootDelete={(id) => schedule(`project:${id}`, id, () => deleteProject(id))}
            onAddRoot={(name) => createProject(name)}
            renderBranchActions={(item) => (
              <button aria-label={`Open memory for ${item.label}`}
                onClick={(e) => { e.stopPropagation(); openMemoryViewer({ scope: "project", projectSlug: item.id }) }}
                className="p-0.5 rounded text-ink-faint hover:text-ink transition-colors"
              >
                <Brain className="h-3 w-3" />
              </button>
            )}
            onMoveElement={async (elementId: string, targetId: string | null) => {
              // Dropping on a project files the chat there, anywhere else makes it a plain history again.
              const slug = projects.find((p) => p.slug === targetId)?.slug ?? ""
              await moveHistoryItem(elementId, slug)
              await refreshAll()
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
            sectionIcon={FolderOpen}
            sectionTitle="Files"
            items={fileItems}
            open={filesOpen}
            onOpenChange={setFilesOpen}
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
            onRootDelete={deleteFileRoot}
          />
        </div>
        </>
        )}
      </div>

      {tabs && <div className="mx-2 max-h-[35vh] shrink-0 overflow-y-auto border-t border-divider/40 pt-2">{tabs}</div>}
      {renderSettings && <div className="shrink-0 p-2">{renderSettings()}</div>}
      </>
      )}
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
      children: buildFileItems(n.children ?? []),
    }
  })
}

function buildProjectItems(
  projects: ProjectListItem[],
  activeProject: string | null,
  branchMeta: Record<string, BranchMeta>,
): FileTreeItem<string>[] {
  return projects.map((project) => {
    const dirPrefix = `${project.slug}/`
    const tabByName = new Map((project.tabs ?? []).map(t => [t.filename, t]))

    const items: FileTreeItem<string>[] = []
    for (const [key, meta] of Object.entries(branchMeta)) {
      // Top-level chats only: branches, project.json and memory/ docs are not listed
      if (!key.startsWith(dirPrefix) || key === `${project.slug}/project.json`) continue
      if (meta.parent_id) continue
      const filename = key.slice(dirPrefix.length)
      if (filename.startsWith("memory/")) continue
      const tab = tabByName.get(filename)
      const label = tab ? (tab.name ?? tab.title ?? filename.replace(".json", "")) : filename.replace(".json", "")
      items.push({ id: key, label, type: "element", data: key })
    }

    return {
      id: project.slug,
      label: project.name,
      type: "root",
      data: project.slug,
      icon: Lightbulb,
      isActive: activeProject === project.slug,
      children: items,
    }
  })
}

function buildHistoryItems(rootFiles: string[]): FileTreeItem<string>[] {
  return rootFiles.map((file) => ({
    id: file,
    label: file.replace(".json", ""),
    type: "element" as const,
    data: file,
  }))
}
