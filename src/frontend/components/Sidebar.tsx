"use client"

import { useCallback, useEffect, useMemo, useState } from "react"
import { motion } from "framer-motion"
import { VaultTree, type VaultTreeItem } from "./VaultTree"
import { useVaultTree } from "@/hooks/useVaultTree"
import { Brain, Clock, FileText, FolderOpen, Lightbulb, LayoutDashboard, PanelLeftClose, PanelLeft, PenLine, Save, RotateCcw } from "lucide-react"
import { useMemoryViewer } from "@/contexts/MemoryViewerContext"
import { SidebarElement } from "./SidebarElement"
import { useProject } from "@/contexts/ProjectContext"
import { useBranches } from "@/contexts/BranchContext"
import { useSidebar, type AppSurface } from "@/contexts/SidebarContext"
import { useUndoDelete } from "@/contexts/UndoDeleteContext"
import type { BranchMeta, VaultNode, ProjectListItem, ProjectDocument } from "@/lib/types"
import { ChatBranchItem } from "./ChatBranchItem"
import { addFileVaultMountpoint, addFileVaultRoot, createDirectory, moveHistoryItem, fileVaultTree, removeFileVaultRoot, listNotes, listProjectDocuments, moveDocument, removeDocument, writeDocument, Vault } from "@/lib/api"

const COLLAPSED_WIDTH = 50
const EXPANDED_WIDTH = 280

/** Drops tree rows whose deletion is pending, at any depth; a hidden folder hides its children. */
function hidePending<T>(items: VaultTreeItem<T>[], hidden: (item: VaultTreeItem<T>) => boolean): VaultTreeItem<T>[] {
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
  onVaultSelect?: (path: string) => void
  onVaultsChanged?: () => void
  activeCanvasPath?: string | null
  onCanvasSelect?: (path: string, scope: string) => void
  onCanvasDeleted?: (path: string) => void
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
  onVaultSelect,
  onVaultsChanged,
  activeCanvasPath,
  onCanvasSelect,
  onCanvasDeleted,
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
  const projectsController = useVaultTree({
    accordion: true,
    activeId: activeProject,
    onActivate: openProject,
    onDeactivate: closeProject,
    // ponytail: mounted-vault ids are absolute paths; only project slugs join the accordion
    isAccordionId: (id) => !id.startsWith("/"),
  })
  const historiesController = useVaultTree({ storageKey: "expanded_history_folders" })
  const vaultsController = useVaultTree({ storageKey: "expanded_vault_folders" })
  const canvasController = useVaultTree({ storageKey: "expanded_canvas_folders" })

  const [canvasOpen, setCanvasOpen] = useState(true)
  const [canvasNotes, setCanvasNotes] = useState<ProjectDocument[]>([])
  const [projectCanvases, setProjectCanvases] = useState<Record<string, ProjectDocument[]>>({})

  const refreshCanvases = useCallback(async () => {
    const notes = await listNotes().catch(() => [])
    setCanvasNotes(notes.filter((d) => d.name.endsWith(".canvas")))
    const entries = await Promise.all(
      projects.map(async (p) => [p.slug, (await listProjectDocuments(p.slug).catch(() => [])).filter((d) => d.name.endsWith(".canvas"))] as const),
    )
    setProjectCanvases(Object.fromEntries(entries))
  }, [projects])

  useEffect(() => {
    if (appMode === "canvas") refreshCanvases()
  }, [appMode, activeCanvasPath, refreshCanvases])

  const canvasItems = useMemo(() =>
    hidePending(buildCanvasItems(canvasNotes, projects, projectCanvases, activeCanvasPath ?? null),
      (item) => !!item.data && pending(`canvas:${item.data.scope}:${item.data.path}`)),
    [canvasNotes, projects, projectCanvases, activeCanvasPath, pending],
  )

  // File vaults: roots maintained server-side in file-vault-roots.json; rendered
  // as a file tree. Clicking a file attaches it to the active chat as a live reference.
  const [vaultTreeData, setVaultTreeData] = useState<VaultNode[]>([])
  const [vaultsOpen, setVaultsOpen] = useState(false)

  useEffect(() => {
    let cancelled = false
    fileVaultTree()
      .then((r) => { if (!cancelled) setVaultTreeData(r.tree) })
      .catch(() => {})
    return () => { cancelled = true }
  }, [])

  // A file-vault root, standalone (Vaults) or mounted to a project (Projects), is keyed by its absolute path.
  const deleteVaultRoot = (path: string) => schedule(`vault:${path}`, path.split("/").pop() ?? path, async () => {
    const result = await removeFileVaultRoot(path)
    setVaultTreeData(result.tree)
    onVaultsChanged?.()
  })

  // Roots mounted to a project render inside that project's subtree, not here.
  const vaultItems = useMemo(() => hidePending(buildVaultItems(vaultTreeData.filter((node) => !node.project)),
    (item) => pending(`vault:${item.id}`)), [vaultTreeData, pending])
  const projectItems = useMemo(() =>
    // Projects tree rows: absolute ids are mounted vault roots, other vault rows are projects, elements are chats.
    hidePending(buildProjectItems(projects, activeProject, branchMeta, vaultTreeData),
      (item) => pending(`${item.id.startsWith("/") ? "vault" : item.type === "vault" ? "project" : "history"}:${item.id}`)),
    [projects, activeProject, branchMeta, vaultTreeData, pending],
  )
  const historyItems = useMemo(() =>
    hidePending(buildHistoryItems(rootFiles, histories, projects), (item) => pending(`history:${item.id}`)),
    [rootFiles, histories, projects, pending],
  )

  const renderChatElement = useCallback((item: VaultTreeItem<string>, depth: number) => {
    const data = typeof item.data === "string" ? item.data : item.id
    // ponytail: absolute path = file inside a mounted vault; chat files are always relative
    if (typeof data === "string" && data.startsWith("/")) {
      return (
        <div style={{ paddingLeft: 8 + depth * 16, paddingRight: 8 }}>
          <button
            onClick={() => onVaultSelect?.(data)}
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
  }, [onVaultSelect])

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
            <VaultTree<CanvasData>
              sectionIcon={PenLine}
              sectionTitle="Canvases"
              items={canvasItems}
              collapsed={collapsed}
              open={canvasOpen}
              onOpenChange={setCanvasOpen}
              onExpand={expandIfCollapsed}
              controller={canvasController}
              plusTitle="New canvas"
              onElementClick={(item) => item.data && onCanvasSelect?.(item.data.path, item.data.scope)}
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
          <VaultTree
            sectionIcon={Lightbulb}
            sectionTitle="Projects"
            items={projectItems}
            collapsed={collapsed}
            open={projectsOpen}
            onOpenChange={setProjectsOpen}
            onExpand={expandIfCollapsed}
            controller={projectsController}
            onVaultDelete={(id) => {
              if (id.startsWith("/")) deleteVaultRoot(id)
              else schedule(`project:${id}`, id, () => deleteProject(id))
            }}
            onAddVault={(name) => createProject(name)}
            onAddMountpoint={async (vaultId: string, path: string) => {
              // Project row → mount a FileVault to the project; mounted vault row → nested mountpoint.
              const r = vaultId.startsWith("/")
                ? await addFileVaultMountpoint(vaultId, path)
                : await addFileVaultRoot(path, vaultId)
              setVaultTreeData(r.tree)
              onVaultsChanged?.()
            }}
            mountpointPlaceholder="Vault filepath…"
            onDashboardClick={(vaultId) => {
              openProject(vaultId)
              setDashboardOpen(true)
            }}
            onMemoryClick={(vaultId) => openMemoryViewer({ scope: "project", projectSlug: vaultId })}
            onMoveElement={async (elementId: string, targetId: string | null) => {
              const slug = targetId && projects.find(p => p.slug === targetId)?.slug
              if (!slug) return
              await moveHistoryItem(elementId, slug)
              await onRefreshAll()
            }}
            renderElement={renderChatElement}
          />
        </div>

        <div className="px-2">
          <VaultTree
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
            onVaultDelete={(id: string) => {
              schedule(`history:${id}`, id.split("/").pop() ?? id, async () => {
                await new Vault(id).delete()
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
          <VaultTree
            sectionIcon={FolderOpen}
            sectionTitle="Vaults"
            items={vaultItems}
            collapsed={collapsed}
            open={vaultsOpen}
            onOpenChange={setVaultsOpen}
            onExpand={expandIfCollapsed}
            plusTitle="Add vault root"
            controller={vaultsController}
            vaultPlaceholder="Vault filepath…"
            onElementClick={(item) => onVaultSelect?.((item.data as string) ?? item.id)}
            onAddVault={async (path: string) => {
              const r = await addFileVaultRoot(path)
              setVaultTreeData(r.tree)
              setVaultsOpen(true)
              onVaultsChanged?.()
            }}
            onAddMountpoint={async (vaultId: string, path: string) => {
              const r = await addFileVaultMountpoint(vaultId, path)
              setVaultTreeData(r.tree)
              onVaultsChanged?.()
            }}
            onVaultDelete={deleteVaultRoot}
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
}

function buildCanvasItems(
  canvasNotes: ProjectDocument[],
  projects: ProjectListItem[],
  projectCanvases: Record<string, ProjectDocument[]>,
  activeCanvasPath: string | null,
): VaultTreeItem<CanvasData>[] {
  const canvasElement = (doc: ProjectDocument, scope: string): VaultTreeItem<CanvasData> => ({
    id: doc.path,
    label: doc.name.replace(/\.canvas$/, ""),
    type: "element",
    icon: PenLine,
    data: { path: doc.path, scope },
    isActive: activeCanvasPath === doc.path,
  })

  const projectItems: VaultTreeItem<CanvasData>[] = projects
    .filter((p) => (projectCanvases[p.slug] ?? []).length > 0)
    .map((p) => ({
      id: p.slug,
      label: p.name,
      type: "vault" as const,
      icon: Lightbulb,
      children: (projectCanvases[p.slug] ?? []).map((doc) => canvasElement(doc, p.slug)),
    }))

  const rootItems = canvasNotes.map((doc) => canvasElement(doc, ""))
  return [...projectItems, ...rootItems]
}

function buildVaultItems(nodes: VaultNode[]): VaultTreeItem<string>[] {
  return nodes.map((n) => {
    if (n.type === "file") {
      return { id: n.path, label: n.name, type: "element" as const, data: n.path, icon: FileText }
    }
    const isVault = n.type === "vault"
    return {
      id: n.path,
      label: n.name,
      type: n.type, // "vault" | "folder"
      data: n.path,
      icon: isVault ? FolderOpen : undefined,
      mountable: isVault,
      mounted: isVault,
      children: buildVaultItems(n.children ?? []),
    }
  })
}

function buildProjectItems(
  projects: ProjectListItem[],
  activeProject: string | null,
  branchMeta: Record<string, BranchMeta>,
  vaultTree: VaultNode[],
): VaultTreeItem<string>[] {
  return projects.map((project) => {
    const dirPrefix = `${project.slug}/`
    const tabByName = new Map((project.tabs ?? []).map(t => [t.filename, t]))

    const items: VaultTreeItem<string>[] = []
    for (const [key, meta] of Object.entries(branchMeta)) {
      if (!key.startsWith(dirPrefix) || key === `${project.slug}/project.json`) continue
      if (meta.parent_id) continue
      const filename = key.slice(dirPrefix.length)
      if (filename.startsWith("memory/")) continue
      const tab = tabByName.get(filename)
      const label = tab ? (tab.name ?? tab.title ?? filename.replace(".json", "")) : filename.replace(".json", "")
      items.push({ id: key, label, type: "element", data: key })
    }

    // FileVault roots mounted to this project appear as vault nodes in its subtree.
    const mountedVaults = buildVaultItems(vaultTree.filter((n) => n.project === project.slug))

    return {
      id: project.slug,
      label: project.name,
      type: "vault",
      data: project.slug,
      icon: Lightbulb,
      mountable: true,
      isActive: activeProject === project.slug,
      children: [
        ...mountedVaults,
        {
          id: `${project.slug}/__dashboard__`,
          label: "Dashboard",
          type: "element" as const,
          icon: LayoutDashboard,
          isSystem: true,
          displayInVault: false,
          data: project.slug,
        },
        ...items,
      ].filter((child: VaultTreeItem<string>) => child.displayInVault !== false),
    }
  })
}

function buildHistoryItems(
  rootFiles: string[],
  histories: Record<string, string[]>,
  projects: ProjectListItem[],
): VaultTreeItem<string>[] {
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
