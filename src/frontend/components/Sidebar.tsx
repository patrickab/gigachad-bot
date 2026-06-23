"use client"

import { useCallback, useEffect, useMemo } from "react"
import { motion } from "framer-motion"
import { VaultTree, type VaultTreeItem } from "./VaultTree"
import { useVaultTree } from "@/hooks/useVaultTree"
import { Brain, FolderKanban, LayoutDashboard, Clock, PanelLeftClose, PanelLeft, Save, RotateCcw } from "lucide-react"
import { useMemoryViewer } from "@/contexts/MemoryViewerContext"
import { SidebarElement } from "./SidebarElement"
import { useProject } from "@/contexts/ProjectContext"
import { useBranches } from "@/contexts/BranchContext"
import { useSidebar } from "@/contexts/SidebarContext"
import type { BranchMeta, ProjectListItem } from "@/lib/types"
import { ChatBranchItem } from "./ChatBranchItem"
import { createDirectory, moveHistoryItem, Vault } from "@/lib/api"

const COLLAPSED_WIDTH = 50
const EXPANDED_WIDTH = 280

interface SidebarProps {
  onOpenChat: (filename: string, qaIndex?: number) => void
  onRefreshAll: () => Promise<void>
  onSave: () => void
  onReset: () => void
  onMerge?: (childFile: string) => Promise<void>
  onCascadeDelete?: (filename: string) => Promise<void>
}

export function Sidebar({
  onOpenChat,
  onRefreshAll,
  onSave,
  onReset,
  onMerge,
  onCascadeDelete,
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

  const handleTimelineDelete = useCallback(async (file: string) => {
    await onCascadeDelete?.(file)
  }, [onCascadeDelete])

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
  })
  const historiesController = useVaultTree({ storageKey: "expanded_history_folders" })

  const projectItems = useMemo(() =>
    buildProjectItems(projects, activeProject, branchMeta),
    [projects, activeProject, branchMeta],
  )

  const historyItems = useMemo(() =>
    buildHistoryItems(rootFiles, histories, projects),
    [rootFiles, histories, projects],
  )

  const renderChatElement = useCallback((item: VaultTreeItem<string>, depth: number) => (
    <ChatBranchItem
      file={(item.data as string) ?? item.id}
      label={item.label}
      depth={depth}
    />
  ), [])

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
            <span className="text-base font-semibold tracking-tight text-ink truncate">GigaChat Bot</span>
            <button
              onClick={() => openMemoryViewer({ scope: "global" })}
              aria-label="Open global memory profile"
              className="shrink-0 rounded-md p-1 text-ink-subtle opacity-0 group-hover:opacity-100 hover:bg-surface hover:text-ink transition-all"
            >
              <Brain className="h-4 w-4" />
            </button>
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
        <div className="px-2">
          <VaultTree
            sectionIcon={FolderKanban}
            sectionTitle="Projects"
            count={projects.length}
            items={projectItems}
            collapsed={collapsed}
            open={projectsOpen}
            onOpenChange={setProjectsOpen}
            onExpand={expandIfCollapsed}
            controller={projectsController}
            onVaultDelete={(id) => deleteProject(id)}
            onAddVault={(name) => createProject(name)}
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
            onVaultDelete={async (id: string) => {
              await new Vault(id).delete()
              await onRefreshAll()
            }}
            onMoveElement={async (elementId: string, targetId: string | null) => {
              await moveHistoryItem(elementId, targetId ?? "")
              await onRefreshAll()
            }}
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
      </div>
    </motion.aside>
  )
}

function buildProjectItems(
  projects: ProjectListItem[],
  activeProject: string | null,
  branchMeta: Record<string, BranchMeta>,
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

    return {
      id: project.slug,
      label: project.name,
      type: "vault",
      data: project.slug,
      isActive: activeProject === project.slug,
      children: [
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