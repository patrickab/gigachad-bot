"use client"

import { motion } from "framer-motion"
import { ProjectTree } from "./ProjectTree"
import { HistoryTree } from "./HistoryTree"
import { SidebarElement } from "./SidebarElement"
import { getSidebarConfig } from "@/lib/sidebarConfig"
import { useProject } from "@/contexts/ProjectContext"
import { archiveChatHistory, createDirectory, moveHistoryItem, Element, Vault } from "@/lib/api"

const COLLAPSED_WIDTH = 50
const EXPANDED_WIDTH = 280

interface SidebarProps {
  collapsed: boolean
  onToggle: () => void
  rootFiles: string[]
  histories: Record<string, string[]>
  historiesLoading?: boolean
  onHistoryLoad: (filename: string) => void
  onHistoryRefresh: () => void
  onSave: () => void
  onReset: () => void
  projectsOpen: boolean
  onProjectsOpenChange: (open: boolean) => void
  historiesOpen: boolean
  onHistoriesOpenChange: (open: boolean) => void
}

export function Sidebar({
  collapsed,
  onToggle,
  rootFiles,
  histories,
  historiesLoading,
  onHistoryLoad,
  onHistoryRefresh,
  onSave,
  onReset,
  projectsOpen,
  onProjectsOpenChange,
  historiesOpen,
  onHistoriesOpenChange,
}: SidebarProps) {
  const {
    projects,
    activeProject,
    openProject,
    closeProject,
    createProject,
    deleteProject,
    setDashboardOpen,
  } = useProject()

  const expandIfCollapsed = () => { if (collapsed) onToggle() }

  const sidebarItems = getSidebarConfig({
    onToggleCollapse: onToggle,
    onSave,
    onReset,
    collapsed,
  })

  const toggleItem = sidebarItems.find(item => item.id === "toggle-collapse")
  const mainItems = sidebarItems.filter(
    item => item.id !== "toggle-collapse" && item.id !== "save-chat" && item.id !== "reset-chat",
  )

  return (
    <motion.aside
      initial={false}
      animate={{ width: collapsed ? COLLAPSED_WIDTH : EXPANDED_WIDTH }}
      transition={{ duration: 0.25, ease: "easeInOut" }}
      className="shrink-0 border-r border-zinc-800 bg-zinc-950 flex flex-col overflow-hidden"
    >
      <div className="flex items-center justify-between px-2 py-2 border-b border-zinc-800 h-[60px]">
        {!collapsed && (
          <span className="text-sm font-semibold text-zinc-200 px-2">GigaChat Bot</span>
        )}
        <div className={collapsed ? "flex justify-center w-full" : "flex justify-end"}>
          {toggleItem && (
            <SidebarElement
              {...toggleItem}
              collapsed={collapsed}
              className="w-auto p-1.5 hover:bg-zinc-900"
            />
          )}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto py-3 flex flex-col gap-2">
        <div className="px-2">
          <ProjectTree
            projects={projects}
            activeProject={activeProject}
            collapsed={collapsed}
            open={projectsOpen}
            onOpenChange={onProjectsOpenChange}
            onExpand={expandIfCollapsed}
            onVaultClick={(id) => {
              if (activeProject === id) {
                closeProject()
              } else {
                openProject(id)
              }
            }}
            onVaultDelete={(id) => deleteProject(id)}
            onAddVault={(name) => createProject(name)}
            onDashboardClick={(vaultId) => {
              openProject(vaultId)
              setDashboardOpen(true)
            }}
            onElementClick={(item) => {
              if (item.data && !item.isSystem) onHistoryLoad(item.data)
            }}
          />
        </div>

        <div className="px-2">
          <HistoryTree
            rootFiles={rootFiles}
            histories={histories}
            projects={projects}
            collapsed={collapsed}
            open={historiesOpen}
            onOpenChange={onHistoriesOpenChange}
            onExpand={expandIfCollapsed}
            plusTitle="New folder"
            onElementClick={(item) => {
              if (item.data) onHistoryLoad(item.data)
            }}
            onElementArchive={(item) => {
              if (item.data) archiveChatHistory(item.data).then(onHistoryRefresh)
            }}
            onElementDelete={(item) => {
              if (item.data) new Element(item.data).delete().then(onHistoryRefresh)
            }}
            onAddFolder={async (parentId: string | null, name: string) => {
              await createDirectory(parentId ?? "", name)
              onHistoryRefresh()
              onHistoriesOpenChange(true)
            }}
            onVaultDelete={async (id: string) => {
              await new Vault(id).delete()
              onHistoryRefresh()
            }}
            onMoveElement={async (elementId: string, targetId: string | null) => {
              await moveHistoryItem(elementId, targetId ?? "")
              onHistoryRefresh()
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
