"use client"

import { motion } from "framer-motion"
import { LayoutDashboard } from "lucide-react"
import { ChatHistoryManager } from "./ChatHistoryManager"
import { ProjectManager } from "./ProjectManager"
import { SidebarElement } from "./SidebarElement"
import { getSidebarConfig } from "@/lib/sidebarConfig"
import { useProject } from "@/contexts/ProjectContext"

const COLLAPSED_WIDTH = 50
const EXPANDED_WIDTH = 280

interface SidebarProps {
  collapsed: boolean
  onToggle: () => void
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
  const { activeProject, setDashboardOpen } = useProject()

  const expandIfCollapsed = () => { if (collapsed) onToggle() }

  const sidebarItems = getSidebarConfig({
    onToggleCollapse: onToggle,
    onSave,
    onReset,
    collapsed,
  })

  const toggleItem = sidebarItems.find(item => item.id === "toggle-collapse")
  const mainItems = sidebarItems.filter(item => item.id !== "toggle-collapse" && item.id !== "save-chat" && item.id !== "reset-chat")

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
          <ProjectManager
            collapsed={collapsed}
            open={projectsOpen}
            onOpenChange={onProjectsOpenChange}
            onExpand={expandIfCollapsed}
          />
        </div>

        <div className="px-2">
          <ChatHistoryManager
            histories={histories}
            historiesLoading={historiesLoading}
            onLoad={onHistoryLoad}
            onRefresh={onHistoryRefresh}
            collapsed={collapsed}
            open={historiesOpen}
            onOpenChange={onHistoriesOpenChange}
            onExpand={expandIfCollapsed}
          />
        </div>

        {activeProject && (
          <div className="px-2">
            <SidebarElement
              icon={LayoutDashboard}
              title="Dashboard"
              collapsed={collapsed}
              onClick={() => setDashboardOpen(true)}
              isActive={false}
            />
          </div>
        )}

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