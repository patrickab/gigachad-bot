"use client"

import { motion } from "framer-motion"
import { ChatHistoryManager } from "./ChatHistoryManager"
import { PanelLeftClose, PanelLeft } from "lucide-react"

interface SidebarProps {
  collapsed: boolean
  onToggle: () => void
  histories: Record<string, string[]>
  onHistoryLoad: (filename: string) => void
  onHistoryRefresh: () => void
}

export function Sidebar({
  collapsed,
  onToggle,
  histories,
  onHistoryLoad,
  onHistoryRefresh,
}: SidebarProps) {
  return (
    <motion.aside
      initial={false}
      animate={{ width: collapsed ? 40 : 280 }}
      transition={{ duration: 0.25, ease: "easeInOut" }}
      className="shrink-0 border-r border-zinc-800 bg-zinc-950 flex flex-col overflow-hidden"
    >
      <div className={`flex items-center px-2 py-2 border-b border-zinc-800 h-[60px] ${collapsed ? "justify-center" : "justify-end"}`}>
        <button
          onClick={onToggle}
          className="rounded-md p-1.5 text-zinc-500 hover:bg-zinc-900 hover:text-zinc-300 transition-colors"
        >
          {collapsed ? <PanelLeft className="h-4 w-4" /> : <PanelLeftClose className="h-4 w-4" />}
        </button>
      </div>
      {!collapsed && (
        <div className="flex-1 overflow-y-auto px-3 py-3 flex flex-col">
          <div className="flex-1" />
          <ChatHistoryManager histories={histories} onLoad={onHistoryLoad} onRefresh={onHistoryRefresh} />
        </div>
      )}
    </motion.aside>
  )
}
