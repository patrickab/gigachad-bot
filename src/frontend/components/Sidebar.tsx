"use client"

import { motion } from "framer-motion"
import { ChatHistoryManager } from "./ChatHistoryManager"
import { SidebarElement } from "./SidebarElement"
import { getSidebarConfig } from "@/lib/sidebarConfig"

interface SidebarProps {
  collapsed: boolean
  onToggle: () => void
  histories: Record<string, string[]>
  onHistoryLoad: (filename: string) => void
  onHistoryRefresh: () => void
  onSave: () => void
  onReset: () => void
}

export function Sidebar({
  collapsed,
  onToggle,
  histories,
  onHistoryLoad,
  onHistoryRefresh,
  onSave,
  onReset,
}: SidebarProps) {
  const sidebarItems = getSidebarConfig({
    onToggleCollapse: onToggle,
    onSave,
    onReset,
    collapsed,
  })

  // Separate the top-level controls (e.g. toggle) from main items
  const toggleItem = sidebarItems.find(item => item.id === "toggle-collapse")
  const mainItems = sidebarItems.filter(item => item.id !== "toggle-collapse")

  return (
    <motion.aside
      initial={false}
      animate={{ width: collapsed ? 50 : 280 }}
      transition={{ duration: 0.25, ease: "easeInOut" }}
      className="shrink-0 border-r border-zinc-800 bg-zinc-950 flex flex-col overflow-hidden"
    >
      <div className={`flex items-center px-2 py-2 border-b border-zinc-800 h-[60px] ${collapsed ? "justify-center" : "justify-between"}`}>
        {!collapsed && (
          <span className="text-sm font-semibold text-zinc-200 px-2">Gigachad</span>
        )}
        {toggleItem && (
          <SidebarElement
            {...toggleItem}
            collapsed={true} // Toggle button itself is always icon-only
            className="w-auto p-1.5 hover:bg-zinc-900"
          />
        )}
      </div>
      
      <div className="flex-1 overflow-y-auto py-3 flex flex-col gap-2">
        <div className="px-2">
          <ChatHistoryManager 
            histories={histories} 
            onLoad={onHistoryLoad} 
            onRefresh={onHistoryRefresh} 
            collapsed={collapsed}
            onExpand={() => {
              if (collapsed) onToggle()
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
        
        <div className="flex-1" />
      </div>
    </motion.aside>
  )
}
