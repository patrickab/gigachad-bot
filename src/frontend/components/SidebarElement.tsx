import { ElementType } from "react"
import { motion } from "framer-motion"

interface SidebarElementProps {
  icon: ElementType
  title?: string
  onClick: () => void
  collapsed: boolean
  isActive?: boolean
  className?: string
}

export function SidebarElement({
  icon: Icon,
  title,
  onClick,
  collapsed,
  isActive,
  className = "",
}: SidebarElementProps) {
  return (
    <button
      onClick={onClick}
      title={collapsed ? title : undefined}
      className={`w-full flex items-center p-2 rounded-md transition-colors ${
        isActive
          ? "bg-zinc-800 text-zinc-100"
          : "text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-200"
      } ${collapsed ? "justify-center" : "justify-start gap-3"} ${className}`}
    >
      <Icon className="h-4 w-4 shrink-0" />
      {!collapsed && title && (
        <span className="text-sm font-medium truncate">{title}</span>
      )}
    </button>
  )
}

export function ChevronToggle({ open }: { open: boolean }) {
  return (
    <motion.span animate={{ rotate: open ? 180 : 0 }} transition={{ duration: 0.2 }}>
      <svg width="12" height="12" viewBox="0 0 12 12" fill="none" className="text-zinc-500">
        <path d="M4 2L8 6L4 10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
    </motion.span>
  )
}

