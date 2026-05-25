import { ElementType } from "react"

interface SidebarElementProps {
  id: string
  icon: ElementType
  title?: string
  onClick: () => void
  collapsed: boolean
  isActive?: boolean
  className?: string
}

export function SidebarElement({
  id,
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
