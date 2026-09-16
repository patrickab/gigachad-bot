import { ElementType } from "react"

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
      className={`w-full flex items-center p-2 rounded-md transition-colors ${
        isActive
          ? "bg-surface-elevated text-ink"
          : "text-ink-muted hover:bg-surface-elevated/50 hover:text-ink"
      } ${collapsed ? "justify-center" : "justify-start gap-3"} ${className}`}
    >
      <Icon className="h-4 w-4 shrink-0" />
      {!collapsed && title && (
        <span className="text-sm font-medium truncate">{title}</span>
      )}
    </button>
  )
}
