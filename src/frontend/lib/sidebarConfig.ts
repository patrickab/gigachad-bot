import { PanelLeftClose, PanelLeft, PlusCircle } from "lucide-react"

export interface SidebarActionConfig {
  onToggleCollapse: () => void
  onNewChat: () => void
  collapsed: boolean
}

export const getSidebarConfig = (actions: SidebarActionConfig) => {
  return [
    {
      id: "toggle-collapse",
      icon: actions.collapsed ? PanelLeft : PanelLeftClose,
      title: actions.collapsed ? "Expand Sidebar" : "Collapse Sidebar",
      onClick: actions.onToggleCollapse,
    },
    {
      id: "new-chat",
      icon: PlusCircle,
      title: "New Chat",
      onClick: actions.onNewChat,
    },
  ]
}
