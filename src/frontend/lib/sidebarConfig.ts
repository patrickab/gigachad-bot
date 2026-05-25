import { PanelLeftClose, PanelLeft, Save, RotateCcw } from "lucide-react"

export interface SidebarActionConfig {
  onToggleCollapse: () => void
  onSave: () => void
  onReset: () => void
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
      id: "save-chat",
      icon: Save,
      title: "Save Chat",
      onClick: actions.onSave,
    },
    {
      id: "reset-chat",
      icon: RotateCcw,
      title: "Reset History",
      onClick: actions.onReset,
    },
  ]
}
