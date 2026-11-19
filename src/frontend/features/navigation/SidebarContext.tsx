"use client"

import { createContext, useState, type ReactNode } from "react"
import { useRequiredContext } from "@/hooks/useRequiredContext"

interface SidebarContextValue {
  collapsed: boolean
  toggleCollapsed: () => void
  /** Open width in px, dragged on its inner edge. */
  width: number
  setWidth: (width: number) => void
}

const SidebarContext = createContext<SidebarContextValue | null>(null)

/** Owns the sidebar collapsed flag, closed by default, and its width, shared by every tab. */
export function SidebarProvider({ children }: { children: ReactNode }) {
  const [collapsed, setCollapsed] = useState(true)
  const [width, setWidth] = useState(280)

  const toggleCollapsed = () => setCollapsed((c) => !c)

  return (
    <SidebarContext.Provider value={{ collapsed, toggleCollapsed, width, setWidth }}>
      {children}
    </SidebarContext.Provider>
  )
}

export function useSidebar() {
  return useRequiredContext(SidebarContext, "useSidebar must be inside SidebarProvider")
}
