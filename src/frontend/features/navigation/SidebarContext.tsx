"use client"

import { createContext, useState, type ReactNode } from "react"
import { useRequiredContext } from "@/hooks/useRequiredContext"

interface SidebarContextValue {
  collapsed: boolean
  toggleCollapsed: () => void
}

const SidebarContext = createContext<SidebarContextValue | null>(null)

/** Owns the sidebar collapsed flag, closed by default. */
export function SidebarProvider({ children }: { children: ReactNode }) {
  const [collapsed, setCollapsed] = useState(true)

  const toggleCollapsed = () => setCollapsed((c) => !c)

  return (
    <SidebarContext.Provider value={{ collapsed, toggleCollapsed }}>
      {children}
    </SidebarContext.Provider>
  )
}

export function useSidebar() {
  return useRequiredContext(SidebarContext, "useSidebar must be inside SidebarProvider")
}
