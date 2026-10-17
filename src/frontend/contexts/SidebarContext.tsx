"use client"

import { createContext, useContext, useState, type ReactNode } from "react"

/** Top-level workspace mode of a tab. */
export type AppSurface = "chat" | "canvas"

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
  const ctx = useContext(SidebarContext)
  if (!ctx) throw new Error("useSidebar must be inside SidebarProvider")
  return ctx
}
