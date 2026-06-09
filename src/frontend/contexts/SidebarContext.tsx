"use client"

import { createContext, useContext, useState, type ReactNode } from "react"

interface SidebarContextValue {
  collapsed: boolean
  toggleCollapsed: () => void
  projectsOpen: boolean
  setProjectsOpen: (open: boolean) => void
  historiesOpen: boolean
  setHistoriesOpen: (open: boolean) => void
}

const SidebarContext = createContext<SidebarContextValue | null>(null)

export function SidebarProvider({ children }: { children: ReactNode }) {
  const [collapsed, setCollapsed] = useState(true)
  const [projectsOpen, setProjectsOpen] = useState(true)
  const [historiesOpen, setHistoriesOpen] = useState(false)

  const toggleCollapsed = () => setCollapsed((c) => !c)

  return (
    <SidebarContext.Provider value={{ collapsed, toggleCollapsed, projectsOpen, setProjectsOpen, historiesOpen, setHistoriesOpen }}>
      {children}
    </SidebarContext.Provider>
  )
}

export function useSidebar() {
  const ctx = useContext(SidebarContext)
  if (!ctx) throw new Error("useSidebar must be inside SidebarProvider")
  return ctx
}