"use client"

import { createContext, useContext, useState, type ReactNode } from "react"

export type AppSurface = "chat" | "canvas"

const APP_MODE_KEY = "app_mode"

function loadAppMode(): AppSurface {
  if (typeof window === "undefined") return "chat"
  return window.localStorage.getItem(APP_MODE_KEY) === "canvas" ? "canvas" : "chat"
}

interface SidebarContextValue {
  collapsed: boolean
  toggleCollapsed: () => void
  projectsOpen: boolean
  setProjectsOpen: (open: boolean) => void
  historiesOpen: boolean
  setHistoriesOpen: (open: boolean) => void
  appMode: AppSurface
  setAppMode: (mode: AppSurface) => void
}

const SidebarContext = createContext<SidebarContextValue | null>(null)

export function SidebarProvider({ children }: { children: ReactNode }) {
  const [collapsed, setCollapsed] = useState(true)
  const [projectsOpen, setProjectsOpen] = useState(true)
  const [historiesOpen, setHistoriesOpen] = useState(false)
  const [appMode, setAppModeState] = useState<AppSurface>(loadAppMode)

  const toggleCollapsed = () => setCollapsed((c) => !c)
  const setAppMode = (mode: AppSurface) => {
    setAppModeState(mode)
    try { window.localStorage.setItem(APP_MODE_KEY, mode) } catch {}
  }

  return (
    <SidebarContext.Provider value={{ collapsed, toggleCollapsed, projectsOpen, setProjectsOpen, historiesOpen, setHistoriesOpen, appMode, setAppMode }}>
      {children}
    </SidebarContext.Provider>
  )
}

export function useSidebar() {
  const ctx = useContext(SidebarContext)
  if (!ctx) throw new Error("useSidebar must be inside SidebarProvider")
  return ctx
}
