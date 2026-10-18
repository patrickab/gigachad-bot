"use client"

import { createContext, useContext } from "react"
import type { TabConfig } from "@/contexts/SettingsContext"

/** Top-level workspace mode of a tab. */
export type AppSurface = "chat" | "canvas" | "document"

export interface WorkspaceItem {
  kind: "canvas" | "architecture" | "document" | "diff"
  /** A `diff` item holds the mounted directory it compares against HEAD. */
  path: string
  projectSlug: string | null
  rootFile?: boolean
}

export interface Tab {
  id: string
  name: string | null
  chatId: string
  historyFile: string | null
  title: string | null
  config: TabConfig
  appMode: AppSurface
  projectSlug: string | null
  workspaceItem: WorkspaceItem | null
}

/** Actions bound to one tab; each is stable for the tab's lifetime. */
export interface TabActions {
  /** Describe tab activity and optionally show a loading spinner in the workspace switcher. */
  setModeLabel: (label: string, loading?: boolean) => void
  updateConfig: (config: Partial<TabConfig>) => void
  setAppMode: (mode: AppSurface) => void
  setWorkspaceItem: (item: WorkspaceItem | null) => void
  setHistoryFile: (historyFile: string | null) => void
  setTitle: (title: string | null) => void
}

export interface TabContextValue extends TabActions {
  tab: Tab
  /** Tabs stay mounted in the background, so key listeners must check this to fire for the frontmost tab only. */
  isActive: boolean
}

export const TabContext = createContext<TabContextValue | null>(null)

/** The enclosing tab. Throws outside a `TabManager`. */
export function useTab(): TabContextValue {
  const value = useContext(TabContext)
  if (!value) throw new Error("useTab must be used inside a tab")
  return value
}

/** Whether the enclosing tab is frontmost; true outside a tab. */
export function useTabActive(): boolean {
  return useContext(TabContext)?.isActive ?? true
}
