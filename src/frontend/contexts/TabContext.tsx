"use client"

import { createContext, useContext } from "react"
import type { TabConfig } from "@/contexts/SettingsContext"

/** Top-level workspace mode of a tab. */
export type AppSurface = "chat" | "canvas"

export interface Tab {
  id: string
  name: string | null
  chatId: string
  historyFile: string | null
  title: string | null
  config: TabConfig
  appMode: AppSurface
}

/** Actions bound to one tab; each is stable for the tab's lifetime. */
export interface TabActions {
  /** `loading` shows a spinner on the tab row; the `"\0"` label hides the tab label so the command input can take its slot. */
  setModeLabel: (label: string, loading?: boolean) => void
  updateConfig: (config: Partial<TabConfig>) => void
  setAppMode: (mode: AppSurface) => void
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
