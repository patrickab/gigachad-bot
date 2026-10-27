"use client"

import { createContext, useContext } from "react"
import { useRequiredContext } from "@/hooks/useRequiredContext"
import type { TabConfig } from "@/contexts/SettingsContext"
import type { OmpView } from "@/lib/domain/types"

/** Top-level workspace mode of a tab. */
export type AppSurface = "chat" | "canvas" | "document" | "omp" | "factory"

export interface WorkspaceItem {
  kind: "canvas" | "architecture" | "document" | "diff" | "omp" | "factory"
  /** A `diff` item holds the mounted directory it compares against HEAD, an `omp` item its workspace directory ("" for settings), a `factory` item `<repo>` for its board or `<repo>/<task id>`. */
  path: string
  projectSlug: string | null
  rootFile?: boolean
  /** What an `omp` item shows in omp-web's frame. */
  omp?: OmpView
}

/** Surface that shows `item`. */
function surfaceOf(item: WorkspaceItem): AppSurface {
  if (item.kind === "omp") return "omp"
  if (item.kind === "factory") return "factory"
  return item.kind === "document" || item.kind === "diff" ? "document" : "canvas"
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

/** The tab's workspace item when it is of `kind`, else null. */
export function itemOf<K extends WorkspaceItem["kind"]>(tab: Tab, kind: K): (WorkspaceItem & { kind: K }) | null {
  const item = tab.workspaceItem
  return item?.kind === kind ? item as WorkspaceItem & { kind: K } : null
}

/** Tab fields that show `item` on its surface, in its project. */
export function showingItem(item: WorkspaceItem): Pick<Tab, "workspaceItem" | "projectSlug" | "appMode"> {
  return { workspaceItem: item, projectSlug: item.projectSlug, appMode: surfaceOf(item) }
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
  return useRequiredContext(TabContext, "useTab must be used inside a tab")
}

/** Whether the enclosing tab is frontmost; true outside a tab. */
export function useTabActive(): boolean {
  return useContext(TabContext)?.isActive ?? true
}
