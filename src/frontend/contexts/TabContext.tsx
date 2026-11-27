"use client"

import { createContext, useContext } from "react"
import { useRequiredContext } from "@/hooks/useRequiredContext"
import type { TabConfig } from "@/contexts/SettingsContext"
import type { LearnView, PiView } from "@/lib/domain/types"
import { opensInWorkbench } from "@/lib/domain/workbench"

/** Top-level workspace mode of a tab. */
export type AppSurface = "chat" | "canvas" | "document" | "pi" | "workbench" | "factory" | "learn"

export interface WorkspaceItem {
  kind: "canvas" | "architecture" | "scratch" | "document" | "diff" | "pi" | "workbench" | "factory" | "learn"
  /** A `scratch` item is the unsaved canvas, its path "". A `diff` item holds the mounted directory it compares against HEAD, a `pi` item its repo name, a `workbench` item its file, a `factory` item `<repo>` for its board or `<repo>/<task id>`, a `learn` item "". */
  path: string
  projectSlug: string | null
  /** The path a `workbench` item was opened under, kept when the file is renamed so its editor session goes on. */
  sessionKey?: string
  /** The LaTeX project folder a `workbench` item shows: opening another file of it switches this tab's buffer. */
  texFolder?: string
  /** The Pi `document` chat session a `workbench` item or a notebook `document` item shows. */
  chat?: { sessionId: string }
  /** What a `pi` item shows: its repo and session. */
  pi?: PiView
  /** What a `learn` item shows in LearnHarness's frame. */
  learn?: LearnView
}

/** Surface that shows `item`. */
function surfaceOf(item: WorkspaceItem): AppSurface {
  if (item.kind === "pi") return "pi"
  if (item.kind === "workbench") return "workbench"
  if (item.kind === "factory") return "factory"
  if (item.kind === "learn") return "learn"
  return item.kind === "document" || item.kind === "diff" ? "document" : "canvas"
}

/** A new Pi session in `directory`: any folder, its absolute path is the session's `repo`. */
export function newPiItem(directory: string, projectSlug: string | null): WorkspaceItem {
  const repo = directory.replace(/\/+$/, "") || "/"
  return { kind: "pi", path: repo.split("/").pop() || repo, projectSlug, pi: { sessionId: null, repo } }
}

/** The item that shows the file `path`: in the workbench when it is available and the file is one of its kinds, else in the document editor. */
export function fileItem(path: string, projectSlug: string | null, workbench: boolean): WorkspaceItem {
  return { kind: workbench && opensInWorkbench(path) ? "workbench" : "document", path, projectSlug }
}

export interface Tab {
  id: string
  chatId: string
  historyFile: string | null
  title: string | null
  config: TabConfig
  projectSlug: string | null
  workspaceItem: WorkspaceItem | null
  /** The chat covers the held item. */
  itemHidden: boolean
}

/** The item the tab shows; null when it shows its chat (no item, or the item is hidden behind the chat). */
export function shownItem(tab: Tab): WorkspaceItem | null {
  return tab.itemHidden ? null : tab.workspaceItem
}

/** The surface the tab shows: its shown item's, else "chat". */
export function tabSurface(tab: Tab): AppSurface {
  const item = shownItem(tab)
  return item ? surfaceOf(item) : "chat"
}

/** A new unsaved canvas of `projectSlug`. */
export function scratchItem(projectSlug: string | null): WorkspaceItem {
  return { kind: "scratch", path: "", projectSlug }
}

/** The tab's shown item when it is of `kind`, else null. */
export function itemOf<K extends WorkspaceItem["kind"]>(tab: Tab, kind: K): (WorkspaceItem & { kind: K }) | null {
  const item = shownItem(tab)
  return item?.kind === kind ? item as WorkspaceItem & { kind: K } : null
}

/** Tab fields that show `item` on its surface, in its project. */
export function showingItem(item: WorkspaceItem): Pick<Tab, "workspaceItem" | "projectSlug" | "itemHidden"> {
  return { workspaceItem: item, projectSlug: item.projectSlug, itemHidden: false }
}

/** Actions bound to one tab; each is stable for the tab's lifetime. */
export interface TabActions {
  /** Describe tab activity and optionally show a loading spinner in the workspace switcher. */
  setModeLabel: (label: string, loading?: boolean) => void
  updateConfig: (config: Partial<TabConfig>) => void
  setWorkspaceItem: (item: WorkspaceItem) => void
  /** Cover the held item with the chat. */
  hideItem: () => void
  /** Let go of the held item, the chat shows. */
  dropItem: () => void
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

/** Id of the tab the tab picker previews, null while it previews none. */
export const TabPeekContext = createContext<string | null>(null)

/** Whether the tab picker previews the enclosing tab, never the frontmost one. */
export function useIsPeeked(): boolean {
  const { tab } = useTab()
  return useContext(TabPeekContext) === tab.id
}
