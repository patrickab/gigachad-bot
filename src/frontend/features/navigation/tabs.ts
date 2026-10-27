import { createContext } from "react"
import { DEFAULT_TAB_CONFIG, type TabConfig } from "@/contexts/SettingsContext"
import type { Tab, WorkspaceItem } from "@/contexts/TabContext"
import { baseName } from "@/lib/paths"

/** The display name project `slug` gives its file `path`, if any. */
type ShownName = (slug: string | null, path: string) => string | undefined

/** The name a canvas or document item shows: its project's display name, else its file name. */
function itemName(item: WorkspaceItem, shownName: ShownName): string {
  return shownName(item.projectSlug, item.path) ?? baseName(item.path)
}

/** A fresh chat tab with `config` over the default tab config. */
export function nextTab(config?: Partial<TabConfig>): Tab {
  return { id: `tab-${crypto.randomUUID()}`, name: null, chatId: crypto.randomUUID(), historyFile: null, title: null, config: { ...DEFAULT_TAB_CONFIG, ...config }, appMode: "chat", projectSlug: null, workspaceItem: null }
}

/** `tab` with its project references moved from `oldSlug` to `newSlug`, or `tab` itself when it has none. */
export function relocatedTab(tab: Tab, oldSlug: string, newSlug: string): Tab {
  const relocate = (path: string, prefix: string) => path.startsWith(`${prefix}${oldSlug}/`) ? `${prefix}${newSlug}${path.slice(prefix.length + oldSlug.length)}` : path
  const historyFile = tab.historyFile && relocate(tab.historyFile, "")
  const item = tab.workspaceItem
  const isInProject = tab.projectSlug === oldSlug || historyFile !== tab.historyFile || item?.projectSlug === oldSlug
  if (!isInProject) return tab
  return {
    ...tab,
    historyFile,
    projectSlug: tab.projectSlug === oldSlug ? newSlug : tab.projectSlug,
    workspaceItem: item?.projectSlug === oldSlug
      ? { ...item, projectSlug: newSlug, path: relocate(relocate(item.path, "project/"), "graph/") }
      : item,
  }
}

/** Id of the tab that takes over when `id` closes: the one before it, else the new first tab. */
export function tabAfterClosing(tabs: Tab[], id: string): string {
  const remaining = tabs.filter((tab) => tab.id !== id)
  return remaining[Math.max(0, tabs.findIndex((tab) => tab.id === id) - 1)]?.id ?? ""
}

/** Describe the visible surface, not a background item retained by the tab; `shownName` resolves project display names. */
export function tabLocation(tab: Tab, shownName: ShownName) {
  if (tab.appMode === "omp") {
    return { category: tab.projectSlug ? "code" : "omp", file: tab.name ?? tab.title ?? "omp" }
  }
  if (tab.appMode === "factory") {
    return { category: "factory", file: tab.name ?? (tab.workspaceItem?.path.includes("/") ? "task" : tab.workspaceItem?.path ?? "board") }
  }
  if (tab.appMode === "canvas") {
    return tab.workspaceItem?.kind === "architecture"
      ? { category: "architecture", file: tab.name ?? "System overview" }
      : { category: "canvases", file: tab.name ?? (tab.workspaceItem?.kind === "canvas" ? itemName(tab.workspaceItem, shownName) : "new.canvas") }
  }
  if (tab.appMode === "document" && tab.workspaceItem?.kind === "diff") {
    return { category: "diff", file: tab.name ?? baseName(tab.workspaceItem.path) }
  }
  if (tab.appMode === "document") {
    return { category: "files", file: tab.name ?? (tab.workspaceItem ? itemName(tab.workspaceItem, shownName) : "New file") }
  }
  return { category: "chats", file: tab.name ?? tab.title ?? (tab.historyFile ? baseName(tab.historyFile) : "New chat") }
}

/** Activity description of `tab` for screen readers; a chat shows its latest mode label. */
export function tabLabel(tab: Tab, modeLabel: string | undefined, shownName: ShownName): string {
  if (tab.appMode !== "chat" && tab.workspaceItem) {
    if (tab.appMode === "omp") return tab.name ?? tab.title ?? "omp"
    if (tab.appMode === "factory") return tab.name ?? "Factory"
    return tab.name ?? (tab.workspaceItem.kind === "architecture" ? "Architecture" : itemName(tab.workspaceItem, shownName))
  }
  return tab.name ?? (modeLabel || "Chat")
}

/** Tab state and actions shared by toolbar workspace switchers. */
export interface TabStripValue {
  tabs: Tab[]
  activeTabId: string
  /** Activity description exposed alongside the workspace breadcrumb. */
  labelOf: (tab: Tab) => string
  isLoading: (tab: Tab) => boolean
  select: (id: string) => void
  close: (id: string) => void
  add: () => void
}
export const TabStripContext = createContext<TabStripValue | null>(null)
