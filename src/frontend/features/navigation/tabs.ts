import { createContext } from "react"
import { DEFAULT_TAB_CONFIG, type TabConfig } from "@/contexts/SettingsContext"
import { showingItem, type Tab, type WorkspaceItem } from "@/contexts/TabContext"
import { relocatedWorkbenchPath, workbenchKindOf } from "@/lib/domain/workbench"
import { baseName, isInside, kindOf } from "@/lib/paths"

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

/** `tab` with its project references moved from `oldSlug` to `newSlug`, or `tab` itself when it has none. A file in the project's workbench folder moves with it. */
export function relocatedTab(tab: Tab, oldSlug: string, newSlug: string): Tab {
  const relocate = (path: string, prefix: string) => path.startsWith(`${prefix}${oldSlug}/`) ? `${prefix}${newSlug}${path.slice(prefix.length + oldSlug.length)}` : path
  const historyFile = tab.historyFile && relocate(tab.historyFile, "")
  const item = tab.workspaceItem
  const itemInProject = item?.projectSlug === oldSlug
  // A moved project folder remounts its workbench file, so the old editor session key no longer applies.
  const movedItem = item && {
    ...item,
    sessionKey: undefined,
    projectSlug: itemInProject ? newSlug : item.projectSlug,
    path: relocatedWorkbenchPath(itemInProject ? relocate(relocate(item.path, "project/"), "graph/") : item.path, oldSlug, newSlug),
  }
  const itemMoved = itemInProject || (!!item && movedItem?.path !== item.path)
  if (tab.projectSlug !== oldSlug && historyFile === tab.historyFile && !itemMoved) return tab
  return {
    ...tab,
    historyFile,
    projectSlug: tab.projectSlug === oldSlug ? newSlug : tab.projectSlug,
    workspaceItem: itemMoved ? movedItem : item,
  }
}

const FILE_ITEM_KINDS: readonly WorkspaceItem["kind"][] = ["canvas", "document", "workbench"]

/** `tab` with its file item moved from `oldPath` to `newPath` after a rename, or `tab` itself. A workbench item keeps its editor session. */
export function renamedTab(tab: Tab, oldPath: string, newPath: string): Tab {
  const item = tab.workspaceItem
  if (!item || item.path !== oldPath || !FILE_ITEM_KINDS.includes(item.kind)) return tab
  const sessionKey = item.kind === "workbench" ? { sessionKey: item.sessionKey ?? oldPath } : {}
  return { ...tab, workspaceItem: { ...item, path: newPath, ...sessionKey } }
}

/** The tab to show LaTeX file `item` in as another buffer: a workbench tab whose project folder holds it, the active one first. */
export function texProjectTab(tabs: Tab[], item: WorkspaceItem, activeId: string): Tab | undefined {
  if (item.kind !== "workbench" || workbenchKindOf(item.path) !== "latex") return undefined
  const holdsItem = ({ workspaceItem }: Tab) =>
    workspaceItem?.kind === "workbench" && !!workspaceItem.texFolder && isInside(item.path, workspaceItem.texFolder)
  return tabs.find((tab) => tab.id === activeId && holdsItem(tab)) ?? tabs.find(holdsItem)
}

/** `tab` showing the file at `path` as another buffer of its workbench item: the editor session, chat and project folder carry on. */
export function texBufferShown(tab: Tab, path: string): Tab {
  const item = tab.workspaceItem
  if (!item) return tab
  return { ...tab, ...showingItem({ ...item, path, sessionKey: item.sessionKey ?? item.path }) }
}

/** Whether `item` opens in a new tab, not in `active`: an image from a LaTeX workbench tab, so its Neovim session and unsaved buffers survive. */
export function opensBesideWorkbench(active: Tab | undefined, item: WorkspaceItem): boolean {
  const shown = active?.appMode === "workbench" ? active.workspaceItem : null
  return item.kind === "document" && kindOf(item.path) === "image" && shown?.kind === "workbench" && workbenchKindOf(shown.path) === "latex"
}

/** Whether `tab` shows the deleted `path` (a file or a folder): its file item lies inside it, or it is a workbench of a LaTeX project folder inside it. */
export function showsRemoved(tab: Tab, path: string): boolean {
  const item = tab.workspaceItem
  if (!item || !FILE_ITEM_KINDS.includes(item.kind)) return false
  return isInside(item.path, path) || (item.kind === "workbench" && !!item.texFolder && isInside(item.texFolder, path))
}

/**
 * `tabs` without the ones showing the deleted `path`, and the tab to show: `activeId` while it stays, else the nearest remaining tab before it, else the first.
 * Every tab gone leaves `fresh` alone.
 */
export function tabsAfterRemoval(tabs: Tab[], path: string, activeId: string, fresh: () => Tab): { tabs: Tab[]; activeId: string } {
  const remaining = tabs.filter((tab) => !showsRemoved(tab, path))
  if (remaining.length === tabs.length) return { tabs, activeId }
  if (remaining.length === 0) {
    const replacement = fresh()
    return { tabs: [replacement], activeId: replacement.id }
  }
  if (remaining.some((tab) => tab.id === activeId)) return { tabs: remaining, activeId }
  const before = tabs.slice(0, tabs.findIndex((tab) => tab.id === activeId)).findLast((tab) => remaining.includes(tab))
  return { tabs: remaining, activeId: (before ?? remaining[0]).id }
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
  if (tab.appMode === "workbench") {
    const kind = tab.workspaceItem ? workbenchKindOf(tab.workspaceItem.path) : null
    return { category: kind ?? "files", file: tab.name ?? (tab.workspaceItem ? baseName(tab.workspaceItem.path) : "File") }
  }
  if (tab.appMode === "factory") {
    return { category: "factory", file: tab.name ?? (tab.workspaceItem?.path.includes("/") ? "task" : tab.workspaceItem?.path ?? "board") }
  }
  if (tab.appMode === "learn") {
    return { category: "learn", file: tab.name ?? tab.title ?? "Learn" }
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
    if (tab.appMode === "learn") return tab.name ?? tab.title ?? "Learn"
    if (tab.appMode === "workbench") return tab.name ?? baseName(tab.workspaceItem.path)
    if (tab.appMode === "factory") return tab.name ?? "Factory"
    return tab.name ?? (tab.workspaceItem.kind === "architecture" ? "Architecture" : itemName(tab.workspaceItem, shownName))
  }
  return tab.name ?? (modeLabel || "Chat")
}

/** Tab state and actions shared by the sidebar's tab list. */
export interface TabStripValue {
  tabs: Tab[]
  activeTabId: string
  /** Activity description exposed on each tab's sidebar row. */
  labelOf: (tab: Tab) => string
  isLoading: (tab: Tab) => boolean
  select: (id: string) => void
  close: (id: string) => void
  add: () => void
}
export const TabStripContext = createContext<TabStripValue | null>(null)
