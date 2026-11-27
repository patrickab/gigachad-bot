import { createContext } from "react"
import { DEFAULT_TAB_CONFIG, type TabConfig } from "@/contexts/SettingsContext"
import { itemOf, scratchItem, showingItem, shownItem, type Tab, type WorkspaceItem } from "@/contexts/TabContext"
import { parseFactoryPath } from "@/features/factory/items"
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
  return { id: `tab-${crypto.randomUUID()}`, chatId: crypto.randomUUID(), historyFile: null, title: null, config: { ...DEFAULT_TAB_CONFIG, ...config }, projectSlug: null, workspaceItem: null, itemHidden: false }
}

/** `tab` with its project references moved from `oldSlug` to `newSlug`, or `tab` itself when it has none. A file in the project's workbench folder moves with it. Reads the held item, shown or not. */
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

/** `tab` with its file item moved from `oldPath` to `newPath` after a rename, or `tab` itself. A workbench item keeps its editor session. Reads the held item, shown or not. */
export function renamedTab(tab: Tab, oldPath: string, newPath: string): Tab {
  const item = tab.workspaceItem
  if (!item || item.path !== oldPath || !FILE_ITEM_KINDS.includes(item.kind)) return tab
  const sessionKey = item.kind === "workbench" ? { sessionKey: item.sessionKey ?? oldPath } : {}
  return { ...tab, workspaceItem: { ...item, path: newPath, ...sessionKey } }
}

/** The tab to show LaTeX file `item` in as another buffer: a workbench tab whose project folder holds it, the active one first. Reads the held item, shown or not. */
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
  const shown = active && shownItem(active)
  return item.kind === "document" && kindOf(item.path) === "image" && shown?.kind === "workbench" && workbenchKindOf(shown.path) === "latex"
}

/** The tab already showing file `item` or Pi session `item`, the active one first: opening it again switches there instead of duplicating it. Reads the held item, shown or not. */
export function tabShowingItem(tabs: Tab[], item: WorkspaceItem, activeId: string): Tab | undefined {
  const sessionId = item.kind === "pi" ? item.pi?.sessionId : null
  if (!sessionId && !FILE_ITEM_KINDS.includes(item.kind)) return undefined
  const shows = ({ workspaceItem }: Tab) => workspaceItem?.kind === item.kind
    && (sessionId ? workspaceItem.pi?.sessionId === sessionId : workspaceItem.path === item.path)
  return tabs.find((tab) => tab.id === activeId && shows(tab)) ?? tabs.find(shows)
}

/** Open tabs saved across a reload of this browser tab. */
export interface SavedTabs { tabs: Tab[]; activeId: string }

/** The tabs `raw` (a `JSON.stringify(SavedTabs)`) holds with current config defaults, or null when it holds none or is malformed. */
export function parseSavedTabs(raw: string | null): SavedTabs | null {
  if (!raw) return null
  try {
    const saved = JSON.parse(raw) as Partial<SavedTabs>
    if (!Array.isArray(saved.tabs)) return null
    // An older build's omp tab: omp-web is gone, nothing shows it any more
    const kept = (saved.tabs as (Tab & { appMode?: string })[]).filter((tab) => tab.appMode !== "omp" && (tab.workspaceItem?.kind as string | undefined) !== "omp")
    const tabs = kept.map(({ appMode, ...tab }): Tab => {
      // An older build's tab: `appMode` named its surface, "chat" over an item hid it, "canvas" without one was the scratch canvas
      const workspaceItem = appMode === "canvas" && !tab.workspaceItem ? scratchItem(tab.projectSlug) : tab.workspaceItem
      const itemHidden = tab.itemHidden ?? (appMode === "chat" && workspaceItem !== null)
      return { ...tab, config: { ...DEFAULT_TAB_CONFIG, ...tab.config }, workspaceItem, itemHidden }
    })
    if (tabs.length === 0) return null
    return { tabs, activeId: tabs.some((tab) => tab.id === saved.activeId) ? saved.activeId! : tabs[0].id }
  } catch {
    return null
  }
}

/** The paths of the files `tabs` hold: a tab holds its file item's file whether or not its editor is open, until the tab closes or takes another item. Reads the held item, shown or not. */
export function filesOfTabs(tabs: readonly Tab[]): Set<string> {
  const files = new Set<string>()
  for (const { workspaceItem } of tabs) {
    if (workspaceItem && FILE_ITEM_KINDS.includes(workspaceItem.kind)) files.add(workspaceItem.path)
  }
  return files
}

/** Whether `tab` shows the deleted `path` (a file or a folder): its file item lies inside it, or it is a workbench of a LaTeX project folder inside it. */
export function showsRemoved(tab: Tab, path: string): boolean {
  const item = tab.workspaceItem
  if (!item || !FILE_ITEM_KINDS.includes(item.kind)) return false
  return isInside(item.path, path) || (item.kind === "workbench" && !!item.texFolder && isInside(item.texFolder, path))
}

/**
 * `tabs` without the ones showing the deleted `path`, and the tab to show: `activeId` while it stays, else the nearest remaining tab before it, else the first.
 * A tab holding it hidden behind its chat stays and lets go of it. Every tab gone leaves `fresh` alone.
 */
export function tabsAfterRemoval(tabs: Tab[], path: string, activeId: string, fresh: () => Tab): { tabs: Tab[]; activeId: string } {
  // An empty or root path names no file, every tab would close
  if (path.replace(/\/+$/, "") === "") return { tabs, activeId }
  if (!tabs.some((tab) => showsRemoved(tab, path))) return { tabs, activeId }
  const kept = tabs.map((tab) => tab.itemHidden && showsRemoved(tab, path) ? { ...tab, workspaceItem: null, itemHidden: false } : tab)
  const remaining = kept.filter((tab) => !showsRemoved(tab, path))
  if (remaining.length === 0) {
    const replacement = fresh()
    return { tabs: [replacement], activeId: replacement.id }
  }
  if (remaining.some((tab) => tab.id === activeId)) return { tabs: remaining, activeId }
  const before = kept.slice(0, kept.findIndex((tab) => tab.id === activeId)).findLast((tab) => remaining.includes(tab))
  return { tabs: remaining, activeId: (before ?? remaining[0]).id }
}

/** Id of the tab that takes over when `id` closes: the one before it, else the new first tab. */
export function tabAfterClosing(tabs: Tab[], id: string): string {
  const remaining = tabs.filter((tab) => tab.id !== id)
  return remaining[Math.max(0, tabs.findIndex((tab) => tab.id === id) - 1)]?.id ?? ""
}

/** The Pi session `tab` shows, null for other tabs and a Pi tab before its first message. */
export function piSessionOf(tab: Tab): string | null {
  return itemOf(tab, "pi")?.pi?.sessionId ?? null
}

/** Describe the visible surface, not a background item retained by the tab; `shownName` resolves project display names. */
export function tabLocation(tab: Tab, shownName: ShownName) {
  const item = shownItem(tab)
  switch (item?.kind) {
    case "pi": return { category: "code", file: tab.title ?? (item.pi ? baseName(item.pi.repo) : "Pi") }
    case "workbench": return { category: workbenchKindOf(item.path) ?? "files", file: baseName(item.path) }
    case "factory": {
      const { repo, taskId } = parseFactoryPath(item.path)
      return { category: "factory", file: taskId ? "task" : baseName(repo) }
    }
    case "learn": return { category: "learn", file: tab.title ?? "Learn" }
    case "architecture": return { category: "architecture", file: "System overview" }
    case "canvas": return { category: "canvases", file: itemName(item, shownName) }
    case "scratch": return { category: "canvases", file: "new.canvas" }
    case "diff": return { category: "diff", file: baseName(item.path) }
    case "document": return { category: "files", file: itemName(item, shownName) }
    default: return { category: "chats", file: tab.title ?? (tab.historyFile ? baseName(tab.historyFile) : "New chat") }
  }
}

/** Activity description of `tab` for screen readers; a chat and the scratch canvas show their latest mode label. */
export function tabLabel(tab: Tab, modeLabel: string | undefined, shownName: ShownName): string {
  const item = shownItem(tab)
  switch (item?.kind) {
    case "pi": return tab.title ?? (item.pi ? baseName(item.pi.repo) : "Pi")
    case "learn": return tab.title ?? "Learn"
    case "workbench": return baseName(item.path)
    case "factory": return "Factory"
    case "architecture": return "Architecture"
    case "canvas": case "document": case "diff": return itemName(item, shownName)
    default: return modeLabel || "Chat"
  }
}

/** The tab the tab picker previews and the box in its card the tab is scaled into. */
export interface TabPeek {
  id: string
  slot: HTMLElement
}

/** Tab state and actions shared by the top bar's tab picker. */
export interface TabStripValue {
  tabs: Tab[]
  activeTabId: string
  /** Activity description shown in each tab's preview. */
  labelOf: (tab: Tab) => string
  isLoading: (tab: Tab) => boolean
  select: (id: string) => void
  close: (id: string) => void
  /** Open a fresh chat tab in the current one's project, frontmost unless `activate` is false. Returns its id. */
  add: (options?: { activate?: boolean }) => string
  /** Preview a background tab scaled into `slot`, null ends the preview. */
  peek: (peek: TabPeek | null) => void
}
export const TabStripContext = createContext<TabStripValue | null>(null)
