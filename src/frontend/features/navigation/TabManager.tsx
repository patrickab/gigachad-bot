"use client"

import { createContext, useCallback, useContext, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react"
import { Plus, X } from "lucide-react"
import { cn } from "@/lib/utils"
import { DEFAULT_TAB_CONFIG, type TabConfig } from "@/contexts/SettingsContext"
import { TabContext, type Tab, type TabActions, type TabContextValue } from "@/contexts/TabContext"

interface TabManagerProps {
  /** Rendered once per tab inside that tab's `TabContext`. */
  children: React.ReactNode
  onTabsChange?: (tabs: Tab[]) => void
  defaultConfig?: TabConfig
  ref?: React.Ref<TabManagerHandle>
}

export interface TabManagerHandle {
  /** Swap one tab for `tab` in place and activate it; the others stay open. */
  replaceTab: (tabId: string, tab: Tab) => void
  switchToTab: (tabId: string) => void
  getTabs: () => Tab[]
  getActiveTabId: () => string
  updateTabHistoryFile: (tabId: string, historyFile: string | null) => void
}

/** Tab state and actions shared with `SidebarTabs`, which renders inside each tab's own sidebar. */
interface TabStripValue {
  tabs: Tab[]
  activeTabId: string
  editingTabId: string | null
  /** null hides the label so the command bar input can take its slot */
  labelOf: (tab: Tab) => string | null
  isLoading: (tab: Tab) => boolean
  select: (id: string) => void
  close: (id: string) => void
  add: () => void
  startRename: (id: string) => void
  commitRename: (id: string, value: string) => void
  cancelRename: () => void
}
const TabStripContext = createContext<TabStripValue | null>(null)

// Monotonic id source, module-level so ids stay unique across TabManager remounts
let tabIdCounter = 0

export function nextTab(config?: Partial<TabConfig>): Tab {
  return { id: `tab-${++tabIdCounter}`, name: null, chatId: crypto.randomUUID(), historyFile: null, title: null, config: { ...DEFAULT_TAB_CONFIG, ...config }, appMode: "chat" }
}

const INITIAL_TAB: Tab = nextTab()

/** Provides one tab's state and actions; memoized so the context only changes with the tab or its active flag. */
function TabScope({ tab, isActive, updateTab, updateModeLabel, children }: {
  tab: Tab
  isActive: boolean
  updateTab: (id: string, patch: (tab: Tab) => Partial<Tab>) => void
  updateModeLabel: (id: string, label: string, loading?: boolean) => void
  children: React.ReactNode
}) {
  // One stable action set per tab. Consumers list these as effect dependencies, so a
  // fresh closure per render re-fires them all, and two mode-label producers in one
  // tab (the chat label plus an open document's filename) then overwrite each other forever.
  // TabScope is keyed by tab id, so the actions are built once for the tab's lifetime.
  const [actions] = useState<TabActions>(() => {
    const id = tab.id
    return {
      setModeLabel: (label, loading) => updateModeLabel(id, label, loading),
      updateConfig: (config) => updateTab(id, (t) => ({ config: { ...t.config, ...config } })),
      setAppMode: (appMode) => updateTab(id, () => ({ appMode })),
      setHistoryFile: (historyFile) => updateTab(id, () => ({ historyFile })),
      setTitle: (title) => updateTab(id, () => ({ title })),
    }
  })
  const value = useMemo<TabContextValue>(() => ({ tab, isActive, ...actions }), [tab, isActive, actions])
  return <TabContext.Provider value={value}>{children}</TabContext.Provider>
}

function Spinner() {
  return <span className="h-2.5 w-2.5 shrink-0 animate-spin rounded-full border border-ink-faint border-t-ink-muted" />
}

interface SidebarTabsProps {
  ownerTabId: string
  /** Lifts the owner row above the command bar's click-away overlay */
  commandOpen: boolean
  onLabelSlot: (element: HTMLElement | null) => void
  onMenuSlot: (element: HTMLElement | null) => void
}

/** "Tabs" section for the sidebar bottom: one quiet row per tab, the active one softly boxed.
 *
 * - Every tab mounts its own sidebar, so `ownerTabId` marks the row of the tab rendering this list
 * - The owner row hosts the command bar: its label slot takes the input, the menu slot opens below in flow
 */
export function SidebarTabs(props: SidebarTabsProps) {
  const strip = useContext(TabStripContext)
  if (!strip) return null

  return (
    <div className="space-y-0.5">
      <div className="group flex items-center justify-between px-2 pb-0.5">
        <span className="text-[11px] font-medium text-ink-faint">Tabs</span>
        <button
          aria-label="New tab"
          onClick={strip.add}
          className="rounded p-0.5 text-ink-faint opacity-0 transition-opacity hover:text-ink group-hover:opacity-100"
        >
          <Plus className="h-3 w-3" />
        </button>
      </div>
      {strip.tabs.map((tab) => <TabRow key={tab.id} tab={tab} strip={strip} {...props} />)}
    </div>
  )
}

function TabRow({ tab, strip, ownerTabId, commandOpen, onLabelSlot, onMenuSlot }: SidebarTabsProps & {
  tab: Tab
  strip: TabStripValue
}) {
  const isOwner = tab.id === ownerTabId
  const isActive = tab.id === strip.activeTabId
  const label = strip.labelOf(tab)

  return (
    <div className={cn(isOwner && commandOpen && "relative z-[80]")}>
      <div
        className={cn(
          "group flex h-6 cursor-pointer select-none items-center gap-1.5 rounded-md px-2 text-[11px] transition-colors",
          isActive
            ? "bg-surface-elevated/70 text-ink-muted"
            : "text-ink-subtle hover:bg-surface-elevated/40 hover:text-ink-muted",
        )}
        onClick={() => strip.select(tab.id)}
        onContextMenu={(e) => { e.preventDefault(); strip.close(tab.id) }}
        onDoubleClick={() => strip.startRename(tab.id)}
      >
        <div ref={isOwner ? onLabelSlot : undefined} className="flex min-w-0 flex-1 items-center">
          {label === null ? null : tab.id === strip.editingTabId ? (
            <input
              autoFocus
              defaultValue={label}
              className="min-w-0 flex-1 bg-transparent text-xs text-ink outline-none"
              onFocus={(e) => e.target.select()}
              onBlur={(e) => strip.commitRename(tab.id, e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") strip.commitRename(tab.id, e.currentTarget.value)
                if (e.key === "Escape") strip.cancelRename()
              }}
              onClick={(e) => e.stopPropagation()}
            />
          ) : (
            <span className="truncate">{label}</span>
          )}
        </div>
        {strip.isLoading(tab) && <Spinner />}
        {strip.tabs.length > 1 && (
          <button
            aria-label={`Close ${label ?? "tab"}`}
            className={cn(
              "shrink-0 rounded p-0.5 text-ink-faint transition-opacity hover:text-ink",
              isActive ? "opacity-100" : "opacity-0 group-hover:opacity-100",
            )}
            onClick={(e) => { e.stopPropagation(); strip.close(tab.id) }}
          >
            <X className="h-3 w-3" />
          </button>
        )}
      </div>
      {isOwner && <div ref={onMenuSlot} />}
    </div>
  )
}

export function TabManager({ children, onTabsChange, defaultConfig = DEFAULT_TAB_CONFIG, ref }: TabManagerProps) {
  const [tabs, setTabs] = useState<Tab[]>([INITIAL_TAB])
  const [activeTab, setActiveTab] = useState(INITIAL_TAB.id)
  const [editingTab, setEditingTab] = useState<string | null>(null)
  const [modeLabels, setModeLabels] = useState<Record<string, string>>({})
  const [modeLoading, setModeLoading] = useState<Record<string, boolean>>({})
  const onTabsChangeRef = useRef(onTabsChange)
  useEffect(() => { onTabsChangeRef.current = onTabsChange }, [onTabsChange])

  const updateTab = useCallback((id: string, patch: (tab: Tab) => Partial<Tab>) => {
    setTabs((prev) => prev.map((t) => (t.id === id ? { ...t, ...patch(t) } : t)))
  }, [])

  useImperativeHandle(ref, () => ({
    replaceTab: (tabId: string, tab: Tab) => {
      setTabs((prev) => prev.map((t) => t.id === tabId ? tab : t))
      setActiveTab(tab.id)
    },
    switchToTab: (tabId: string) => setActiveTab(tabId),
    getTabs: () => tabs,
    getActiveTabId: () => activeTab,
    updateTabHistoryFile: (tabId: string, historyFile: string | null) => updateTab(tabId, () => ({ historyFile })),
  }), [tabs, activeTab, updateTab])

  useEffect(() => {
    onTabsChangeRef.current?.(tabs)
  }, [tabs])

  const addTab = useCallback(() => {
    const tab = nextTab(defaultConfig)
    setTabs((prev) => [...prev, tab])
    setActiveTab(tab.id)
  }, [defaultConfig])

  const closeTab = useCallback((id: string) => {
    setTabs((prev) => {
      if (prev.length <= 1) return prev
      const next = prev.filter((t) => t.id !== id)
      if (id === activeTab) {
        const idx = prev.findIndex((t) => t.id === id)
        const targetIdx = Math.max(0, idx - 1)
        setActiveTab(next[targetIdx]?.id ?? "")
      }
      return next
    })
  }, [activeTab])

  const closeActiveTab = useCallback(() => {
    closeTab(activeTab)
  }, [activeTab, closeTab])

  const switchToNextTab = useCallback(() => {
    setActiveTab((current) => {
      const idx = tabs.findIndex((t) => t.id === current)
      const next = (idx + 1) % tabs.length
      return tabs[next].id
    })
  }, [tabs])

  const switchToPrevTab = useCallback(() => {
    setActiveTab((current) => {
      const idx = tabs.findIndex((t) => t.id === current)
      const prev = (idx - 1 + tabs.length) % tabs.length
      return tabs[prev].id
    })
  }, [tabs])

  useEffect(() => {
    function handler(e: KeyboardEvent) {
      if (e.altKey && e.key === "t") {
        e.preventDefault()
        addTab()
      }
      if (e.altKey && e.key === "w") {
        e.preventDefault()
        closeActiveTab()
      }
      if (e.key === "Tab" && !e.altKey && !e.ctrlKey && !e.metaKey) {
        const target = e.target as HTMLElement
        const isEditable =
          target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.isContentEditable
        if (!isEditable) {
          e.preventDefault()
          if (e.shiftKey) {
            switchToPrevTab()
          } else {
            switchToNextTab()
          }
        }
      }
    }
    document.addEventListener("keydown", handler)
    return () => document.removeEventListener("keydown", handler)
  }, [addTab, closeActiveTab, switchToNextTab, switchToPrevTab])

  const commitRename = useCallback((id: string, value: string) => {
    setEditingTab(null)
    updateTab(id, () => ({ name: value.trim() || null }))
  }, [updateTab])

  const updateModeLabel = useCallback((id: string, label: string, loading = false) => {
    setModeLabels((prev) => (prev[id] === label ? prev : { ...prev, [id]: label }))
    setModeLoading((prev) => (prev[id] === loading ? prev : { ...prev, [id]: loading }))
  }, [])

  const strip = useMemo<TabStripValue>(() => ({
    tabs,
    activeTabId: activeTab,
    editingTabId: editingTab,
    labelOf: (tab) => {
      const label = modeLabels[tab.id]
      // "\0" is a sentinel meaning the tab renders no label at all
      if (label === "\0") return null
      return tab.name ?? (label || "Chat")
    },
    isLoading: (tab) => tab.name == null && (modeLoading[tab.id] ?? false),
    select: setActiveTab,
    close: closeTab,
    add: addTab,
    startRename: setEditingTab,
    commitRename,
    cancelRename: () => setEditingTab(null),
  }), [tabs, activeTab, editingTab, modeLabels, modeLoading, closeTab, addTab, commitRename])

  return (
    <TabStripContext.Provider value={strip}>
      <div className="relative h-screen overflow-hidden">
        {tabs.map((tab) => (
          <div
            key={tab.id}
            className="h-full"
            style={{ display: tab.id === activeTab ? undefined : "none" }}
          >
            <TabScope tab={tab} isActive={tab.id === activeTab} updateTab={updateTab} updateModeLabel={updateModeLabel}>
              {children}
            </TabScope>
          </div>
        ))}
      </div>
    </TabStripContext.Provider>
  )
}

