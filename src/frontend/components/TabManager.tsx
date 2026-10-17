"use client"

import { createContext, useCallback, useContext, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react"
import { Plus, X } from "lucide-react"
import { cn } from "@/lib/utils"
import type { SettingsState } from "@/contexts/SettingsContext"
import type { AppSurface } from "@/contexts/SidebarContext"

// Tabs stay mounted in the background (CSS-hidden below, not unmounted), so key
// listeners deep in a tab's content must check this instead of firing for whichever
// tab happens to be frontmost. Provided per-tab in the render loop below.
const TabActiveContext = createContext(true)
export const TabActiveProvider = TabActiveContext.Provider
export function useTabActive(): boolean {
  return useContext(TabActiveContext)
}

export interface TabConfig {
  selectedModel: string
  selectedPrompt: string | null
  reasoningEffort: string
  downscaleImages: boolean
  researchFastModel: string
  researchSmartModel: string
  researchStrategicModel: string
  researchDepth: number
  researchBreadth: number
  researchReasoning: string
  researchReportType: string
  searchSystemInstructions: string
  searchDomain: string
}

// Settings copied into a new tab (search* fields are intentionally not per-tab)
const TAB_CONFIG_KEYS: (keyof TabConfig)[] = [
  "selectedModel", "selectedPrompt", "reasoningEffort",
  "downscaleImages", "researchFastModel", "researchSmartModel", "researchStrategicModel",
  "researchDepth", "researchBreadth", "researchReasoning", "researchReportType",
]

// Selected model starts blank and is seeded from the backend's default chat model
// by the settings-sync effect in app/page.tsx. Never hardcode a model name here.
const DEFAULT_CONFIG: TabConfig = {
  selectedModel: "",
  selectedPrompt: null,
  reasoningEffort: "medium",
  downscaleImages: true,
  researchFastModel: "",
  researchSmartModel: "",
  researchStrategicModel: "",
  researchDepth: 2,
  researchBreadth: 4,
  researchReasoning: "medium",
  researchReportType: "deep",
  searchSystemInstructions: "",
  searchDomain: "",
}

export function settingsToTabConfig(settings: SettingsState): TabConfig {
  const config: Partial<TabConfig> = {}
  for (const key of TAB_CONFIG_KEYS) {
    if (key in settings) {
      (config as Record<string, unknown>)[key] = settings[key as keyof SettingsState]
    }
  }
  return { ...DEFAULT_CONFIG, ...config }
}

export interface Tab {
  id: string
  name: string | null
  chatId: string
  historyFile: string | null
  title: string | null
  config: TabConfig
  appMode: AppSurface
}

interface TabManagerProps {
  renderContent: (tab: Tab, onModeLabel: (label: string, loading?: boolean) => void, isActive: boolean, onConfigChange: (config: Partial<TabConfig>) => void, onAppModeChange: (mode: AppSurface) => void) => React.ReactNode
  onCloseTab?: (tab: Tab) => void
  onTabsChange?: (tabs: Tab[]) => void
  defaultConfig?: TabConfig
  ref?: React.Ref<TabManagerHandle>
}

export interface TabManagerHandle {
  /** Swap one tab for `tab` in place and activate it; the others stay open. */
  replaceTab: (tabId: string, tab: Tab) => void
  switchToTab: (tabId: string) => void
  addTabWithName: (name: string | null, historyFile: string | null) => string
  getTabs: () => Tab[]
  getActiveTabId: () => string
  updateTabHistoryFile: (tabId: string, historyFile: string | null) => void
  updateTabTitle: (tabId: string, title: string | null) => void
  updateTabConfig: (tabId: string, config: Partial<TabConfig>) => void
  updateTabAppMode: (tabId: string, mode: AppSurface) => void
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

export function nextTab(name: string | null = null, historyFile: string | null = null, title: string | null = null, config?: Partial<TabConfig>, appMode: AppSurface = "chat"): Tab {
  return { id: `tab-${++tabIdCounter}`, name, chatId: crypto.randomUUID(), historyFile, title, config: { ...DEFAULT_CONFIG, ...config }, appMode }
}

const INITIAL_TAB: Tab = nextTab()
const INITIAL_CONFIG: TabConfig = { ...DEFAULT_CONFIG }

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

export function TabManager({ renderContent, onCloseTab, onTabsChange, defaultConfig = INITIAL_CONFIG, ref }: TabManagerProps) {
  const [tabs, setTabs] = useState<Tab[]>([INITIAL_TAB])
  const [activeTab, setActiveTab] = useState(INITIAL_TAB.id)
  const [editingTab, setEditingTab] = useState<string | null>(null)
  const [modeLabels, setModeLabels] = useState<Record<string, string>>({})
  const [modeLoading, setModeLoading] = useState<Record<string, boolean>>({})
  const [customNames, setCustomNames] = useState<Record<string, string | null>>({})
  const onTabsChangeRef = useRef(onTabsChange)
  useEffect(() => { onTabsChangeRef.current = onTabsChange }, [onTabsChange])

  useImperativeHandle(ref, () => ({
    replaceTab: (tabId: string, tab: Tab) => {
      setTabs((prev) => prev.map((t) => t.id === tabId ? tab : t))
      setActiveTab(tab.id)
    },
    switchToTab: (tabId: string) => setActiveTab(tabId),
    addTabWithName: (name: string | null, historyFile: string | null) => {
      const tab = nextTab(name, historyFile)
      setTabs((prev) => [...prev, tab])
      setActiveTab(tab.id)
      return tab.id
    },
    getTabs: () => tabs,
    getActiveTabId: () => activeTab,
    updateTabHistoryFile: (tabId: string, historyFile: string | null) => {
      setTabs((prev) => prev.map((t) => t.id === tabId ? { ...t, historyFile } : t))
    },
    updateTabTitle: (tabId: string, title: string | null) => {
      setTabs((prev) => prev.map((t) => t.id === tabId ? { ...t, title } : t))
    },
    updateTabConfig: (tabId: string, config: Partial<TabConfig>) => {
      setTabs((prev) => prev.map((t) => t.id === tabId ? { ...t, config: { ...t.config, ...config } } : t))
    },
    updateTabAppMode: (tabId: string, mode: AppSurface) => {
      setTabs((prev) => prev.map((t) => t.id === tabId ? { ...t, appMode: mode } : t))
    },
  }), [tabs, activeTab])

  useEffect(() => {
    onTabsChangeRef.current?.(tabs)
  }, [tabs])

  const addTab = useCallback(() => {
    const tab = nextTab(null, null, null, defaultConfig)
    setTabs((prev) => [...prev, tab])
    setActiveTab(tab.id)
  }, [defaultConfig])

  const closeTab = useCallback((id: string) => {
    setTabs((prev) => {
      if (prev.length <= 1) return prev
      const tab = prev.find((t) => t.id === id)
      if (tab) onCloseTab?.(tab)
      const next = prev.filter((t) => t.id !== id)
      if (id === activeTab) {
        const idx = prev.findIndex((t) => t.id === id)
        const targetIdx = Math.max(0, idx - 1)
        setActiveTab(next[targetIdx]?.id ?? "")
      }
      return next
    })
  }, [activeTab, onCloseTab])

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

  const startRename = useCallback((id: string) => {
    setEditingTab(id)
  }, [])

  const commitRename = useCallback((id: string, value: string) => {
    setEditingTab(null)
    setCustomNames((prev) => {
      const next = { ...prev }
      if (value.trim()) {
        next[id] = value.trim()
      } else {
        next[id] = null
      }
      return next
    })
    setTabs((prev) =>
      prev.map((t) => (t.id === id ? { ...t, name: value.trim() || null } : t))
    )
  }, [])

  const updateModeLabel = useCallback((id: string, label: string, loading = false) => {
    setModeLabels((prev) => (prev[id] === label ? prev : { ...prev, [id]: label }))
    setModeLoading((prev) => (prev[id] === loading ? prev : { ...prev, [id]: loading }))
  }, [])
  // One stable label callback per tab. Consumers list it as an effect dependency,
  // so a fresh closure per render re-fires them all, and two producers in one
  // tab (the chat mode label plus an open document's filename) then overwrite
  // each other forever.
  const modeLabelHooks = useRef(new Map<string, (label: string, loading?: boolean) => void>())

  const strip = useMemo<TabStripValue>(() => ({
    tabs,
    activeTabId: activeTab,
    editingTabId: editingTab,
    labelOf: (tab) => {
      const label = modeLabels[tab.id]
      // "\0" is a sentinel meaning the tab renders no label at all
      if (label === "\0") return null
      return customNames[tab.id] ?? (label || "Chat")
    },
    isLoading: (tab) => customNames[tab.id] == null && (modeLoading[tab.id] ?? false),
    select: setActiveTab,
    close: closeTab,
    add: addTab,
    startRename: setEditingTab,
    commitRename,
    cancelRename: () => setEditingTab(null),
  }), [tabs, activeTab, editingTab, modeLabels, modeLoading, customNames, closeTab, addTab, commitRename])

  return (
    <TabStripContext.Provider value={strip}>
      <div className="relative h-screen overflow-hidden">
        {tabs.map((tab) => {
          const isActive = tab.id === activeTab
          let hook = modeLabelHooks.current.get(tab.id)
          if (!hook) {
            hook = (label: string, loading?: boolean) => updateModeLabel(tab.id, label, loading)
            modeLabelHooks.current.set(tab.id, hook)
          }
          return (
            <div
              key={tab.id}
              className="h-full"
              style={{ display: isActive ? undefined : "none" }}
            >
              <TabActiveProvider value={isActive}>
                {renderContent(tab, hook, isActive, (config) => {
                  setTabs((prev) => prev.map((t) => t.id === tab.id ? { ...t, config: { ...t.config, ...config } } : t))
                }, (mode) => {
                  setTabs((prev) => prev.map((t) => t.id === tab.id ? { ...t, appMode: mode } : t))
                })}
              </TabActiveProvider>
            </div>
          )
        })}
      </div>
    </TabStripContext.Provider>
  )
}

