"use client"

import { useCallback, useEffect, useImperativeHandle, useRef, useState } from "react"
import { Plus, X } from "lucide-react"
import { cn } from "@/lib/utils"
import { DEFAULT_MODEL, DEFAULT_TEMPERATURE } from "@/lib/config"
import type { SettingsState } from "@/contexts/SettingsContext"

export interface TabConfig {
  selectedModel: string
  selectedPrompt: string | null
  temperature: number
  reasoningEffort: string
  downscaleImages: boolean
  researchFastModel: string
  researchSmartModel: string
  researchStrategicModel: string
  researchDepth: number
  researchBreadth: number
  researchReasoning: string
  researchReportType: string
  searchDepth: "quick" | "adaptive"
}

const TAB_CONFIG_KEYS: (keyof TabConfig)[] = [
  "selectedModel", "selectedPrompt", "temperature", "reasoningEffort",
  "downscaleImages", "researchFastModel", "researchSmartModel", "researchStrategicModel",
  "researchDepth", "researchBreadth", "researchReasoning", "researchReportType", "searchDepth",
]

const DEFAULT_CONFIG: TabConfig = {
  selectedModel: DEFAULT_MODEL,
  selectedPrompt: null,
  temperature: DEFAULT_TEMPERATURE,
  reasoningEffort: "none",
  downscaleImages: true,
  researchFastModel: "",
  researchSmartModel: "",
  researchStrategicModel: "",
  researchDepth: 2,
  researchBreadth: 4,
  researchReasoning: "medium",
  researchReportType: "deep",
  searchDepth: "adaptive",
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
}

interface TabManagerProps {
  renderContent: (tab: Tab, onModeLabel: (label: string) => void, isActive: boolean, onConfigChange: (config: Partial<TabConfig>) => void) => React.ReactNode
  onCloseTab?: (tab: Tab) => void
  onTabsChange?: (tabs: Tab[]) => void
  defaultConfig?: TabConfig
  ref?: React.Ref<TabManagerHandle>
}

export interface TabManagerHandle {
  initTabs: (tabs: Tab[]) => void
  switchToTab: (tabId: string) => void
  addTabWithName: (name: string | null, historyFile: string | null) => string
  getTabs: () => Tab[]
  getActiveTabId: () => string
  updateTabHistoryFile: (tabId: string, historyFile: string) => void
  updateTabTitle: (tabId: string, title: string | null) => void
  updateTabConfig: (tabId: string, config: Partial<TabConfig>) => void
}

let tabIdCounter = 0

export function nextTab(name: string | null = null, historyFile: string | null = null, title: string | null = null, config?: Partial<TabConfig>): Tab {
  return { id: `tab-${++tabIdCounter}`, name, chatId: crypto.randomUUID(), historyFile, title, config: { ...DEFAULT_CONFIG, ...config } }
}

const INITIAL_TAB: Tab = nextTab()
const INITIAL_CONFIG: TabConfig = { ...DEFAULT_CONFIG }

export function TabManager({ renderContent, onCloseTab, onTabsChange, defaultConfig = INITIAL_CONFIG, ref }: TabManagerProps) {
  const [tabs, setTabs] = useState<Tab[]>([INITIAL_TAB])
  const [activeTab, setActiveTab] = useState(INITIAL_TAB.id)
  const [editingTab, setEditingTab] = useState<string | null>(null)
  const [modeLabels, setModeLabels] = useState<Record<string, string>>({})
  const [customNames, setCustomNames] = useState<Record<string, string | null>>({})
  const inputRef = useRef<HTMLInputElement>(null)
  const initInProgressRef = useRef(false)
  const onTabsChangeRef = useRef(onTabsChange)
  useEffect(() => { onTabsChangeRef.current = onTabsChange }, [onTabsChange])

  useImperativeHandle(ref, () => ({
    initTabs: (newTabs: Tab[]) => {
      initInProgressRef.current = true
      setTabs(newTabs)
      if (newTabs.length > 0) setActiveTab(newTabs[0].id)
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
    updateTabHistoryFile: (tabId: string, historyFile: string) => {
      setTabs((prev) => prev.map((t) => t.id === tabId ? { ...t, historyFile } : t))
    },
    updateTabTitle: (tabId: string, title: string | null) => {
      setTabs((prev) => prev.map((t) => t.id === tabId ? { ...t, title } : t))
    },
    updateTabConfig: (tabId: string, config: Partial<TabConfig>) => {
      setTabs((prev) => prev.map((t) => t.id === tabId ? { ...t, config: { ...t.config, ...config } } : t))
    },
  }), [tabs, activeTab])

  useEffect(() => {
    if (editingTab) inputRef.current?.focus()
  }, [editingTab])

  useEffect(() => {
    if (initInProgressRef.current) {
      initInProgressRef.current = false
      return
    }
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

  const updateModeLabel = useCallback((id: string, label: string) => {
    setModeLabels((prev) => (prev[id] === label ? prev : { ...prev, [id]: label }))
  }, [])

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent, id: string, value: string) => {
      if (e.key === "Enter") commitRename(id, value)
      if (e.key === "Escape") setEditingTab(null)
    },
    [commitRename]
  )

  function getTabDisplayName(tab: Tab): string {
    const custom = customNames[tab.id]
    if (custom !== undefined && custom !== null) return custom
    return modeLabels[tab.id] || "Chat"
  }

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      <div className="h-8 shrink-0 flex items-center border-b border-divider/50 bg-paper select-none">
        <div className="flex-1 flex items-center h-full min-w-0">
          {tabs.map((tab) => {
            const isActive = tab.id === activeTab
            const isEditing = tab.id === editingTab
            const displayName = getTabDisplayName(tab)

            return (
              <div
                key={tab.id}
                className={cn(
                  "flex-1 min-w-0 h-full flex items-center justify-between px-3 gap-1 border-r border-divider/50 cursor-pointer transition-colors text-xs whitespace-nowrap",
                  isActive
                    ? "bg-surface/50 text-ink"
                    : "text-ink-subtle hover:text-ink hover:bg-surface/30"
                )}
                onClick={() => setActiveTab(tab.id)}
                onContextMenu={(e) => { e.preventDefault(); closeTab(tab.id) }}
                onDoubleClick={() => startRename(tab.id)}
              >
                {isEditing ? (
                  <input
                    ref={inputRef}
                    autoFocus
                    defaultValue={displayName}
                    className="flex-1 bg-transparent outline-none text-ink text-xs min-w-0"
                    onBlur={(e) => commitRename(tab.id, e.target.value)}
                    onKeyDown={(e) => handleKeyDown(e, tab.id, (e.target as HTMLInputElement).value)}
                    onClick={(e) => e.stopPropagation()}
                  />
                ) : (
                  <span className="truncate">{displayName}</span>
                )}
                {tabs.length > 1 && (
                  <button
                    className="shrink-0 p-0.5 rounded hover:bg-hover text-ink-faint hover:text-ink"
                    onClick={(e) => {
                      e.stopPropagation()
                      closeTab(tab.id)
                    }}
                  >
                    <X className="h-3 w-3" />
                  </button>
                )}
              </div>
            )
          })}
        </div>
        <button
          className="shrink-0 h-full px-2.5 flex items-center justify-center text-ink-subtle hover:text-ink hover:bg-surface-elevated/50 transition-colors"
          onClick={addTab}
        >
          <Plus className="h-3.5 w-3.5" />
        </button>
      </div>
      <div className="flex-1 min-h-0 overflow-hidden relative">
        {tabs.map((tab) => {
          const isActive = tab.id === activeTab
          const hook = (label: string) => updateModeLabel(tab.id, label)
          return (
            <div
              key={tab.id}
              className="h-full"
              style={{ display: isActive ? undefined : "none" }}
            >
              {renderContent(tab, hook, isActive, (config) => {
                setTabs((prev) => prev.map((t) => t.id === tab.id ? { ...t, config: { ...t.config, ...config } } : t))
              })}
            </div>
          )
        })}
      </div>
    </div>
  )
}
