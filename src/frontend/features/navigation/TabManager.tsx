"use client"

import { useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react"
import { isEditableTarget } from "@/lib/utils"
import { DEFAULT_TAB_CONFIG, type TabConfig } from "@/contexts/SettingsContext"
import { TabContext, showingItem, type Tab, type TabActions, type TabContextValue, type WorkspaceItem } from "@/contexts/TabContext"
import { parseHistoryFile } from "@/lib/domain/historyFile"
import { useEventListener } from "@/hooks/useEventListener"
import { TabStripContext, nextTab, relocatedTab, tabAfterClosing, tabLabel, type TabStripValue } from "./tabs"
import { WORKSPACE_SWITCH_MS, useWorkspaceSwitch } from "./useWorkspaceSwitch"

interface TabManagerProps {
  /** Rendered once per tab inside that tab's `TabContext`. */
  children: React.ReactNode
  onTabsChange?: (tabs: Tab[]) => void
  onActiveTabChange?: (tab: Tab) => void
  defaultConfig?: TabConfig
  ref?: React.Ref<TabManagerHandle>
}

export interface TabManagerHandle {
  switchToTab: (tabId: string) => void
  getTabs: () => Tab[]
  openTab: (tab: Tab) => void
  showChat: (historyFile: string) => void
  showItem: (item: WorkspaceItem) => void
  relocateHistories: (moves: ReadonlyMap<string, string>) => void
  /** Point every tab of project `oldSlug` at `newSlug`, its content then reloads from the moved keys. */
  relocateProject: (oldSlug: string, newSlug: string) => void
}

/** A tab's latest activity description, and whether the switcher shows a spinner for it. */
interface TabMode {
  label: string
  loading: boolean
}

const INITIAL_TAB: Tab = nextTab()

/** Provides one tab's state and actions; memoized so the context only changes with the tab or its active flag. */
function TabScope({ tab, isActive, updateTab, setMode, children }: {
  tab: Tab
  isActive: boolean
  updateTab: (id: string, patch: (tab: Tab) => Partial<Tab>) => void
  setMode: (id: string, label: string, loading?: boolean) => void
  children: React.ReactNode
}) {
  // One stable action set per tab. Consumers list these as effect dependencies, so a
  // fresh closure per render re-fires them all, and two mode-label producers in one
  // tab (the chat label plus an open document's filename) then overwrite each other forever.
  // TabScope is keyed by tab id, so the actions are built once for the tab's lifetime.
  const [actions] = useState<TabActions>(() => {
    const id = tab.id
    return {
      setModeLabel: (label, loading) => setMode(id, label, loading),
      updateConfig: (config) => updateTab(id, (t) => ({ config: { ...t.config, ...config } })),
      setAppMode: (appMode) => updateTab(id, () => ({ appMode })),
      setWorkspaceItem: (workspaceItem) => updateTab(id, () => workspaceItem ? showingItem(workspaceItem) : { workspaceItem: null, appMode: "canvas" }),
      setHistoryFile: (historyFile) => updateTab(id, () => ({ historyFile })),
      setTitle: (title) => updateTab(id, () => ({ title })),
    }
  })
  const value = useMemo<TabContextValue>(() => ({ tab, isActive, ...actions }), [tab, isActive, actions])
  return <TabContext.Provider value={value}>{children}</TabContext.Provider>
}

export function TabManager({ children, onTabsChange, onActiveTabChange, defaultConfig = DEFAULT_TAB_CONFIG, ref }: TabManagerProps) {
  const [tabs, setTabs] = useState<Tab[]>([INITIAL_TAB])
  const { activeTab, setActiveTab, phaseOf, isShown, onAnimationEnd } = useWorkspaceSwitch(INITIAL_TAB.id)
  const [modes, setModes] = useState<Record<string, TabMode>>({})
  const onTabsChangeRef = useRef(onTabsChange)
  useEffect(() => { onTabsChangeRef.current = onTabsChange }, [onTabsChange])

  const updateTab = useCallback((id: string, patch: (tab: Tab) => Partial<Tab>) => {
    setTabs((prev) => prev.map((t) => (t.id === id ? { ...t, ...patch(t) } : t)))
  }, [])

  useImperativeHandle(ref, () => ({
    switchToTab: (tabId: string) => setActiveTab(tabId),
    getTabs: () => tabs,
    openTab: (tab: Tab) => {
      setTabs((prev) => [...prev, tab])
      setActiveTab(tab.id)
    },
    showChat: (historyFile) => {
      updateTab(activeTab, (tab) => ({
        appMode: "chat",
        projectSlug: parseHistoryFile(historyFile).slug,
        ...(tab.historyFile !== historyFile ? {
          historyFile, chatId: crypto.randomUUID(), title: null, name: null, workspaceItem: null,
        } : {}),
      }))
    },
    showItem: (item) => {
      updateTab(activeTab, (tab) => ({
        ...showingItem(item),
        // Reopening the same project's architecture keeps the system it shows.
        workspaceItem: item.kind === "architecture" && tab.workspaceItem?.kind === "architecture"
          && tab.workspaceItem.projectSlug === item.projectSlug ? tab.workspaceItem : item,
      }))
    },
    relocateHistories: (moves) => {
      setTabs((current) => current.map((tab) => {
        const historyFile = tab.historyFile ? moves.get(tab.historyFile) : undefined
        return historyFile ? {
          ...tab, historyFile,
          projectSlug: tab.appMode === "chat" ? parseHistoryFile(historyFile).slug : tab.projectSlug,
        } : tab
      }))
    },
    relocateProject: (oldSlug, newSlug) => {
      setTabs((current) => current.map((tab) => relocatedTab(tab, oldSlug, newSlug)))
    },
  }), [tabs, activeTab, updateTab, setActiveTab])

  useEffect(() => {
    onTabsChangeRef.current?.(tabs)
  }, [tabs])

  useEffect(() => {
    const selected = tabs.find((tab) => tab.id === activeTab)
    if (selected) onActiveTabChange?.(selected)
  }, [tabs, activeTab, onActiveTabChange])

  const addTab = useCallback(() => {
    const tab = nextTab(defaultConfig)
    tab.projectSlug = tabs.find((current) => current.id === activeTab)?.projectSlug ?? null
    setTabs((prev) => [...prev, tab])
    setActiveTab(tab.id)
  }, [defaultConfig, tabs, activeTab, setActiveTab])

  const closeTab = useCallback((id: string) => {
    if (tabs.length <= 1) return
    setTabs((prev) => prev.filter((t) => t.id !== id))
    if (id === activeTab) setActiveTab(tabAfterClosing(tabs, id))
  }, [tabs, activeTab, setActiveTab])

  const cycleTab = useCallback((delta: 1 | -1) => {
    setActiveTab((current) => {
      const idx = tabs.findIndex((t) => t.id === current)
      return tabs[(idx + delta + tabs.length) % tabs.length].id
    })
  }, [tabs, setActiveTab])

  useEventListener("document", "keydown", (e) => {
    if (e.altKey && e.key === "t") {
      e.preventDefault()
      addTab()
    }
    if (e.altKey && e.key === "w") {
      e.preventDefault()
      closeTab(activeTab)
    }
    if (e.key === "Tab" && !e.altKey && !e.ctrlKey && !e.metaKey) {
      const target = e.target as HTMLElement
      if (!isEditableTarget(target) && !target.closest("[data-file-tree], [data-tab-switcher]")) {
        e.preventDefault()
        cycleTab(e.shiftKey ? -1 : 1)
      }
    }
  })

  const setMode = useCallback((id: string, label: string, loading = false) => {
    setModes((prev) => (prev[id]?.label === label && prev[id].loading === loading ? prev : { ...prev, [id]: { label, loading } }))
  }, [])

  const strip = useMemo<TabStripValue>(() => ({
    tabs,
    activeTabId: activeTab,
    labelOf: (tab) => tabLabel(tab, modes[tab.id]?.label),
    isLoading: (tab) => tab.name == null && (modes[tab.id]?.loading ?? false),
    select: setActiveTab,
    close: closeTab,
    add: addTab,
  }), [tabs, activeTab, modes, closeTab, addTab, setActiveTab])

  return (
    <TabStripContext.Provider value={strip}>
      <div className="relative h-screen overflow-hidden" style={{ "--workspace-switch-duration": `${WORKSPACE_SWITCH_MS}ms` } as React.CSSProperties}>
        {tabs.map((tab) => (
          <div
            key={tab.id}
            className="workspace-tab absolute inset-0 h-full"
            data-phase={phaseOf(tab.id)}
            aria-hidden={tab.id !== activeTab}
            inert={tab.id !== activeTab}
            onAnimationEnd={(event) => onAnimationEnd(tab.id, event.animationName)}
            style={{ display: isShown(tab.id) ? undefined : "none" }}
          >
            <TabScope key={tab.chatId} tab={tab} isActive={tab.id === activeTab} updateTab={updateTab} setMode={setMode}>
              {children}
            </TabScope>
          </div>
        ))}
      </div>
    </TabStripContext.Provider>
  )
}
