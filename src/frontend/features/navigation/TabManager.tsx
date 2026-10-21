"use client"

import { createContext, useCallback, useContext, useEffect, useId, useImperativeHandle, useMemo, useRef, useState, useSyncExternalStore } from "react"
import { ArrowRightLeft, Check, ChevronDown, Plus, X } from "lucide-react"
import { cn } from "@/lib/utils"
import { DEFAULT_TAB_CONFIG, type TabConfig } from "@/contexts/SettingsContext"
import { TabContext, surfaceOf, useTab, type Tab, type TabActions, type TabContextValue, type WorkspaceItem } from "@/contexts/TabContext"
import { baseName } from "@/lib/paths"
import { parseHistoryFile } from "@/lib/domain/historyFile"
import { useProject } from "@/contexts/ProjectContext"
import { useClickOutside } from "@/hooks/useClickOutside"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"

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
}

/** Tab state and actions shared by toolbar workspace switchers. */
interface TabStripValue {
  tabs: Tab[]
  activeTabId: string
  /** Activity description exposed alongside the workspace breadcrumb. */
  labelOf: (tab: Tab) => string
  isLoading: (tab: Tab) => boolean
  select: (id: string) => void
  close: (id: string) => void
  add: () => void
}
const TabStripContext = createContext<TabStripValue | null>(null)


export function nextTab(config?: Partial<TabConfig>): Tab {
  return { id: `tab-${crypto.randomUUID()}`, name: null, chatId: crypto.randomUUID(), historyFile: null, title: null, config: { ...DEFAULT_TAB_CONFIG, ...config }, appMode: "chat", projectSlug: null, workspaceItem: null }
}

const INITIAL_TAB: Tab = nextTab()
const WORKSPACE_SWITCH_MS = 160

function subscribeReducedMotion(onChange: () => void) {
  const media = window.matchMedia?.("(prefers-reduced-motion: reduce)")
  media?.addEventListener("change", onChange)
  return () => media?.removeEventListener("change", onChange)
}

function prefersReducedMotion() {
  return window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? true
}

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
      setWorkspaceItem: (workspaceItem) => updateTab(id, (tab) => ({
        workspaceItem,
        projectSlug: workspaceItem ? workspaceItem.projectSlug : tab.projectSlug,
        appMode: workspaceItem ? surfaceOf(workspaceItem) : "canvas",
      })),
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

/** Describe the visible surface, not a background item retained by the tab. */
function tabLocation(tab: Tab) {
  if (tab.appMode === "omp") {
    return { category: tab.projectSlug ? "code" : "omp", file: tab.name ?? tab.title ?? "omp" }
  }
  if (tab.appMode === "factory") {
    return { category: "factory", file: tab.name ?? (tab.workspaceItem?.path.includes("/") ? "task" : tab.workspaceItem?.path ?? "board") }
  }
  if (tab.appMode === "canvas") {
    return tab.workspaceItem?.kind === "architecture"
      ? { category: "architecture", file: tab.name ?? "System overview" }
      : { category: "canvases", file: tab.name ?? (tab.workspaceItem?.kind === "canvas" ? baseName(tab.workspaceItem.path) : "new.canvas") }
  }
  if (tab.appMode === "document" && tab.workspaceItem?.kind === "diff") {
    return { category: "diff", file: tab.name ?? baseName(tab.workspaceItem.path) }
  }
  if (tab.appMode === "document") {
    return { category: "files", file: tab.name ?? (tab.workspaceItem ? baseName(tab.workspaceItem.path) : "New file") }
  }
  return { category: "chats", file: tab.name ?? tab.title ?? (tab.historyFile ? baseName(tab.historyFile) : "New chat") }
}

const TAB_PATH_COLLATOR = new Intl.Collator("en", { numeric: true, sensitivity: "base" })

/** Reveal centered workspace navigation on hover, keyboard focus, or while its menu is open. */
export function TabSwitcher() {
  const strip = useContext(TabStripContext)
  const { tab, isActive } = useTab()
  const { projects } = useProject()
  const activityDescriptionId = useId()
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState("")
  const [showSearch, setShowSearch] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)
  const triggerRef = useRef<HTMLButtonElement>(null)
  const panelRef = useRef<HTMLDivElement>(null)
  const close = useCallback(() => {
    setOpen(false)
    setQuery("")
  }, [])
  useClickOutside(rootRef, close)
  useEscapeLayer(() => {
    close()
    triggerRef.current?.focus()
  }, open && isActive)
  useEffect(() => {
    if (!open || !isActive) return
    const panel = panelRef.current
    const current = panel?.querySelector<HTMLElement>("[aria-current=page]")
    current?.scrollIntoView?.({ block: "nearest" })
    const input = panel?.querySelector<HTMLInputElement>("input")
    const focusTarget = input ?? current
    focusTarget?.focus()
  }, [open, isActive])
  if (!strip) return null

  const projectName = (slug: string | null) => projects.find((project) => project.slug === slug)?.name ?? slug ?? "general"
  const location = tabLocation(tab)
  const breadcrumb = `${projectName(tab.projectSlug)} / ${location.category} / ${location.file}`
  const search = query.trim().toLowerCase()
  const orderedTabs = strip.tabs.map((item) => ({ item, project: projectName(item.projectSlug), ...tabLocation(item) }))
    .sort((left, right) => Number(left.project === "general") - Number(right.project === "general")
      || TAB_PATH_COLLATOR.compare(left.project, right.project)
      || TAB_PATH_COLLATOR.compare(left.category, right.category)
      || TAB_PATH_COLLATOR.compare(left.file, right.file)
      || left.item.id.localeCompare(right.item.id))
  const visibleTabs = orderedTabs.filter((entry) => `${entry.project} ${entry.category} ${entry.file}`.toLowerCase().includes(search))
  const nextTab = orderedTabs[(orderedTabs.findIndex(({ item }) => item.id === strip.activeTabId) + 1) % orderedTabs.length].item

  return <div ref={rootRef} data-tab-switcher data-open={open} className="pointer-events-auto flex w-[28rem] max-w-full flex-col items-center opacity-0 transition-opacity duration-160 [&:hover]:opacity-100 focus-within:opacity-100 data-[open=true]:opacity-100 motion-reduce:transition-none"
    onBlur={(event) => { if (!event.currentTarget.contains(event.relatedTarget)) close() }}>
    <div className="relative flex min-w-0 max-w-[calc(100%_-_4.75rem)]">
    <button ref={triggerRef} type="button" aria-label={`Switch tabs: ${breadcrumb}`} aria-haspopup="dialog" aria-expanded={open}
      aria-describedby={activityDescriptionId}
      onClick={() => {
        if (open) close()
        else {
          setShowSearch(strip.tabs.length > 8)
          setOpen(true)
        }
      }}
      className="flex h-8.5 max-w-full min-w-0 items-center gap-1.5 rounded-full bg-paper px-2 text-xs text-ink-muted transition-colors hover:bg-surface-elevated hover:text-ink focus:bg-surface-elevated focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-subtle">
      <span className="flex min-w-0 items-center gap-1.5">
          <span data-workspace-breadcrumb className="flex min-w-0 items-center gap-1.5 whitespace-nowrap">
            <span className="max-w-24 truncate text-ink-subtle">{projectName(tab.projectSlug)}</span>
            <span aria-hidden="true" className="text-ink-faint">/</span>
            <span className="shrink-0 text-ink-subtle">{location.category}</span>
            <span aria-hidden="true" className="text-ink-faint">/</span>
            <span className="min-w-0 truncate text-ink">{location.file}</span>
          </span>
          <ChevronDown aria-hidden="true" className="h-3 w-3 shrink-0 text-ink-subtle" />
        </span>
    </button>
      <button type="button" data-new-chat aria-label="New chat" onClick={() => {
        close()
        strip.add()
      }} className="absolute left-full ml-1 flex h-8.5 w-8.5 items-center justify-center rounded-full bg-paper text-ink-muted transition-colors hover:bg-surface-elevated hover:text-ink focus:bg-surface-elevated focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-subtle">
        <Plus aria-hidden="true" className="h-3 w-3" />
      </button>
    </div>
      <button type="button" aria-label="Next tab" disabled={strip.tabs.length < 2}
        onClick={() => {
          close()
          strip.select(nextTab.id)
        }}
        className="flex h-8.5 w-8.5 shrink-0 items-center justify-center rounded-full bg-paper text-ink-muted transition-colors hover:bg-surface-elevated hover:text-ink focus:bg-surface-elevated focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-subtle disabled:cursor-default disabled:text-ink-faint">
        <ArrowRightLeft aria-hidden="true" className="h-3 w-3" />
      </button>
    <span id={activityDescriptionId} className="sr-only">{strip.labelOf(tab)}</span>
    {open && isActive && <div ref={panelRef} role="dialog" aria-label="Open tabs"
      className="absolute left-1/2 top-full z-50 mt-1 flex w-max max-w-[min(36rem,100%)] -translate-x-1/2 flex-col overflow-hidden rounded-xl border border-divider bg-surface shadow-[var(--shadow-lg)]">
      {showSearch && <div className="border-b border-divider p-2">
        <input aria-label="Search open tabs" placeholder="Find a tab…" value={query} onChange={(event) => setQuery(event.target.value)}
          className="w-full rounded-md bg-paper px-2 py-1.5 text-xs text-ink outline-none focus-visible:ring-1 focus-visible:ring-ink-faint" />
      </div>}
      <div role="group" aria-label="Workspaces" className="max-h-[min(28rem,60vh)] overflow-y-auto p-1.5">
          {visibleTabs.map(({ item, project, category, file }) => {
            const active = item.id === strip.activeTabId
            return <div key={item.id} className={cn("group/tab flex items-center rounded-md transition-colors duration-160 motion-reduce:transition-none", active ? "bg-surface-elevated" : "hover:bg-surface-elevated/60")}>
              <button type="button" aria-label={`Switch to ${project} / ${category} / ${file}`} aria-current={active ? "page" : undefined}
                aria-describedby={`${activityDescriptionId}-${item.id}`}
                onClick={() => {
                  close()
                  strip.select(item.id)
                }}
                className="flex min-w-0 flex-1 items-center gap-2 rounded-md px-2 py-2 text-left text-xs text-ink-muted focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-subtle">
                <span className="min-w-0 flex-1 truncate whitespace-nowrap"><span className="text-ink-subtle">{project} / {category} / </span><span className={active ? "text-ink" : undefined}>{file}</span></span>
                <span id={`${activityDescriptionId}-${item.id}`} className="sr-only">{strip.labelOf(item)}</span>
                {strip.isLoading(item) && <Spinner />}
                {active && <Check aria-hidden="true" className="h-3 w-3 shrink-0 text-ink-subtle" />}
              </button>
              {strip.tabs.length > 1 && <button type="button" aria-label={`Close ${project} / ${category} / ${file}`}
                onClick={() => {
                  if (active) close()
                  else {
                    const target = panelRef.current?.querySelector<HTMLInputElement>("input") ?? triggerRef.current
                    target?.focus()
                  }
                  strip.close(item.id)
                }}
                className="mr-1 rounded p-1 text-ink-faint opacity-0 hover:text-ink group-hover/tab:opacity-100 group-focus-within/tab:opacity-100 focus-visible:opacity-100">
                <X aria-hidden="true" className="h-3 w-3" />
              </button>}
            </div>
          })}
        {visibleTabs.length === 0 && <p className="px-2 py-3 text-xs text-ink-subtle">No matching tabs</p>}
      </div>
    </div>}
  </div>
}


export function TabManager({ children, onTabsChange, onActiveTabChange, defaultConfig = DEFAULT_TAB_CONFIG, ref }: TabManagerProps) {
  const [tabs, setTabs] = useState<Tab[]>([INITIAL_TAB])
  const reduceMotion = useSyncExternalStore(subscribeReducedMotion, prefersReducedMotion, () => true)
  const [{ activeTab, outgoingTab }, setVisibleTabs] = useState<{ activeTab: string; outgoingTab: string | null }>({ activeTab: INITIAL_TAB.id, outgoingTab: null })
  const [modeLabels, setModeLabels] = useState<Record<string, string>>({})
  const [modeLoading, setModeLoading] = useState<Record<string, boolean>>({})
  const onTabsChangeRef = useRef(onTabsChange)
  useEffect(() => { onTabsChangeRef.current = onTabsChange }, [onTabsChange])

  const setActiveTab = useCallback((next: React.SetStateAction<string>) => {
    setVisibleTabs((current) => {
      const id = typeof next === "function" ? next(current.activeTab) : next
      if (id === current.activeTab) return current
      return { activeTab: id, outgoingTab: reduceMotion ? null : current.activeTab }
    })
  }, [reduceMotion])

  useEffect(() => {
    if (!outgoingTab || !reduceMotion) return
    const timer = window.setTimeout(() => {
      setVisibleTabs((current) => current.activeTab === activeTab && current.outgoingTab === outgoingTab
        ? { ...current, outgoingTab: null }
        : current)
    }, 0)
    return () => window.clearTimeout(timer)
  }, [activeTab, outgoingTab, reduceMotion])

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
        workspaceItem: item.kind === "architecture" && tab.workspaceItem?.kind === "architecture"
          && tab.workspaceItem.projectSlug === item.projectSlug ? tab.workspaceItem : item,
        projectSlug: item.projectSlug,
        appMode: surfaceOf(item),
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
  }, [activeTab, setActiveTab])

  const closeActiveTab = useCallback(() => {
    closeTab(activeTab)
  }, [activeTab, closeTab])

  const switchToNextTab = useCallback(() => {
    setActiveTab((current) => {
      const idx = tabs.findIndex((t) => t.id === current)
      const next = (idx + 1) % tabs.length
      return tabs[next].id
    })
  }, [tabs, setActiveTab])

  const switchToPrevTab = useCallback(() => {
    setActiveTab((current) => {
      const idx = tabs.findIndex((t) => t.id === current)
      const prev = (idx - 1 + tabs.length) % tabs.length
      return tabs[prev].id
    })
  }, [tabs, setActiveTab])

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
        if (!isEditable && !target.closest("[data-file-tree], [data-tab-switcher]")) {
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


  const updateModeLabel = useCallback((id: string, label: string, loading = false) => {
    setModeLabels((prev) => (prev[id] === label ? prev : { ...prev, [id]: label }))
    setModeLoading((prev) => (prev[id] === loading ? prev : { ...prev, [id]: loading }))
  }, [])

  const strip = useMemo<TabStripValue>(() => ({
    tabs,
    activeTabId: activeTab,
    labelOf: (tab) => {
      if (tab.appMode !== "chat" && tab.workspaceItem) {
        if (tab.appMode === "omp") return tab.name ?? tab.title ?? "omp"
        if (tab.appMode === "factory") return tab.name ?? "Factory"
        return tab.name ?? (tab.workspaceItem.kind === "architecture" ? "Architecture" : baseName(tab.workspaceItem.path))
      }
      const label = modeLabels[tab.id]
      return tab.name ?? (label || "Chat")
    },
    isLoading: (tab) => tab.name == null && (modeLoading[tab.id] ?? false),
    select: setActiveTab,
    close: closeTab,
    add: addTab,
  }), [tabs, activeTab, modeLabels, modeLoading, closeTab, addTab, setActiveTab])

  return (
    <TabStripContext.Provider value={strip}>
      <div className="relative h-screen overflow-hidden" style={{ "--workspace-switch-duration": `${WORKSPACE_SWITCH_MS}ms` } as React.CSSProperties}>
        {tabs.map((tab) => (
          <div
            key={tab.id}
            className="workspace-tab absolute inset-0 h-full"
            data-phase={tab.id === activeTab ? outgoingTab && !reduceMotion ? "incoming" : "idle" : "outgoing"}
            aria-hidden={tab.id !== activeTab}
            inert={tab.id !== activeTab}
            onAnimationEnd={(event) => {
              if (tab.id !== activeTab || event.animationName !== "workspace-dissolve-in") return
              setVisibleTabs((current) => current.activeTab === tab.id ? { ...current, outgoingTab: null } : current)
            }}
            style={{ display: tab.id === activeTab || tab.id === outgoingTab && !reduceMotion ? undefined : "none" }}
          >
            <TabScope key={tab.chatId} tab={tab} isActive={tab.id === activeTab} updateTab={updateTab} updateModeLabel={updateModeLabel}>
              {children}
            </TabScope>
          </div>
        ))}
      </div>
    </TabStripContext.Provider>
  )
}

