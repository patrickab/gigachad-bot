"use client"

import { useCallback, useEffect, useEffectEvent, useRef, useState } from "react"
import type { ChatFocus } from "@/features/chat/ChatContainer"
import { MemoryViewer } from "@/features/memory/MemoryViewer"
import { ProjectDashboard } from "@/features/projects/ProjectDashboard"
import { TabManager, nextTab, type TabManagerHandle } from "@/features/navigation/TabManager"
import { BranchProvider, useBranches } from "@/features/navigation/BranchContext"
import { EnabledToolsProvider } from "@/features/chat/EnabledToolsContext"
import { MemoryViewerProvider } from "@/features/memory/MemoryViewerContext"
import { FilePanelProvider } from "@/features/editor/FilePanel"
import { useProject } from "@/contexts/ProjectContext"
import { useSettings } from "@/contexts/SettingsContext"
import { SidebarProvider } from "@/features/navigation/SidebarContext"
import { OmpProvider } from "@/features/omp/OmpProvider"
import { surfaceOf, type Tab, type WorkspaceItem } from "@/contexts/TabContext"
import { TabContent } from "./TabContent"
import { parseOpenParam, withoutOpenParam } from "./deepLink"

/** Keep each project's list entry (saved chat labels) current: histories refresh after every chat save. */
function ProjectListSync() {
  const { branchMeta } = useBranches()
  const { refreshProjects } = useProject()
  const seen = useRef(branchMeta)
  useEffect(() => {
    if (seen.current === branchMeta) return
    seen.current = branchMeta
    void refreshProjects().catch(() => {})
  }, [branchMeta, refreshProjects])
  return null
}

/** Keep project navigation and memory shared while each tab retains its own workspace. */
export function AppShell() {
  const tabManagerRef = useRef<TabManagerHandle>(null)
  const lastProjectTabs = useRef(new Map<string, string>())
  const activeTabRef = useRef<Tab | null>(null)
  const [focus, setFocus] = useState<ChatFocus>({ qaIndex: null, key: 0 })
  // Project whose dashboard is showing; set from the current tab and never read back as "the" project.
  const [dashboardSlug, setDashboardSlug] = useState<string | null>(null)
  const settings = useSettings()

  const handleActiveTabChange = useCallback((tab: Tab) => {
    activeTabRef.current = tab
    if (tab.projectSlug) lastProjectTabs.current.set(tab.projectSlug, tab.id)
    setDashboardSlug((current) => current === tab.projectSlug ? current : null)
  }, [])

  const handleNewChat = useCallback((slug: string | null) => {
    const tab = nextTab(settings.tabDefaults)
    tab.projectSlug = slug
    tabManagerRef.current?.openTab(tab)
  }, [settings.tabDefaults])

  const handleOpenProject = useCallback((slug: string) => {
    const manager = tabManagerRef.current
    if (!manager) return
    const tabs = manager.getTabs()
    const remembered = tabs.find((tab) => tab.id === lastProjectTabs.current.get(slug) && tab.projectSlug === slug)
    const existing = remembered ?? tabs.findLast((tab) => tab.projectSlug === slug)
    if (existing) manager.switchToTab(existing.id)
    else handleNewChat(slug)
  }, [handleNewChat])

  const handleOpenChat = useCallback((historyFile: string, qaIndex?: number) => {
    setFocus((current) => ({ qaIndex: qaIndex ?? null, key: current.key + 1 }))
    tabManagerRef.current?.showChat(historyFile)
  }, [])

  const handleOpenItem = useCallback((item: WorkspaceItem, inNewTab = false) => {
    if (!inNewTab) {
      tabManagerRef.current?.showItem(item)
      return
    }
    const tab = nextTab(settings.tabDefaults)
    tabManagerRef.current?.openTab({ ...tab, workspaceItem: item, projectSlug: item.projectSlug, appMode: surfaceOf(item) })
  }, [settings.tabDefaults])

  // A factory preview opens its scenario's chat, canvas, document or architecture through `?open=`, once at startup.
  const openFromLink = useEffectEvent(() => {
    const target = parseOpenParam(window.location.search)
    if (!target) return
    window.history.replaceState(window.history.state, "", withoutOpenParam(new URL(window.location.href)))
    if (target.kind === "chat") handleOpenChat(target.historyFile)
    else handleOpenItem(target.item)
  })
  // eslint-disable-next-line react-hooks/set-state-in-effect -- opening the linked item is a one-shot startup action that must follow the commit, once the tab manager's handle exists
  useEffect(() => openFromLink(), [])

  const handleHistoriesMoved = useCallback((moves: ReadonlyMap<string, string>) => {
    tabManagerRef.current?.relocateHistories(moves)
  }, [])

  const handleProjectRenamed = useCallback((oldSlug: string, newSlug: string) => {
    tabManagerRef.current?.relocateProject(oldSlug, newSlug)
  }, [])

  useEffect(() => {
    function handler(event: KeyboardEvent) {
      if (!event.altKey || event.key !== "d") return
      event.preventDefault()
      const slug = activeTabRef.current?.projectSlug ?? null
      setDashboardSlug((current) => current ? null : slug)
    }
    document.addEventListener("keydown", handler)
    return () => document.removeEventListener("keydown", handler)
  }, [])

  const closeDashboard = useCallback(() => setDashboardSlug(null), [])

  return (
    <BranchProvider onFileClick={handleOpenChat} onHistoriesMoved={handleHistoriesMoved}>
      <SidebarProvider>
        <MemoryViewerProvider>
          <OmpProvider>
            <TabManager ref={tabManagerRef} onActiveTabChange={handleActiveTabChange} defaultConfig={settings.tabDefaults}>
              <EnabledToolsProvider>
                <FilePanelProvider>
                  <TabContent onOpenChat={handleOpenChat} onOpenProject={handleOpenProject} onProjectRenamed={handleProjectRenamed} onNewChat={handleNewChat} onOpenItem={handleOpenItem} focus={focus} />
                </FilePanelProvider>
              </EnabledToolsProvider>
            </TabManager>
          </OmpProvider>
          {dashboardSlug && <ProjectDashboard key={dashboardSlug} slug={dashboardSlug} onClose={closeDashboard} />}
          <ProjectListSync />
          <MemoryViewer />
        </MemoryViewerProvider>
      </SidebarProvider>
    </BranchProvider>
  )
}
