"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import type { ChatFocus } from "@/features/chat/ChatContainer"
import { MemoryViewer } from "@/features/memory/MemoryViewer"
import { ProjectDashboard } from "@/features/projects/ProjectDashboard"
import { TabManager, nextTab, type TabManagerHandle } from "@/features/navigation/TabManager"
import { BranchProvider } from "@/features/navigation/BranchContext"
import { EnabledToolsProvider } from "@/features/chat/EnabledToolsContext"
import { MemoryViewerProvider } from "@/features/memory/MemoryViewerContext"
import { useProject } from "@/contexts/ProjectContext"
import { useSettings } from "@/contexts/SettingsContext"
import { SidebarProvider } from "@/features/navigation/SidebarContext"
import type { Tab, WorkspaceItem } from "@/contexts/TabContext"
import { parseHistoryFile } from "@/lib/domain/historyFile"
import { TabContent } from "./TabContent"

/** Keep project navigation and memory shared while each tab retains its own workspace. */
export function AppShell() {
  const tabManagerRef = useRef<TabManagerHandle>(null)
  const lastProjectTabs = useRef(new Map<string, string>())
  const requestedProject = useRef<string | null | undefined>(undefined)
  const [focus, setFocus] = useState<ChatFocus>({ qaIndex: null, key: 0 })
  const { activeProject, dashboardOpen, syncTabs, setDashboardOpen, openProject, closeProject } = useProject()
  const settings = useSettings()

  const handleActiveTabChange = useCallback((tab: Tab) => {
    if (tab.projectSlug) lastProjectTabs.current.set(tab.projectSlug, tab.id)
    if (requestedProject.current === tab.projectSlug) return
    requestedProject.current = tab.projectSlug
    if (tab.projectSlug) {
      void openProject(tab.projectSlug).catch(() => { requestedProject.current = undefined })
    } else {
      closeProject()
    }
  }, [openProject, closeProject])

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
    const manager = tabManagerRef.current
    if (!manager) return
    const existing = manager.getTabs().find((tab) => tab.historyFile === historyFile)
    if (existing) {
      manager.showChat(existing.id)
      return
    }
    const tab = nextTab(settings.tabDefaults)
    tab.historyFile = historyFile
    tab.projectSlug = parseHistoryFile(historyFile).slug
    manager.openTab(tab)
  }, [settings.tabDefaults])

  const handleOpenItem = useCallback((item: WorkspaceItem) => {
    const manager = tabManagerRef.current
    if (!manager) return
    const existing = manager.getTabs().find((tab) => tab.workspaceItem?.kind === item.kind
      && tab.projectSlug === item.projectSlug
      && (item.kind === "architecture" || tab.workspaceItem.path === item.path)
      && !!tab.workspaceItem.rootFile === !!item.rootFile)
    if (existing?.workspaceItem) {
      manager.showItem(existing.id, item.kind === "architecture" ? existing.workspaceItem : item)
      return
    }
    const tab = nextTab(settings.tabDefaults)
    tab.workspaceItem = item
    tab.projectSlug = item.projectSlug
    tab.appMode = item.kind === "document" ? "document" : "canvas"
    manager.openTab(tab)
  }, [settings.tabDefaults])

  const handleTabsChange = useCallback((tabs: Tab[]) => {
    if (!activeProject) return
    void syncTabs(tabs
      .filter((tab) => tab.projectSlug === activeProject && tab.historyFile?.startsWith(`${activeProject}/`))
      .map((tab) => ({ id: tab.id, name: tab.name, historyFile: tab.historyFile, title: tab.title, chatId: tab.chatId })),
    ).catch(() => {})
  }, [activeProject, syncTabs])

  useEffect(() => {
    function handler(event: KeyboardEvent) {
      if (event.altKey && event.key === "d") { event.preventDefault(); setDashboardOpen(!dashboardOpen) }
    }
    document.addEventListener("keydown", handler)
    return () => document.removeEventListener("keydown", handler)
  }, [setDashboardOpen, dashboardOpen])

  return (
    <BranchProvider onFileClick={handleOpenChat}>
      <SidebarProvider>
        <MemoryViewerProvider>
          <TabManager ref={tabManagerRef} onTabsChange={handleTabsChange} onActiveTabChange={handleActiveTabChange} defaultConfig={settings.tabDefaults}>
            <EnabledToolsProvider>
              <TabContent onOpenChat={handleOpenChat} onOpenProject={handleOpenProject} onNewChat={handleNewChat} onOpenItem={handleOpenItem} focus={focus} />
            </EnabledToolsProvider>
          </TabManager>
          {dashboardOpen && <ProjectDashboard />}
          <MemoryViewer />
        </MemoryViewerProvider>
      </SidebarProvider>
    </BranchProvider>
  )
}
