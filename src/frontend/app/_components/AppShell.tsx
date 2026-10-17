"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import type { ChatFocus } from "@/components/ChatContainer"
import { MemoryViewer } from "@/components/MemoryViewer"
import { ProjectDashboard } from "@/components/ProjectDashboard"
import { TabManager, nextTab, type TabManagerHandle } from "@/components/TabManager"
import { BranchProvider } from "@/contexts/BranchContext"
import { EnabledToolsProvider } from "@/contexts/EnabledToolsContext"
import { MemoryViewerProvider } from "@/contexts/MemoryViewerContext"
import { useProject } from "@/contexts/ProjectContext"
import { useSettings } from "@/contexts/SettingsContext"
import { SidebarProvider } from "@/contexts/SidebarContext"
import type { Tab } from "@/contexts/TabContext"
import { TabContent } from "./TabContent"

/** The tab strip with every tab's workspace, plus the cross-tab project dashboard, memory viewer and shortcuts. */
export function AppShell() {
  const tabManagerRef = useRef<TabManagerHandle>(null)
  const [focus, setFocus] = useState<ChatFocus>({ qaIndex: null, key: 0 })
  const { activeProject, dashboardOpen, syncTabs, setDashboardOpen, closeProject } = useProject()
  const settings = useSettings()

  const handleOpenChat = useCallback((historyFile: string, qaIndex?: number) => {
    setFocus((f) => {
      if (qaIndex != null) return { qaIndex, key: f.key + 1 }
      return f.qaIndex == null ? f : { ...f, qaIndex: null }
    })
    const tabs = tabManagerRef.current?.getTabs() ?? []
    const activeTabId = tabManagerRef.current?.getActiveTabId()
    const activeTab = tabs.find((t) => t.id === activeTabId)

    // toggle: clicking the active chat swaps only this tab for a blank one
    if (activeTab && activeTab.historyFile === historyFile && qaIndex == null) {
      tabManagerRef.current?.replaceTab(activeTab.id, nextTab(settings.tabDefaults))
      return
    }

    const existing = tabs.find((t) => t.historyFile === historyFile)
    if (existing) {
      tabManagerRef.current?.switchToTab(existing.id)
      return
    }
    // A chat outside the open project leaves that project; open tabs stay as they are.
    if (activeProject && !historyFile.startsWith(`${activeProject}/`)) closeProject()
    if (activeTabId) {
      tabManagerRef.current?.updateTabHistoryFile(activeTabId, historyFile)
    }
  }, [activeProject, closeProject, settings.tabDefaults])

  // Only the open project's chats belong in its saved tabs; tabs from elsewhere stay open but unsaved there.
  const handleTabsChange = useCallback((tabs: Tab[]) => {
    if (!activeProject) return
    syncTabs(
      tabs
        .filter((t) => t.historyFile?.startsWith(`${activeProject}/`))
        .map((t) => ({ id: t.id, name: t.name, historyFile: t.historyFile, title: t.title, chatId: t.chatId })),
    ).catch(() => { })
  }, [activeProject, syncTabs])

  useEffect(() => {
    function handler(e: KeyboardEvent) {
      if (e.altKey && e.key === "d") { e.preventDefault(); setDashboardOpen(!dashboardOpen) }
    }
    document.addEventListener("keydown", handler)
    return () => document.removeEventListener("keydown", handler)
  }, [setDashboardOpen, dashboardOpen])

  return (
    <BranchProvider onFileClick={handleOpenChat}>
      <SidebarProvider>
        <MemoryViewerProvider>
          <TabManager ref={tabManagerRef} onTabsChange={handleTabsChange} defaultConfig={settings.tabDefaults}>
            <EnabledToolsProvider>
              <TabContent onOpenChat={handleOpenChat} focus={focus} />
            </EnabledToolsProvider>
          </TabManager>
          {dashboardOpen && <ProjectDashboard />}
          <MemoryViewer />
        </MemoryViewerProvider>
      </SidebarProvider>
    </BranchProvider>
  )
}
