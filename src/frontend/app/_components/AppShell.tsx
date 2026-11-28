"use client"

import { useCallback, useEffect, useEffectEvent, useRef, useState } from "react"
import { installClickHints } from "@gigachad/chat-ui/hints"
import "@gigachad/chat-ui/hints.css"
import type { ChatReveal, ChatRevealHit } from "@/features/chat/ChatContainer"
import { MemoryViewer } from "@/features/memory/MemoryViewer"
import { ProjectDashboard } from "@/features/projects/ProjectDashboard"
import { TabManager, type TabManagerHandle } from "@/features/navigation/TabManager"
import { nextTab, piSessionOf, tabLabel } from "@/features/navigation/tabs"
import { BranchProvider } from "@/features/navigation/BranchContext"
import { EnabledToolsProvider } from "@/features/chat/EnabledToolsContext"
import { forgetComposerDraft, newChatDraftKey } from "@/features/chat/composer/useComposerDraft"
import { MemoryViewerProvider } from "@/features/memory/MemoryViewerContext"
import { ShellTerminals } from "@/features/editor/ShellTerminal"
import { useSettings } from "@/contexts/SettingsContext"
import { ProviderUsagePoll } from "@/features/navigation/ProviderUsage"
import { SidebarProvider } from "@/features/navigation/SidebarContext"
import { useEventListener } from "@/hooks/useEventListener"
import { LearnProvider } from "@/features/learn/LearnProvider"
import { markPiSessionRead, PiSessionsProvider } from "@/features/pi/sessionStore"
import { startPiNotifications } from "@/features/pi/notifications"
import { showingItem, type Tab, type WorkspaceItem } from "@/contexts/TabContext"
import { ContentAreaProvider } from "@/components/ui/ContentArea"
import { useKernelShutdown } from "@/features/notebook/useKernelShutdown"
import { onFileMove } from "@/lib/backend/fileEvents"
import { watchDocumentLists } from "@/lib/backend/resources"
import { AmbientBackdrop } from "@/features/ambient/AmbientBackdrop"
import { AppChrome, type SaveAsRequest } from "./AppChrome"
import { TabContent } from "./TabContent"
import { parseOpenParam, withoutOpenParam } from "./deepLink"

/** Whether `ids` already lists the ids of `tabs` in order, so a tab update that keeps them re-renders nothing. */
function hasTabIds(ids: readonly string[], tabs: readonly Tab[]): boolean {
  return ids.length === tabs.length && tabs.every((tab, index) => tab.id === ids[index])
}

/** Keep project navigation and memory shared while each tab retains its own workspace. */
export function AppShell() {
  const tabManagerRef = useRef<TabManagerHandle>(null)
  const lastProjectTabs = useRef(new Map<string, string>())
  const activeTabRef = useRef<Tab | null>(null)
  const [reveal, setReveal] = useState<ChatReveal>({ historyFile: null, qaIndex: null, seq: 0 })
  // Project whose dashboard is showing; set from the current tab and never read back as "the" project.
  const [dashboardSlug, setDashboardSlug] = useState<string | null>(null)
  // What the shells (one per tab) need: the frontmost tab and which tabs still exist
  const [activeTabId, setActiveTabId] = useState<string | null>(null)
  const [tabIds, setTabIds] = useState<readonly string[]>([])
  const settings = useSettings()
  useKernelShutdown()

  const handleActiveTabChange = useCallback((tab: Tab) => {
    activeTabRef.current = tab
    setActiveTabId(tab.id)
    if (tab.projectSlug) lastProjectTabs.current.set(tab.projectSlug, tab.id)
    setDashboardSlug((current) => current === tab.projectSlug ? current : null)
    // The frontmost Pi tab's session is read while the page shows it
    const sessionId = piSessionOf(tab)
    if (sessionId && document.visibilityState === "visible") markPiSessionRead(sessionId)
  }, [])

  useEffect(() => {
    const tabOf = (sessionId: string) => tabManagerRef.current?.getTabs().find((tab) => piSessionOf(tab) === sessionId)
    return startPiNotifications({
      activeSession: () => activeTabRef.current && piSessionOf(activeTabRef.current),
      titleOf: (sessionId) => {
        const tab = tabOf(sessionId)
        return tab ? tabLabel(tab, undefined, () => undefined) : null
      },
      open: (sessionId) => {
        const tab = tabOf(sessionId)
        if (tab) tabManagerRef.current?.switchToTab(tab.id)
      },
    })
  }, [])

  const openTabIds = useRef<readonly string[]>([])
  const handleTabsChange = useCallback((tabs: Tab[]) => {
    setTabIds((current) => hasTabIds(current, tabs) ? current : tabs.map((tab) => tab.id))
    // A closed tab takes the draft of its unsent new chat with it
    const open = new Set(tabs.map((tab) => tab.id))
    for (const id of openTabIds.current) if (!open.has(id)) forgetComposerDraft(newChatDraftKey(id))
    openTabIds.current = [...open]
  }, [])

  const openNewTab = useCallback((patch: Partial<Tab>) => {
    tabManagerRef.current?.openTab({ ...nextTab(settings.tabDefaults), ...patch })
  }, [settings.tabDefaults])

  const handleNewChat = useCallback((slug: string | null) => {
    openNewTab({ projectSlug: slug })
  }, [openNewTab])

  const handleOpenProject = useCallback((slug: string) => {
    const manager = tabManagerRef.current
    if (!manager) return
    const tabs = manager.getTabs()
    const remembered = tabs.find((tab) => tab.id === lastProjectTabs.current.get(slug) && tab.projectSlug === slug)
    const existing = remembered ?? tabs.findLast((tab) => tab.projectSlug === slug)
    if (existing) manager.switchToTab(existing.id)
    else handleNewChat(slug)
  }, [handleNewChat])

  const handleOpenChat = useCallback((historyFile: string, qaIndex?: number, hit?: ChatRevealHit) => {
    setReveal((current) => ({ historyFile, qaIndex: qaIndex ?? null, hit: hit ?? null, seq: current.seq + 1 }))
    tabManagerRef.current?.showChat(historyFile)
  }, [])

  const handleOpenItem = useCallback((item: WorkspaceItem, inNewTab = false) => {
    if (!inNewTab) {
      tabManagerRef.current?.showItem(item)
      return
    }
    openNewTab(showingItem(item))
  }, [openNewTab])

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

  // A file moved by this device or another one moves the tabs that show it
  useEffect(() => onFileMove(({ from, to }) => tabManagerRef.current?.relocateDocument(from, to)), [])

  // One subscription keeps the sidebar's lists current
  useEffect(() => watchDocumentLists(), [])

  // Click hints (`f`) for the whole app, the shell's clickables only
  useEffect(() => installClickHints(), [])

  useEventListener("document", "keydown", (event) => {
    if (!event.altKey || event.key !== "d") return
    event.preventDefault()
    const slug = activeTabRef.current?.projectSlug ?? null
    setDashboardSlug((current) => current ? null : slug)
  })

  const closeDashboard = useCallback(() => setDashboardSlug(null), [])

  // The save dialog of the app's chrome, opened by the tab in front and shown while it is
  const [saveAs, setSaveAs] = useState<SaveAsRequest | null>(null)
  const handleSaveAs = useCallback((save: SaveAsRequest["save"]) => {
    const tab = activeTabRef.current
    if (tab) setSaveAs({ tabId: tab.id, save })
  }, [])
  const closeSaveAs = useCallback(() => setSaveAs(null), [])

  return (
    <ContentAreaProvider>
      <AmbientBackdrop />
      <BranchProvider onFileClick={handleOpenChat} onHistoriesMoved={handleHistoriesMoved}>
        <SidebarProvider>
          <MemoryViewerProvider>
            <PiSessionsProvider>
              <LearnProvider onOpenItem={handleOpenItem}>
                <EnabledToolsProvider>
                  <TabManager
                    ref={tabManagerRef}
                    onTabsChange={handleTabsChange}
                    onActiveTabChange={handleActiveTabChange}
                    defaultConfig={settings.tabDefaults}
                    chrome={(stack) => (
                      <AppChrome
                        onOpenChat={handleOpenChat}
                        onOpenProject={handleOpenProject}
                        onProjectRenamed={handleProjectRenamed}
                        onNewChat={handleNewChat}
                        onOpenItem={handleOpenItem}
                        reveal={reveal}
                        saveAs={saveAs}
                        onSaveAsDone={closeSaveAs}
                      >
                        {stack}
                      </AppChrome>
                    )}
                  >
                    <TabContent onOpenChat={handleOpenChat} onOpenItem={handleOpenItem} onSaveAs={handleSaveAs} reveal={reveal} />
                  </TabManager>
                </EnabledToolsProvider>
              </LearnProvider>
            </PiSessionsProvider>
            {dashboardSlug && <ProjectDashboard key={dashboardSlug} slug={dashboardSlug} onClose={closeDashboard} />}
            <ProviderUsagePoll />
            <MemoryViewer />
            <ShellTerminals activeTabId={activeTabId} tabIds={tabIds} />
          </MemoryViewerProvider>
        </SidebarProvider>
      </BranchProvider>
    </ContentAreaProvider>
  )
}
