"use client"

// Renders the app's one Sidebar for the tab in front, apart from every chat, so streamed tokens never re-render navigation.
// Consumes the tab context (in the app's chrome: the tab in front), the shared model and prompt stores, settings and stable shell callbacks.
// Provides `AppSidebar`, memoized here rather than on Sidebar: the chrome re-renders with every tab manager change (another tab's
// title, a mode label, a peek), the sidebar only when the tab in front does. React 19.2 keeps stale `useEffectEvent` closures in a memo component that context re-renders.

import { memo, useCallback } from "react"
import { Sidebar } from "@/features/navigation/Sidebar"
import { SettingsMenu } from "@/features/settings/SettingsMenu"
import type { ChatReveal } from "@/features/chat/ChatContainer"
import { useEnabledTools } from "@/features/chat/EnabledToolsContext"
import { useHostAccess, useNeovimSetting } from "@/contexts/SettingsContext"
import { itemOf, scratchItem, tabSurface, useTab, type WorkspaceItem } from "@/contexts/TabContext"
import { useResource } from "@/hooks/useResource"
import { modelsResource, promptsResource } from "@/lib/backend/resources"

const NO_PROMPTS: Record<string, string> = {}

interface AppSidebarProps {
  onOpenProject: (slug: string) => void
  onProjectRenamed: (oldSlug: string, newSlug: string) => void
  onNewChat: (slug: string | null) => void
  onOpenItem: (item: WorkspaceItem, inNewTab?: boolean) => void
  onEditPrompts: () => void
  reveal: ChatReveal
}

/** The sidebar of the tab in front, fed only by stable callbacks and stores the chat stream never touches. */
export const AppSidebar = memo(function AppSidebar({ onOpenProject, onProjectRenamed, onNewChat, onOpenItem, onEditPrompts, reveal }: AppSidebarProps) {
  const { tab, updateConfig, setWorkspaceItem, dropItem } = useTab()
  const { config, workspaceItem, itemHidden, projectSlug } = tab
  const { researchEnabled } = useEnabledTools()
  // The workbench needs Neovim for the source and host access for its Pi chats and the LaTeX build
  const neovimEnabled = useNeovimSetting().enabled
  const hostAccess = useHostAccess()
  const models = useResource("Models", modelsResource) ?? null
  const prompts = useResource("Prompts", promptsResource) ?? NO_PROMPTS
  const canvasPath = itemOf(tab, "canvas")?.path ?? null

  const onCanvasDeleted = useCallback((path: string) => {
    if (workspaceItem?.kind !== "canvas" || workspaceItem.path !== path) return
    if (itemHidden) dropItem()
    else setWorkspaceItem(scratchItem(projectSlug))
  }, [workspaceItem, itemHidden, projectSlug, dropItem, setWorkspaceItem])
  const renderSettings = useCallback(() => (
    <SettingsMenu prompts={prompts} config={config} onConfigChange={updateConfig} onEditPrompts={onEditPrompts} models={models} researchEnabled={researchEnabled} />
  ), [prompts, config, updateConfig, onEditPrompts, models, researchEnabled])
  const activeDocumentPath = (itemOf(tab, "document") ?? itemOf(tab, "workbench"))?.path
  // Another chat's reveal marks no question of this one
  const activeQaIndex = reveal.historyFile === tab.historyFile ? reveal.qaIndex : null

  return (
    <Sidebar
      activeCanvasPath={canvasPath}
      onCanvasDeleted={onCanvasDeleted}
      onProjectSelect={onOpenProject}
      onProjectRenamed={onProjectRenamed}
      onNewChat={onNewChat}
      activeDocumentPath={activeDocumentPath}
      workbench={neovimEnabled && hostAccess}
      onItemOpen={onOpenItem}
      renderSettings={renderSettings}
      appMode={tabSurface(tab)}
      activeQaIndex={activeQaIndex}
    />
  )
})
