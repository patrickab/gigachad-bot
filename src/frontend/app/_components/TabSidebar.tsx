"use client"

// Renders one tab's Sidebar apart from its chat, so streamed tokens never re-render navigation.
// Consumes the tab context, the shared model and prompt stores, settings and stable shell callbacks.
// Provides `TabSidebar`, memoized here rather than on Sidebar: React 19.2 keeps stale `useEffectEvent` closures in a memo component that context re-renders.

import { memo, useCallback } from "react"
import { Sidebar } from "@/features/navigation/Sidebar"
import { SettingsMenu } from "@/features/settings/SettingsMenu"
import { useEnabledTools } from "@/features/chat/EnabledToolsContext"
import { useHostAccess, useNeovimSetting } from "@/contexts/SettingsContext"
import { itemOf, scratchItem, tabSurface, useTab, type WorkspaceItem } from "@/contexts/TabContext"
import { useResource } from "@/hooks/useResource"
import { modelsResource, promptsResource } from "@/lib/backend/resources"

const NO_PROMPTS: Record<string, string> = {}

interface TabSidebarProps {
  onOpenProject: (slug: string) => void
  onProjectRenamed: (oldSlug: string, newSlug: string) => void
  onNewChat: (slug: string | null) => void
  onOpenItem: (item: WorkspaceItem, inNewTab?: boolean) => void
  onEditPrompts: () => void
}

/** One tab's sidebar, fed only by stable callbacks and stores the chat stream never touches. */
export const TabSidebar = memo(function TabSidebar({ onOpenProject, onProjectRenamed, onNewChat, onOpenItem, onEditPrompts }: TabSidebarProps) {
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
    />
  )
})
