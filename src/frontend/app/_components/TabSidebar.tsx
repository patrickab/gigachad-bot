"use client"

// Renders one tab's Sidebar apart from its chat, so streamed tokens never re-render navigation.
// Consumes the tab context, the shared model and prompt stores, settings and stable shell callbacks.
// Provides `TabSidebar`, memoized here rather than on Sidebar: React 19.2 keeps stale `useEffectEvent` closures in a memo component that context re-renders.

import { memo, useCallback, useMemo } from "react"
import { Sidebar } from "@/features/navigation/Sidebar"
import { SettingsMenu } from "@/features/settings/SettingsMenu"
import { useEnabledTools } from "@/features/chat/EnabledToolsContext"
import { useCanvasView } from "@/features/editor/canvas/useCanvasView"
import { useHostAccess, useNeovimSetting } from "@/contexts/SettingsContext"
import { itemOf, useTab, type WorkspaceItem } from "@/contexts/TabContext"
import { useResource } from "@/hooks/useResource"
import { modelsResource, promptsResource } from "@/lib/backend/resources"
import { opensInWorkbench } from "@/lib/domain/workbench"

const NO_PROMPTS: Record<string, string> = {}

interface TabSidebarProps {
  onOpenProject: (slug: string) => void
  onProjectRenamed: (oldSlug: string, newSlug: string) => void
  onDocumentRenamed: (oldPath: string, newPath: string) => void
  onNewChat: (slug: string | null) => void
  onOpenItem: (item: WorkspaceItem, inNewTab?: boolean) => void
  onEditPrompts: () => void
}

/** One tab's sidebar, fed only by stable callbacks and stores the chat stream never touches. */
export const TabSidebar = memo(function TabSidebar({ onOpenProject, onProjectRenamed, onDocumentRenamed, onNewChat, onOpenItem, onEditPrompts }: TabSidebarProps) {
  const { tab, updateConfig } = useTab()
  const { config, appMode } = tab
  const { researchEnabled } = useEnabledTools()
  // The workbench needs Neovim for the source and host access for omp and the LaTeX build
  const neovimEnabled = useNeovimSetting().enabled
  const hostAccess = useHostAccess()
  const opensWorkbench = neovimEnabled && hostAccess
  const models = useResource("Models", modelsResource) ?? null
  const prompts = useResource("Prompts", promptsResource) ?? NO_PROMPTS
  const { closeCanvas, view } = useCanvasView()
  const canvasPath = view.kind === "canvas" ? view.selection?.path ?? null : null

  const onFileSelect = useCallback((path: string, scope: string | null) => {
    onOpenItem({ kind: opensWorkbench && opensInWorkbench(path) ? "workbench" : "document", path, projectSlug: scope, rootFile: true })
  }, [onOpenItem, opensWorkbench])
  const onDiffSelect = useCallback((path: string) => onOpenItem({ kind: "diff", path, projectSlug: null }), [onOpenItem])
  const onCanvasDeleted = useCallback((path: string) => {
    if (canvasPath === path) closeCanvas()
  }, [canvasPath, closeCanvas])
  const onDocumentSelect = useCallback((path: string, scope: string) => {
    onOpenItem({ kind: "document", path, projectSlug: scope || null })
  }, [onOpenItem])
  const onWorkbenchSelect = useMemo(() => opensWorkbench
    ? (path: string, scope: string) => onOpenItem({ kind: opensInWorkbench(path) ? "workbench" : "document", path, projectSlug: scope || null })
    : undefined, [onOpenItem, opensWorkbench])
  const onCanvasSelect = useCallback((path: string, scope: string) => {
    onOpenItem({ kind: "canvas", path, projectSlug: scope || null })
  }, [onOpenItem])
  const onArchitectureSelect = useCallback((slug: string) => {
    onOpenItem({ kind: "architecture", path: `graph/${slug}/`, projectSlug: slug })
  }, [onOpenItem])
  const renderSettings = useCallback(() => (
    <SettingsMenu prompts={prompts} config={config} onConfigChange={updateConfig} onEditPrompts={onEditPrompts} models={models} researchEnabled={researchEnabled} />
  ), [prompts, config, updateConfig, onEditPrompts, models, researchEnabled])
  const activeDocumentPath = appMode === "document" ? itemOf(tab, "document")?.path : appMode === "workbench" ? itemOf(tab, "workbench")?.path : null

  return (
    <Sidebar
      onFileSelect={onFileSelect}
      onDiffSelect={onDiffSelect}
      activeCanvasPath={canvasPath}
      onCanvasDeleted={onCanvasDeleted}
      onProjectSelect={onOpenProject}
      onProjectRenamed={onProjectRenamed}
      onDocumentRenamed={onDocumentRenamed}
      onNewChat={onNewChat}
      onDocumentSelect={onDocumentSelect}
      onWorkbenchSelect={onWorkbenchSelect}
      activeDocumentPath={activeDocumentPath}
      onCanvasSelect={onCanvasSelect}
      onArchitectureSelect={onArchitectureSelect}
      onItemOpen={onOpenItem}
      renderSettings={renderSettings}
      appMode={appMode}
    />
  )
})
