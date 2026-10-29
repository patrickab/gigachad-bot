"use client"

import { useContext } from "react"
import { Plus, X } from "lucide-react"
import { FileTree, type FileTreeItem } from "@/components/ui/file-tree/FileTree"
import { usePathTree } from "@/components/ui/file-tree/useFileTree"
import { useStoredString } from "@/hooks/useStoredString"
import { PATH_COLLATOR } from "@/lib/paths"
import { useDisplayName, useProject } from "@/contexts/ProjectContext"
import { TabStripContext, tabLocation } from "./tabs"

function Spinner() {
  return <span className="h-2.5 w-2.5 shrink-0 animate-spin rounded-full border border-ink-faint border-t-ink-muted" />
}

/** Collapsible sidebar section listing every open tab as a tree element, sorted by project, category and file.
 *
 * - Collapsed by default, expansion is stored so every tab's sidebar shares it
 */
export function TabList() {
  const strip = useContext(TabStripContext)
  const { projects } = useProject()
  const shownName = useDisplayName()
  const controller = usePathTree()
  const [storedExpanded, saveExpanded] = useStoredString("sidebar_tabs_expanded")
  const expanded = storedExpanded === "true"
  if (!strip) return null

  const projectName = (slug: string | null) => projects.find((project) => project.slug === slug)?.name ?? slug ?? "general"
  const items: FileTreeItem[] = strip.tabs.map((item) => ({ item, project: projectName(item.projectSlug), ...tabLocation(item, shownName) }))
    .sort((left, right) => Number(left.project === "general") - Number(right.project === "general")
      || PATH_COLLATOR.compare(left.project, right.project)
      || PATH_COLLATOR.compare(left.category, right.category)
      || PATH_COLLATOR.compare(left.file, right.file)
      || left.item.id.localeCompare(right.item.id))
    .map(({ item, project, category, file }) => ({
      id: item.id,
      type: "element",
      prefix: `${project} / ${category} /`,
      label: file,
      isActive: item.id === strip.activeTabId,
      description: strip.labelOf(item),
      meta: strip.isLoading(item) ? <Spinner /> : undefined,
      onSelect: () => strip.select(item.id),
      actions: strip.tabs.length > 1 ? [{ label: `Close ${project} / ${category} / ${file}`, icon: X, onClick: () => strip.close(item.id) }] : undefined,
    }))

  return <FileTree label="Tabs" items={items} controller={controller}
    labelAction={{ label: `Tabs, ${strip.tabs.length} open`, expanded, onClick: () => saveExpanded(String(!expanded)) }}
    labelSuffix={<>
      <span className="text-[10px] tabular-nums text-ink-faint">{strip.tabs.length}</span>
      <button type="button" data-new-chat aria-label="New chat" onClick={strip.add}
        className="rounded p-0.5 text-ink-faint hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted">
        <Plus aria-hidden="true" className="h-3 w-3" />
      </button>
    </>}
  />
}
