"use client"

import { useCallback, useEffect, useState, type Dispatch, type SetStateAction } from "react"
import { useTab } from "@/contexts/TabContext"
import { useMemorizeFlow, type BoundMemoryActions, type MemoryPanelState } from "@/features/memory/useMemorizeFlow"
import { useMemoryCategories } from "@/features/memory/useMemoryCategories"
import type { CategoryDef, Message } from "@/lib/domain/types"

/** Which memory stores an extraction targets: `null` is both global and project. */
export type MemorizeScope = "global" | "project" | null

export interface MemoryCommandBar {
  /** Neither the menu nor a memory panel is showing, so either may open. */
  isIdle: boolean
  menuOpen: boolean
  openMenu: () => void
  closeMenu: () => void
  query: string
  setQuery: Dispatch<SetStateAction<string>>
  /** Extract memories from the main branch; a branch chat does nothing. Starting one closes the menu. */
  memorize: (scope: MemorizeScope) => Promise<void>
  /** Null while no memorize flow is running. */
  panelState: MemoryPanelState | null
  panelActions: BoundMemoryActions
  categories: { globalCategories: CategoryDef[]; projectCategories: CategoryDef[] }
}

/** Own the workspace command menu and the memory flow it launches. */
export function useMemoryCommandBar({ messages, chatId, branchMessageIdx }: {
  messages: Message[]
  chatId: string
  /** Index where this chat branched off its parent, null for a main chat. */
  branchMessageIdx: number | null
}): MemoryCommandBar {
  const { tab, setModeLabel } = useTab()
  const activeProject = tab.projectSlug
  const flow = useMemorizeFlow()
  const categories = useMemoryCategories(activeProject)
  const [menuOpen, setMenuOpen] = useState(false)
  const [query, setQuery] = useState("")

  const phase = flow.state.phase
  const memoryCount = flow.state.phase === "review"
    ? flow.state.globalMemories.length + (flow.state.projectMemories?.length ?? 0)
    : 0
  const docReviewLoading = flow.state.phase === "doc-review" && flow.state.loadingScopes.length > 0

  const openMenu = useCallback(() => {
    setQuery("")
    setMenuOpen(true)
  }, [])
  const closeMenu = useCallback(() => setMenuOpen(false), [])

  useEffect(() => {
    if (tab.appMode !== "chat") return
    if (phase === "extracting") setModeLabel("Memory Management", true)
    else if (phase === "review") setModeLabel(`${memoryCount} ${memoryCount === 1 ? "memory" : "memories"} proposed`)
    else if (phase === "doc-review") setModeLabel("Review memory docs", docReviewLoading)
    else if (phase === "error") setModeLabel("Memory error")
    else setModeLabel(tab.title ?? "Chat")
  }, [phase, docReviewLoading, memoryCount, tab.title, tab.appMode, setModeLabel])

  const { extract } = flow

  const memorize = useCallback(async (scope: MemorizeScope) => {
    if (branchMessageIdx != null) return
    setMenuOpen(false)
    await extract(messages, activeProject, chatId, scope)
  }, [branchMessageIdx, extract, messages, activeProject, chatId])

  return {
    isIdle: !menuOpen && phase === "idle",
    menuOpen,
    openMenu,
    closeMenu,
    query,
    setQuery,
    memorize,
    panelState: flow.state.phase === "idle" ? null : flow.state,
    panelActions: flow.bindProject(activeProject),
    categories,
  }
}
