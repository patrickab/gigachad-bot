"use client"

import { useCallback, useEffect, useState, type Dispatch, type SetStateAction } from "react"
import { useTab } from "@/contexts/TabContext"
import { countOperations, useMemorizeFlow, type MemorizeSource, type MemoryPanelState, type MemoryReviewActions } from "@/features/memory/useMemorizeFlow"

export interface MemoryCommandBar {
  /** Neither the menu nor a memory panel is showing, so either may open. */
  isIdle: boolean
  menuOpen: boolean
  openMenu: () => void
  closeMenu: () => void
  query: string
  setQuery: Dispatch<SetStateAction<string>>
  /** Extract global and (in a project) project memories from the tab's conversation, nothing without one. Starting one closes the menu. */
  memorize: () => Promise<void>
  /** Null while no memorize flow is running. */
  panelState: MemoryPanelState | null
  panelActions: MemoryReviewActions
}

/** Own the workspace command menu and the memory flow it launches. */
export function useMemoryCommandBar({ loadSource }: {
  /** Loads the conversation `/memorize` reads, null when the tab has none to memorize (a branch chat, no omp session). */
  loadSource: (() => Promise<MemorizeSource>) | null
}): MemoryCommandBar {
  const { tab, setModeLabel } = useTab()
  const projectSlug = tab.projectSlug
  const flow = useMemorizeFlow()
  const [menuOpen, setMenuOpen] = useState(false)
  const [query, setQuery] = useState("")

  const phase = flow.state.phase
  const changeCount = flow.state.phase === "review" ? countOperations(flow.state.scopes) : 0

  const openMenu = useCallback(() => {
    setQuery("")
    setMenuOpen(true)
  }, [])
  const closeMenu = useCallback(() => setMenuOpen(false), [])

  useEffect(() => {
    if (tab.appMode !== "chat") return
    if (phase === "extracting") setModeLabel("Memory Management", true)
    else if (phase === "review") setModeLabel(`${changeCount} memory ${changeCount === 1 ? "change" : "changes"} proposed`)
    else if (phase === "error") setModeLabel("Memory error")
    else setModeLabel(tab.title ?? "Chat")
  }, [phase, changeCount, tab.title, tab.appMode, setModeLabel])

  const { extract } = flow

  const memorize = useCallback(async () => {
    if (!loadSource) return
    setMenuOpen(false)
    await extract(loadSource, projectSlug)
  }, [loadSource, extract, projectSlug])

  return {
    isIdle: !menuOpen && phase === "idle",
    menuOpen,
    openMenu,
    closeMenu,
    query,
    setQuery,
    memorize,
    panelState: flow.state.phase === "idle" ? null : flow.state,
    panelActions: flow.actions,
  }
}
