"use client"

import { useCallback, useEffect, useRef, useState, type Dispatch, type RefObject, type SetStateAction } from "react"
import { useProject } from "@/contexts/ProjectContext"
import { useTab } from "@/contexts/TabContext"
import { useMemorizeFlow, type BoundMemoryActions, type MemoryPanelState } from "@/hooks/useMemorizeFlow"
import { useMemoryCategories } from "@/hooks/useMemoryCategories"
import type { CategoryDef, Message } from "@/lib/types"

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
  inputRef: RefObject<HTMLInputElement | null>
  /** Tab-row slots the menu input and list portal into. */
  labelSlot: HTMLElement | null
  setLabelSlot: Dispatch<SetStateAction<HTMLElement | null>>
  menuSlot: HTMLElement | null
  setMenuSlot: Dispatch<SetStateAction<HTMLElement | null>>
  /** Extract memories from the main branch; a branch chat does nothing. Starting one closes the menu. */
  memorize: (scope: MemorizeScope) => Promise<void>
  /** Null while no memorize flow is running. */
  panelState: MemoryPanelState | null
  panelActions: BoundMemoryActions
  categories: { globalCategories: CategoryDef[]; projectCategories: CategoryDef[] }
}

/** Slash-command menu in the tab row plus the memorize flow it launches, which share the tab's label slot. */
export function useMemoryCommandBar({ messages, chatId, branchMessageIdx }: {
  messages: Message[]
  chatId: string
  /** Index where this chat branched off its parent, null for a main chat. */
  branchMessageIdx: number | null
}): MemoryCommandBar {
  const { tab, setModeLabel } = useTab()
  const { activeProject } = useProject()
  const flow = useMemorizeFlow()
  const categories = useMemoryCategories(activeProject)
  const [menuOpen, setMenuOpen] = useState(false)
  const [query, setQuery] = useState("")
  const [labelSlot, setLabelSlot] = useState<HTMLElement | null>(null)
  const [menuSlot, setMenuSlot] = useState<HTMLElement | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const phase = flow.state.phase
  const memoryCount = flow.state.phase === "review"
    ? flow.state.globalMemories.length + (flow.state.projectMemories?.length ?? 0)
    : 0
  const docReviewLoading = flow.state.phase === "doc-review" && flow.state.loadingScopes.length > 0

  useEffect(() => {
    if (!menuOpen) return
    setQuery("")
    // Defer so the portaled input exists before focusing.
    setTimeout(() => inputRef.current?.focus(), 0)
  }, [menuOpen])

  useEffect(() => {
    // "\0" is a sentinel that hides the tab label so the command input can take its slot.
    if (menuOpen) setModeLabel("\0")
    else if (phase === "extracting") setModeLabel("Memory Management", true)
    else if (phase === "review") setModeLabel(`${memoryCount} ${memoryCount === 1 ? "memory" : "memories"} proposed`)
    else if (phase === "doc-review") setModeLabel("Review memory docs", docReviewLoading)
    else if (phase === "error") setModeLabel("Memory error")
    else setModeLabel(tab.title ?? "Chat")
  }, [menuOpen, phase, docReviewLoading, memoryCount, tab.title, setModeLabel])

  const openMenu = useCallback(() => setMenuOpen(true), [])
  const closeMenu = useCallback(() => setMenuOpen(false), [])

  const memorize = useCallback(async (scope: MemorizeScope) => {
    if (branchMessageIdx != null) return
    setMenuOpen(false)
    await flow.extract(messages, activeProject, chatId, scope)
  }, [branchMessageIdx, flow.extract, messages, activeProject, chatId])

  return {
    isIdle: !menuOpen && phase === "idle",
    menuOpen,
    openMenu,
    closeMenu,
    query,
    setQuery,
    inputRef,
    labelSlot,
    setLabelSlot,
    menuSlot,
    setMenuSlot,
    memorize,
    panelState: flow.state.phase === "idle" ? null : flow.state,
    panelActions: flow.bindProject(activeProject),
    categories,
  }
}
