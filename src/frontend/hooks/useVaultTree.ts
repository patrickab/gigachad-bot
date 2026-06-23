"use client"

import { useCallback, useEffect, useState } from "react"

/**
 * The selection primitive shared by every VaultTree (Projects, Histories, …).
 *
 * It owns one concern: which *branches* (vaults / folders) are expanded, and what
 * expanding/collapsing one means. Element (chat) selection is a separate primitive
 * already owned by BranchContext (activeFile), so it lives outside this hook.
 *
 * Two modes, one interface:
 *   • accordion — expansion IS the externally-active branch (`activeId`). One open
 *     at a time, no internal state, so it can never drift from the owning context
 *     (e.g. ProjectContext.activeProject). Toggling fires onActivate/onDeactivate.
 *   • multi     — an internal, optionally-persisted Set. Folders behave like a file
 *     browser: many open at once, no activation side effects.
 *
 * Each VaultTree gets its own controller, so collapsing a project never touches a
 * histories folder — cross-tree independence falls out for free.
 */
export interface VaultBranchController {
  isExpanded: (id: string) => boolean
  toggleBranch: (id: string) => void
  expand: (id: string) => void
}

export interface VaultTreeConfig {
  /** One branch open at a time, sourced from `activeId` instead of internal state. */
  accordion?: boolean
  /** externally-active branch (e.g. activeProject); the source of truth in accordion mode. */
  activeId?: string | null
  /** localStorage key for persisting the open set (multi mode only). */
  storageKey?: string
  /** a branch was opened — e.g. load its project context. */
  onActivate?: (id: string) => void
  /** a branch was closed — e.g. close its project + chat. */
  onDeactivate?: (id: string) => void
}

export function useVaultTree(config: VaultTreeConfig = {}): VaultBranchController {
  const { accordion, activeId, storageKey, onActivate, onDeactivate } = config
  const [expanded, setExpanded] = useState<Set<string>>(new Set())

  // multi-mode persistence load
  useEffect(() => {
    if (accordion || !storageKey) return
    try {
      const saved = localStorage.getItem(storageKey)
      if (saved) setExpanded(new Set(JSON.parse(saved)))
    } catch {}
  }, [accordion, storageKey])

  const persist = useCallback((next: Set<string>) => {
    if (accordion || !storageKey) return
    try { localStorage.setItem(storageKey, JSON.stringify([...next])) } catch {}
  }, [accordion, storageKey])

  const isExpanded = useCallback(
    (id: string) => (accordion ? activeId === id : expanded.has(id)),
    [accordion, activeId, expanded],
  )

  const expand = useCallback((id: string) => {
    if (accordion) {
      if (activeId !== id) onActivate?.(id)
      return
    }
    setExpanded((prev) => {
      if (prev.has(id)) return prev
      const next = new Set(prev).add(id)
      persist(next)
      return next
    })
  }, [accordion, activeId, onActivate, persist])

  const toggleBranch = useCallback((id: string) => {
    if (accordion) {
      activeId === id ? onDeactivate?.(id) : onActivate?.(id)
      return
    }
    setExpanded((prev) => {
      const next = new Set(prev)
      next.has(id) ? next.delete(id) : next.add(id)
      persist(next)
      return next
    })
  }, [accordion, activeId, onActivate, onDeactivate, persist])

  return { isExpanded, toggleBranch, expand }
}
