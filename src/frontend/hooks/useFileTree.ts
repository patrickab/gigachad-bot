"use client"

import { useCallback, useEffect, useState } from "react"

/**
 * Expansion state shared by every FileTree. Element (chat) selection lives in BranchContext.
 *
 * - accordion: expansion is the externally-active branch (`activeId`). One open at a
 *   time, no internal state, so it can't drift from the owning context. Toggling fires
 *   onActivate/onDeactivate.
 * - multi: internal, optionally persisted Set. Many open at once, no activation side effects.
 *
 * Each FileTree gets its own controller, so trees never affect each other.
 */
export interface FileBranchController {
  isExpanded: (id: string) => boolean
  toggleBranch: (id: string) => void
  expand: (id: string) => void
}

interface FileTreeConfig {
  /** One branch open at a time, sourced from `activeId` instead of internal state. */
  accordion?: boolean
  /** externally-active branch (e.g. activeProject); the source of truth in accordion mode. */
  activeId?: string | null
  /** localStorage key for persisting the open set (multi mode only). */
  storageKey?: string
  /** a branch was opened, e.g. load its project context. */
  onActivate?: (id: string) => void
  /** a branch was closed, e.g. close its project + chat. */
  onDeactivate?: (id: string) => void
}

export function useFileTree(config: FileTreeConfig = {}): FileBranchController {
  const { accordion, activeId, storageKey, onActivate, onDeactivate } = config
  const [expanded, setExpanded] = useState<Set<string>>(new Set())
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
