"use client"

import { useCallback, useMemo, useState, useSyncExternalStore } from "react"

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

function subscribeNever() {
  return () => {}
}

function readSaved(storageKey: string): string | null {
  try { return localStorage.getItem(storageKey) } catch { return null }
}

function parseSaved(saved: string | null): Set<string> {
  if (!saved) return new Set()
  try { return new Set(JSON.parse(saved)) } catch { return new Set() }
}

export function useFileTree(config: FileTreeConfig = {}): FileBranchController {
  const { accordion, activeId, storageKey, onActivate, onDeactivate } = config
  // Persisted set is read on the client only, so server and hydration render start empty.
  const saved = useSyncExternalStore(
    subscribeNever,
    () => (accordion || !storageKey ? null : readSaved(storageKey)),
    () => null,
  )
  const persisted = useMemo(() => parseSaved(saved), [saved])
  // User toggles override the persisted set for the storage key they were made under.
  const [edit, setEdit] = useState<{ storageKey: string | undefined; set: Set<string> } | null>(null)
  const expanded = edit && edit.storageKey === storageKey ? edit.set : persisted

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
    setEdit((prev) => {
      const current = prev && prev.storageKey === storageKey ? prev.set : persisted
      if (current.has(id)) return prev
      const next = new Set(current).add(id)
      persist(next)
      return { storageKey, set: next }
    })
  }, [accordion, activeId, onActivate, persist, persisted, storageKey])

  const toggleBranch = useCallback((id: string) => {
    if (accordion) {
      if (activeId === id) onDeactivate?.(id)
      else onActivate?.(id)
      return
    }
    setEdit((prev) => {
      const next = new Set(prev && prev.storageKey === storageKey ? prev.set : persisted)
      if (!next.delete(id)) next.add(id)
      persist(next)
      return { storageKey, set: next }
    })
  }, [accordion, activeId, onActivate, onDeactivate, persist, persisted, storageKey])

  return { isExpanded, toggleBranch, expand }
}
