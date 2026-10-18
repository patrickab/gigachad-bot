"use client"

import { useCallback, useMemo, useState, useSyncExternalStore } from "react"

export interface FileBranchController {
  isExpanded: (id: string) => boolean
  toggleBranch: (id: string) => void
  expand: (id: string) => void
}

function subscribeExpansion(callback: () => void) {
  window.addEventListener("file-tree-expansion", callback)
  window.addEventListener("storage", callback)
  return () => {
    window.removeEventListener("file-tree-expansion", callback)
    window.removeEventListener("storage", callback)
  }
}

function readSaved(storageKey: string): string | null {
  try { return localStorage.getItem(storageKey) } catch { return null }
}

function parseSaved(saved: string | null): Record<string, boolean> {
  if (!saved) return {}
  try {
    const parsed = JSON.parse(saved)
    // Migrate the old expanded-only list without treating absent defaults as collapsed.
    return Array.isArray(parsed) ? Object.fromEntries(parsed.map((id) => [id, true])) : parsed ?? {}
  } catch { return {} }
}

/** Share persisted expansion across mounted tabs without activating their contents. */
export function useFileTree({ storageKey, defaultExpanded = [] }: { storageKey?: string; defaultExpanded?: readonly string[] } = {}): FileBranchController {
  const saved = useSyncExternalStore(
    subscribeExpansion,
    () => storageKey ? readSaved(storageKey) : null,
    () => null,
  )
  const persisted = useMemo(() => parseSaved(saved), [saved])
  const [localExpanded, setLocalExpanded] = useState<Record<string, boolean>>({})
  const expanded = storageKey && saved !== null ? persisted : localExpanded

  const persist = useCallback((next: Record<string, boolean>) => {
    setLocalExpanded(next)
    if (!storageKey) return
    try {
      localStorage.setItem(storageKey, JSON.stringify(next))
      window.dispatchEvent(new Event("file-tree-expansion"))
    } catch {}
  }, [storageKey])

  const isExpanded = useCallback((id: string) => typeof expanded[id] === "boolean" ? expanded[id] : defaultExpanded.includes(id), [expanded, defaultExpanded])
  const expand = useCallback((id: string) => {
    if (!isExpanded(id)) persist({ ...expanded, [id]: true })
  }, [persist, expanded, isExpanded])
  const toggleBranch = useCallback((id: string) => {
    persist({ ...expanded, [id]: !isExpanded(id) })
  }, [persist, expanded, isExpanded])

  return { isExpanded, toggleBranch, expand }
}

/** Unpersisted expansion for path-keyed trees: everything starts collapsed, and collapsing a folder collapses its contents. */
export function usePathTree(): FileBranchController {
  const [expanded, setExpanded] = useState<ReadonlySet<string>>(() => new Set())
  const isExpanded = useCallback((id: string) => expanded.has(id), [expanded])
  const expand = useCallback((id: string) => setExpanded((current) => current.has(id) ? current : new Set(current).add(id)), [])
  const toggleBranch = useCallback((id: string) => setExpanded((current) => current.has(id)
    ? new Set([...current].filter((other) => other !== id && !other.startsWith(`${id}/`)))
    : new Set(current).add(id)), [])
  return { isExpanded, toggleBranch, expand }
}
