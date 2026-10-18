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

function parseSaved(saved: string | null): Set<string> {
  if (!saved) return new Set()
  try { return new Set(JSON.parse(saved)) } catch { return new Set() }
}

/** Share persisted expansion across mounted tabs without activating their contents. */
export function useFileTree({ storageKey }: { storageKey?: string } = {}): FileBranchController {
  const saved = useSyncExternalStore(
    subscribeExpansion,
    () => storageKey ? readSaved(storageKey) : null,
    () => null,
  )
  const persisted = useMemo(() => parseSaved(saved), [saved])
  const [localExpanded, setLocalExpanded] = useState<Set<string>>(() => new Set())
  const expanded = storageKey && saved !== null ? persisted : localExpanded

  const persist = useCallback((next: Set<string>) => {
    setLocalExpanded(next)
    if (!storageKey) return
    try {
      localStorage.setItem(storageKey, JSON.stringify([...next]))
      window.dispatchEvent(new Event("file-tree-expansion"))
    } catch {}
  }, [storageKey])

  const isExpanded = useCallback((id: string) => expanded.has(id), [expanded])
  const expand = useCallback((id: string) => {
    if (!expanded.has(id)) persist(new Set(expanded).add(id))
  }, [persist, expanded])
  const toggleBranch = useCallback((id: string) => {
    const next = new Set(expanded)
    if (!next.delete(id)) next.add(id)
    persist(next)
  }, [persist, expanded])

  return { isExpanded, toggleBranch, expand }
}
