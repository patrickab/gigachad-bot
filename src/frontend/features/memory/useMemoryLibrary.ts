"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { useUndoDelete } from "@/contexts/UndoDeleteContext"
import { applyMemoryOps, listMemories, readMemory } from "@/lib/backend/api"
import type { Fact, MemoryListing, MemoryOp, MemoryOpResult, MemoryRead } from "@/lib/domain/memory"
import { errorMessage } from "@/lib/utils"
import { newestFirst, ROOT_LOCATION, type MemoryLocation } from "@/features/memory/memoryUtils"

/** One listed level: the root shows a section per top-level memory_store, any other location one section. */
export interface MemorySection extends MemoryLocation {
  listing: MemoryListing
}

/** The memory or document open in the window. */
export interface MemoryPageTarget {
  memoryStore: string
  path: string
}

export type MemoryPage =
  | { kind: "memory"; description: string; facts: Fact[] }
  | { kind: "document"; text: string }

async function loadSections(location: MemoryLocation): Promise<MemorySection[]> {
  if (location.memoryStore) return [{ ...location, listing: await listMemories(location.memoryStore, location.path) }]
  const root = await listMemories()
  return Promise.all(root.memory_stores.map(async (memoryStore) => ({ memoryStore, path: "", listing: await listMemories(memoryStore) })))
}

function toPage(read: MemoryRead): MemoryPage {
  return read.kind === "memory" ? { ...read, facts: newestFirst(read.facts) } : read
}

const sameTarget = (a: MemoryPageTarget | null, b: MemoryPageTarget) => a?.memoryStore === b.memoryStore && a.path === b.path
const retireKey = (at: MemoryPageTarget, id: string) => `memory-fact:${at.memoryStore}:${at.path}:${id}`

/**
 * Browse the memory root and edit the open memory.
 *
 * - Edits show at once and are written in order, one batch at a time; the memory is read again once no write is pending
 * - A retire waits out the undo window, undo brings the fact back; a read meanwhile keeps the fact hidden
 */
export function useMemoryLibrary() {
  const { schedule, pending } = useUndoDelete()
  const [trail, setTrail] = useState<MemoryLocation[]>([ROOT_LOCATION])
  const [sections, setSections] = useState<MemorySection[] | null>(null)
  const [target, setTarget] = useState<MemoryPageTarget | null>(null)
  const [storedPage, setPage] = useState<MemoryPage | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [listVersion, setListVersion] = useState(0)
  const location = trail[trail.length - 1]

  const currentTarget = useRef<MemoryPageTarget | null>(null)
  const writes = useRef<Promise<unknown>>(Promise.resolve())
  const pendingWrites = useRef(0)
  // A shown fact keeps its id while its add or update is written; this maps it to the latest stored id.
  const [storedIds] = useState(() => new Map<string, string>())
  const localIds = useRef(0)

  useEffect(() => {
    let cancelled = false
    loadSections(location)
      .then((loaded) => { if (!cancelled) setSections(loaded) })
      .catch((e) => { if (!cancelled) setError(errorMessage(e, "Failed to load memory")) })
    return () => { cancelled = true }
  }, [location, listVersion])

  const refresh = useCallback(async (at: MemoryPageTarget) => {
    try {
      const read = await readMemory(at.memoryStore, at.path)
      if (sameTarget(currentTarget.current, at)) setPage(toPage(read))
    } catch (e) {
      if (sameTarget(currentTarget.current, at)) setError(errorMessage(e, "Failed to load memory"))
    }
  }, [])

  const openLocation = useCallback((next: MemoryLocation) => {
    setSections(null)
    setTrail((current) => [...current, next])
  }, [])

  const openPage = useCallback((memoryStore: string, path: string) => {
    const at = { memoryStore, path }
    currentTarget.current = at
    setPage(null)
    setTarget(at)
    void refresh(at)
  }, [refresh])

  /** Leave the open page for its listing, else go up one location. */
  const back = useCallback(() => {
    if (target) {
      currentTarget.current = null
      setTarget(null)
      setPage(null)
      setListVersion((version) => version + 1)
    } else if (trail.length > 1) {
      setSections(null)
      setTrail((current) => current.slice(0, -1))
    }
  }, [target, trail.length])

  /** Queue a batch; `ops` is built when the batch runs, so local ids resolve to the ids stored meanwhile. Rejects when an op fails. */
  const write = useCallback((at: MemoryPageTarget, ops: () => MemoryOp[], onResults?: (results: MemoryOpResult[]) => void): Promise<void> => {
    pendingWrites.current += 1
    const run = writes.current.then(async () => {
      const results = await applyMemoryOps(at.memoryStore, ops())
      onResults?.(results)
      const failed = results.find((result) => !result.ok)
      if (failed && !failed.ok) throw new Error(failed.message)
    }).finally(() => {
      pendingWrites.current -= 1
      if (pendingWrites.current === 0) void refresh(at)
    })
    writes.current = run.catch(() => {})
    return run
  }, [refresh])

  const reportError = useCallback((e: unknown) => setError(`Could not save memory: ${errorMessage(e, "unknown error")}`), [])
  const storedId = (id: string) => storedIds.get(id) ?? id

  const patchMemory = (change: (page: MemoryPage & { kind: "memory" }) => MemoryPage) =>
    setPage((current) => current?.kind === "memory" ? change(current) : current)

  const add = (text: string) => {
    const at = target
    if (!at) return
    const now = new Date().toISOString()
    localIds.current += 1
    const localId = `local-${localIds.current}`
    patchMemory((memory) => ({ ...memory, facts: [{ id: localId, text, created: now, updated: now }, ...memory.facts] }))
    write(at, () => [{ op: "add", path: at.path, text }], ([result]) => {
      if (result?.ok && result.id) storedIds.set(localId, result.id)
    }).catch(reportError)
  }

  const update = (id: string, text: string) => {
    const at = target
    if (!at) return
    const now = new Date().toISOString()
    patchMemory((memory) => ({ ...memory, facts: newestFirst(memory.facts.map((fact) => fact.id === id ? { ...fact, text, updated: now } : fact)) }))
    write(at, () => [{ op: "update", path: at.path, id: storedId(id), text }], ([result]) => {
      if (result?.ok && result.id) storedIds.set(id, result.id)
    }).catch(reportError)
  }

  const retire = (id: string) => {
    const at = target
    if (!at) return
    patchMemory((memory) => ({ ...memory, facts: memory.facts.filter((fact) => fact.id !== id) }))
    schedule(retireKey(at, id), "fact",
      () => write(at, () => [{ op: "retire", path: at.path, id: storedId(id) }]),
      () => void refresh(at))
  }

  const describe = (description: string) => {
    const at = target
    if (!at) return
    patchMemory((memory) => ({ ...memory, description }))
    write(at, () => [{ op: "describe", path: at.path, description }]).catch(reportError)
  }

  // A read during a retire's undo window (e.g. after another edit) still has the fact; it stays hidden until undone.
  const shownIds = (id: string) => [id, ...[...storedIds].filter(([, stored]) => stored === id).map(([shown]) => shown)]
  const hidden = (fact: Fact) => target !== null && shownIds(fact.id).some((id) => pending(retireKey(target, id)))
  const page = storedPage?.kind === "memory" && storedPage.facts.some(hidden)
    ? { ...storedPage, facts: storedPage.facts.filter((fact) => !hidden(fact)) }
    : storedPage

  return { location, trail, sections, target, page, error, setError, openLocation, openPage, back, add, update, retire, describe }
}
