"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { useProject } from "@/contexts/ProjectContext"
import { useUndoDelete } from "@/contexts/UndoDeleteContext"
import { getMemories, getTopics, saveMemories, saveTopics } from "@/lib/backend/api"
import { errorMessage } from "@/lib/utils"
import { formatTopicHeading, normalizeMemoryTopic } from "@/features/memory/memoryUtils"
import { deleteMemory, deleteTopic, restoreMemory, restoreTopic, type ScopeChange, type ScopeMemory } from "@/features/memory/scopeEdits"

/** Load every scope once, apply changes locally and write each scope in order. */
export function useMemoryLibrary() {
  const { projects } = useProject()
  const { schedule } = useUndoDelete()
  const [scopes, setScopes] = useState<ScopeMemory[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const latest = useRef<ScopeMemory[]>([])
  const writes = useRef(new Map<string, Promise<void>>())

  // Reload only when the set of visible projects changes, not on every catalog refresh.
  const projectList = projects.filter((p) => !p.archived).map((p) => `${p.slug}\n${p.name}`).join("\n\n")
  const targets = useMemo(() => [
    { key: "global", scope: "global" as const, projectSlug: null, label: "You" },
    ...(projectList ? projectList.split("\n\n") : []).map((entry) => {
      const [slug, name] = entry.split("\n")
      return { key: `project:${slug}`, scope: "project" as const, projectSlug: slug, label: name }
    }),
  ], [projectList])

  useEffect(() => {
    let cancelled = false
    Promise.all(targets.map(async (target) => {
      const [topicsResponse, memoriesResponse] = await Promise.all([
        getTopics(target.scope, target.projectSlug),
        getMemories(target.scope, target.projectSlug),
      ])
      return { ...target, topics: topicsResponse.topics, memories: memoriesResponse.memories }
    }))
      .then((loaded) => {
        if (cancelled) return
        latest.current = loaded
        setScopes(loaded)
      })
      .catch((e) => { if (!cancelled) setError(errorMessage(e, "Failed to load memory")) })
    return () => { cancelled = true }
  }, [targets])

  const change = useCallback((key: string, apply: ScopeChange) => {
    latest.current = latest.current.map((s) => s.key === key ? { ...s, ...apply(s) } : s)
    setScopes(latest.current)
  }, [])

  // Writes queue per scope and read the newest state when they run, so rapid edits never land out of order.
  const save = useCallback((key: string): Promise<void> => {
    const previous = writes.current.get(key) ?? Promise.resolve()
    const write = previous.catch(() => {}).then(async () => {
      const scope = latest.current.find((s) => s.key === key)
      if (!scope) return
      await saveTopics(scope.scope, scope.topics, scope.projectSlug)
      await saveMemories(scope.scope, scope.memories, scope.projectSlug)
    })
    writes.current.set(key, write)
    return write
  }, [])

  const reportError = useCallback((e: unknown) => {
    setError(`Could not save memory: ${errorMessage(e, "unknown error")}`)
  }, [])

  const update = useCallback((key: string, apply: ScopeChange) => {
    change(key, apply)
    save(key).catch(reportError)
  }, [change, save, reportError])

  const scopeOf = useCallback((key: string) => scopes?.find((s) => s.key === key), [scopes])

  /** Remove content optimistically; the delete is written when the undo window ends, `restore` puts it back and writes again. */
  const removeWithUndo = useCallback((scopeKey: string, undoKey: string, label: string, remove: ScopeChange, restore: ScopeChange) => {
    change(scopeKey, remove)
    schedule(undoKey, label, () => save(scopeKey), () => {
      change(scopeKey, restore)
      save(scopeKey).catch(reportError)
    })
  }, [change, save, schedule, reportError])

  const removeTopic = useCallback((scopeKey: string, name: string) => {
    const scope = scopeOf(scopeKey)
    if (!scope) return
    const index = scope.topics.findIndex((t) => t.name === name)
    const memories = scope.memories.filter((m) => normalizeMemoryTopic(m) === name)
    removeWithUndo(scopeKey, `memory-topic:${scopeKey}:${name}`, formatTopicHeading(name), deleteTopic(name), restoreTopic(index, scope.topics[index], memories))
  }, [scopeOf, removeWithUndo])

  const removeMemory = useCallback((scopeKey: string, memoryId: string) => {
    const scope = scopeOf(scopeKey)
    const index = scope?.memories.findIndex((m) => m.id === memoryId) ?? -1
    if (!scope || index < 0) return
    removeWithUndo(scopeKey, `memory:${memoryId}`, "memory", deleteMemory(memoryId), restoreMemory(index, scope.memories[index]))
  }, [scopeOf, removeWithUndo])

  return { scopes, error, setError, scopeOf, update, removeTopic, removeMemory }
}
