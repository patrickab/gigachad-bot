"use client"

import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from "react"
import { ArrowLeft, ChevronDown, ChevronRight, Loader2, Plus, Trash2, X } from "lucide-react"
import { FloatingWindow } from "@/components/ui/FloatingWindow"
import { useProject } from "@/contexts/ProjectContext"
import { useUndoDelete } from "@/contexts/UndoDeleteContext"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import { commitMemoryDoc, getMemories, getTopics, saveTopics } from "@/lib/backend/api"
import { cn } from "@/lib/utils"
import { formatTopicHeading, groupByTopicOrder, newManualMemoryId, normalizeMemoryTopic, topicNameFromTitle } from "@/features/memory/memoryUtils"
import { useMemoryViewer } from "@/features/memory/MemoryViewerContext"
import type { MemoryTopic, PreviewMemory } from "@/lib/domain/types"

/** One memory scope: the global profile ("You") or one project. */
interface ScopeMemory {
  key: string
  scope: "global" | "project"
  projectSlug: string | null
  label: string
  topics: MemoryTopic[]
  memories: PreviewMemory[]
}

type ScopeContent = Pick<ScopeMemory, "topics" | "memories">
type ScopeChange = (scope: ScopeMemory) => ScopeContent

interface TopicRow {
  topic: MemoryTopic
  memories: PreviewMemory[]
  updatedAt: string | null
}

const DAY_MS = 86_400_000
/** Gap between the window and the chat column's edges, matching the overlay's default padding. */
const COLUMN_GAP_PX = 24

/** Overlay padding that centers the window over the visible chat column instead of the whole screen. */
function useChatColumnPadding(): CSSProperties | undefined {
  const [padding, setPadding] = useState<CSSProperties>()
  useEffect(() => {
    const column = Array.from(document.querySelectorAll("main")).find((el) => el.getClientRects().length > 0)
    if (!column) return
    const measure = () => {
      const rect = column.getBoundingClientRect()
      setPadding({ paddingLeft: rect.left + COLUMN_GAP_PX, paddingRight: window.innerWidth - rect.right + COLUMN_GAP_PX })
    }
    // The observer reports once on observe, then on every column resize (sidebar toggle, window resize).
    const observer = new ResizeObserver(measure)
    observer.observe(column)
    return () => observer.disconnect()
  }, [])
  return padding
}

export function MemoryViewer() {
  const { isOpen, closeMemoryViewer } = useMemoryViewer()
  return isOpen ? <MemoryWindow onClose={closeMemoryViewer} /> : null
}

function latestUpdate(memories: PreviewMemory[]): string | null {
  return memories.reduce<string | null>((latest, m) => {
    const stamp = m.updated_at ?? m.created_at ?? null
    return stamp && (!latest || stamp > latest) ? stamp : latest
  }, null)
}

/** Topics in stored order plus topics only referenced by memories, each with its memories. */
function topicRows(scope: ScopeMemory): TopicRow[] {
  return groupByTopicOrder(scope.topics, scope.memories).map(({ topic, items }) => {
    const memories = items as PreviewMemory[]
    return { topic, memories, updatedAt: latestUpdate(memories) }
  })
}

function formatUpdated(iso: string | null): string {
  if (!iso) return "Empty"
  const date = new Date(iso)
  const now = new Date()
  const days = Math.round((new Date(now.toDateString()).getTime() - new Date(date.toDateString()).getTime()) / DAY_MS)
  if (days <= 0) return "Updated today"
  if (days === 1) return "Updated yesterday"
  if (days < 7) return `Updated ${days} days ago`
  const options: Intl.DateTimeFormatOptions = date.getFullYear() === now.getFullYear()
    ? { month: "short", day: "numeric" }
    : { month: "short", day: "numeric", year: "numeric" }
  return `Updated ${date.toLocaleDateString(undefined, options)}`
}

/** Apply `patch` to topic `name`, adding it when only memories reference it. */
function patchTopic(topics: MemoryTopic[], name: string, patch: Partial<MemoryTopic>): MemoryTopic[] {
  if (!topics.some((t) => t.name === name)) return [...topics, { name, summary: "", ...patch }]
  return topics.map((t) => t.name === name ? { ...t, ...patch } : t)
}

function isTopicNameTaken(scope: ScopeMemory, name: string): boolean {
  return topicRows(scope).some((row) => row.topic.name === name)
}

/** Load every scope once, apply changes locally and write each scope in order. */
function useMemoryLibrary() {
  const { projects } = useProject()
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
      .catch((e) => { if (!cancelled) setError((e as Error)?.message || "Failed to load memory") })
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
      await commitMemoryDoc(scope.scope, [], scope.projectSlug, null, scope.memories)
    })
    writes.current.set(key, write)
    return write
  }, [])

  const update = useCallback((key: string, apply: ScopeChange) => {
    change(key, apply)
    save(key).catch((e) => setError(`Could not save memory: ${(e as Error)?.message || "unknown error"}`))
  }, [change, save])

  return { scopes, error, setError, change, save, update }
}

function MemoryWindow({ onClose }: { onClose: () => void }) {
  const library = useMemoryLibrary()
  const columnPadding = useChatColumnPadding()
  const { schedule } = useUndoDelete()
  const [openTopic, setOpenTopic] = useState<{ scopeKey: string; name: string } | null>(null)
  const [expanded, setExpanded] = useState<Set<string> | null>(null)
  const { scopes, change, save, update, setError } = library

  // Projects holding memories start expanded, empty ones collapsed.
  const expandedScopes = expanded ?? new Set(scopes?.filter((s) => s.memories.length > 0).map((s) => s.key))

  const reportError = useCallback((e: unknown) => {
    setError(`Could not save memory: ${(e as Error)?.message || "unknown error"}`)
  }, [setError])

  const deleteTopic = useCallback((scopeKey: string, name: string) => {
    const scope = scopes?.find((s) => s.key === scopeKey)
    if (!scope) return
    const topicIndex = scope.topics.findIndex((t) => t.name === name)
    const removedTopic = scope.topics[topicIndex]
    const removedMemories = scope.memories.filter((m) => normalizeMemoryTopic(m) === name)
    change(scopeKey, (s) => ({
      topics: s.topics.filter((t) => t.name !== name),
      memories: s.memories.filter((m) => normalizeMemoryTopic(m) !== name),
    }))
    setOpenTopic(null)
    schedule(`memory-topic:${scopeKey}:${name}`, formatTopicHeading(name), () => save(scopeKey), () => {
      change(scopeKey, (s) => {
        const topics = [...s.topics]
        if (removedTopic) topics.splice(Math.min(topicIndex, topics.length), 0, removedTopic)
        return { topics, memories: [...s.memories, ...removedMemories] }
      })
      save(scopeKey).catch(reportError)
    })
  }, [scopes, change, save, schedule, reportError])

  const deleteMemory = useCallback((scopeKey: string, memoryId: string) => {
    const scope = scopes?.find((s) => s.key === scopeKey)
    const index = scope?.memories.findIndex((m) => m.id === memoryId) ?? -1
    if (!scope || index < 0) return
    const removed = scope.memories[index]
    change(scopeKey, (s) => ({ topics: s.topics, memories: s.memories.filter((m) => m.id !== memoryId) }))
    schedule(`memory:${memoryId}`, "memory", () => save(scopeKey), () => {
      change(scopeKey, (s) => {
        const memories = [...s.memories]
        memories.splice(Math.min(index, memories.length), 0, removed)
        return { topics: s.topics, memories }
      })
      save(scopeKey).catch(reportError)
    })
  }, [scopes, change, save, schedule, reportError])

  const createTopic = useCallback((scopeKey: string, title: string): string | void => {
    const scope = scopes?.find((s) => s.key === scopeKey)
    const name = topicNameFromTitle(title)
    if (!scope || !name) return
    if (isTopicNameTaken(scope, name)) return "A topic with this name exists"
    update(scopeKey, (s) => ({ topics: [...s.topics, { name, summary: "" }], memories: s.memories }))
    setOpenTopic({ scopeKey, name })
  }, [scopes, update])

  const activeScope = openTopic && scopes?.find((s) => s.key === openTopic.scopeKey)
  const activeRow = activeScope ? topicRows(activeScope).find((row) => row.topic.name === openTopic?.name) : undefined

  return (
    <FloatingWindow onClose={onClose} label="Memory" overlayClassName="z-[95]" overlayStyle={columnPadding}
      panelClassName="h-auto max-h-[min(86vh,840px)] max-w-4xl bg-surface">
      {activeScope && activeRow ? (
        <TopicPage
          key={`${activeScope.key}:${activeRow.topic.name}`}
          scope={activeScope}
          row={activeRow}
          onBack={() => setOpenTopic(null)}
          onRename={(title) => {
            const name = topicNameFromTitle(title)
            if (!name || name === activeRow.topic.name) return
            if (isTopicNameTaken(activeScope, name)) return "A topic with this name exists"
            const previous = activeRow.topic.name
            update(activeScope.key, (s) => ({
              topics: patchTopic(s.topics, previous, { name }),
              memories: s.memories.map((m) => normalizeMemoryTopic(m) === previous ? { ...m, topic: name } : m),
            }))
            setOpenTopic({ scopeKey: activeScope.key, name })
          }}
          onSummary={(summary) => update(activeScope.key, (s) => ({
            topics: patchTopic(s.topics, activeRow.topic.name, { summary }), memories: s.memories,
          }))}
          onAddMemory={(text) => {
            const now = new Date().toISOString()
            const memory: PreviewMemory = {
              id: newManualMemoryId(), text, topic: activeRow.topic.name, scope: activeScope.scope, created_at: now, updated_at: now,
            }
            update(activeScope.key, (s) => ({ topics: patchTopic(s.topics, activeRow.topic.name, {}), memories: [...s.memories, memory] }))
          }}
          onEditMemory={(id, text) => update(activeScope.key, (s) => ({
            topics: s.topics,
            memories: s.memories.map((m) => m.id === id ? { ...m, text, updated_at: new Date().toISOString() } : m),
          }))}
          onDeleteMemory={(id) => deleteMemory(activeScope.key, id)}
          onDelete={() => deleteTopic(activeScope.key, activeRow.topic.name)}
        />
      ) : (
        <>
          <div className="flex shrink-0 items-center justify-between px-6 pb-2 pt-5">
            <h2 className="text-base font-semibold text-ink">Memory</h2>
            <button onClick={onClose} aria-label="Close memory"
              className="rounded-md p-1 text-ink-subtle transition-colors hover:bg-hover hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted">
              <X className="h-4 w-4" />
            </button>
          </div>
          {scopes ? (
            <Overview
              scopes={scopes}
              expanded={expandedScopes}
              onToggle={(key) => setExpanded(() => {
                const next = new Set(expandedScopes)
                if (next.has(key)) next.delete(key)
                else next.add(key)
                return next
              })}
              onOpen={(scopeKey, name) => setOpenTopic({ scopeKey, name })}
              onCreate={createTopic}
              onDelete={deleteTopic}
            />
          ) : !library.error ? (
            <div className="flex items-center justify-center py-16">
              <Loader2 className="h-4 w-4 animate-spin text-ink-muted" aria-label="Loading memory" />
            </div>
          ) : null}
        </>
      )}
      {library.error && (
        <div role="alert" className="flex shrink-0 items-center justify-between gap-3 border-t border-divider/50 px-6 py-2 text-xs text-danger">
          <span>{library.error}</span>
          <button onClick={() => setError(null)} aria-label="Dismiss memory error" className="text-ink-muted hover:text-ink">
            <X className="h-3 w-3" />
          </button>
        </div>
      )}
    </FloatingWindow>
  )
}

function Overview({ scopes, expanded, onToggle, onOpen, onCreate, onDelete }: {
  scopes: ScopeMemory[]
  expanded: Set<string>
  onToggle: (scopeKey: string) => void
  onOpen: (scopeKey: string, name: string) => void
  onCreate: (scopeKey: string, title: string) => string | void
  onDelete: (scopeKey: string, name: string) => void
}) {
  const [creatingIn, setCreatingIn] = useState<string | null>(null)
  const [you, ...projects] = scopes

  const rows = (scope: ScopeMemory, indent: boolean) => (
    <>
      {topicRows(scope).map((row) => (
        <TopicListRow key={row.topic.name} row={row} indent={indent}
          onOpen={() => onOpen(scope.key, row.topic.name)}
          onDelete={() => onDelete(scope.key, row.topic.name)} />
      ))}
      {creatingIn === scope.key && (
        <div className={cn("border-b border-divider/40 px-3 py-2", indent && "pl-9")}>
          <EditableText value="" label={`New topic in ${scope.label}`} startEditing className="text-sm text-ink"
            onCommit={(title) => onCreate(scope.key, title)} onDone={() => setCreatingIn(null)} />
        </div>
      )}
    </>
  )

  const addButton = (scope: ScopeMemory) => (
    <button onClick={(e) => { e.stopPropagation(); setCreatingIn(scope.key) }} aria-label={`New topic in ${scope.label}`}
      className="rounded-md p-1 text-ink-subtle opacity-0 transition-opacity hover:bg-hover hover:text-ink group-hover/section:opacity-100 focus-visible:opacity-100 focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted">
      <Plus className="h-3.5 w-3.5" />
    </button>
  )

  return (
    <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-6">
      <section aria-label="You" className="group/section">
        <div className="flex items-center justify-between px-3 pb-2 pt-4">
          <h3 className="text-sm font-semibold text-ink">You</h3>
          {addButton(you)}
        </div>
        {rows(you, false)}
      </section>

      {projects.length > 0 && (
        <section aria-label="Projects">
          <h3 className="px-3 pb-2 pt-8 text-sm font-semibold text-ink">Projects</h3>
          {projects.map((project) => {
            const isExpanded = expanded.has(project.key)
            return (
              <div key={project.key} className="group/section">
                <div className="flex items-center rounded-lg pr-2 transition-colors hover:bg-hover">
                  <button onClick={() => onToggle(project.key)} aria-expanded={isExpanded}
                    className="flex min-w-0 flex-1 items-center gap-2 px-3 py-2.5 text-left text-sm text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted">
                    {isExpanded ? <ChevronDown className="h-3.5 w-3.5 shrink-0 text-ink-subtle" /> : <ChevronRight className="h-3.5 w-3.5 shrink-0 text-ink-subtle" />}
                    <span className="truncate">{project.label}</span>
                  </button>
                  {addButton(project)}
                </div>
                {(isExpanded || creatingIn === project.key) && rows(project, true)}
              </div>
            )
          })}
        </section>
      )}
    </div>
  )
}

function TopicListRow({ row, indent, onOpen, onDelete }: {
  row: TopicRow
  indent: boolean
  onOpen: () => void
  onDelete: () => void
}) {
  const title = formatTopicHeading(row.topic.name)
  return (
    <div className="group/row flex items-center rounded-lg border-b border-divider/40 transition-colors hover:border-transparent hover:bg-hover">
      <button onClick={onOpen} aria-label={`Open ${title}`}
        className={cn("grid min-w-0 flex-1 grid-cols-[minmax(0,14rem)_minmax(0,1fr)_9rem] items-center gap-4 rounded-lg px-3 py-2.5 text-left text-sm focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted", indent && "pl-9")}>
        <span className="truncate text-ink">{title}</span>
        <span className="truncate text-ink-muted">{row.topic.summary}</span>
        <span className="truncate text-ink-subtle">{formatUpdated(row.updatedAt)}</span>
      </button>
      <button onClick={onDelete} aria-label={`Delete ${title}`}
        className="mr-2 rounded-md p-1.5 text-ink-subtle opacity-0 transition-opacity hover:text-danger group-hover/row:opacity-100 focus-visible:opacity-100 focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted">
        <Trash2 className="h-3.5 w-3.5" />
      </button>
    </div>
  )
}

function TopicPage({ scope, row, onBack, onRename, onSummary, onAddMemory, onEditMemory, onDeleteMemory, onDelete }: {
  scope: ScopeMemory
  row: TopicRow
  onBack: () => void
  onRename: (title: string) => string | void
  onSummary: (summary: string) => void
  onAddMemory: (text: string) => void
  onEditMemory: (id: string, text: string) => void
  onDeleteMemory: (id: string) => void
  onDelete: () => void
}) {
  const [adding, setAdding] = useState(false)
  useEscapeLayer(onBack)
  const lastUpdated = row.updatedAt
    ? new Date(row.updatedAt).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" })
    : "Never"

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex shrink-0 items-center gap-2 px-5 pb-2 pt-5 text-sm">
        <button onClick={onBack}
          className="flex items-center gap-2 rounded-md px-1 py-0.5 text-ink-muted transition-colors hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted">
          <ArrowLeft className="h-4 w-4" />Memory
        </button>
        {scope.scope === "project" && <span className="truncate text-ink-subtle">/ {scope.label}</span>}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-6 pb-8">
        <div className="flex items-start justify-between gap-4 pt-3">
          <EditableText value={formatTopicHeading(row.topic.name)} label="Topic title" onCommit={onRename}
            className="min-w-0 flex-1 text-lg font-semibold text-ink" />
          <button onClick={onDelete}
            className="shrink-0 rounded-md bg-surface-elevated px-3 py-1.5 text-sm text-ink transition-colors hover:bg-hover focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted">
            Delete
          </button>
        </div>

        <h4 className="mt-6 text-xs text-ink-subtle">Last updated</h4>
        <p className="mt-1.5 text-sm text-ink">{lastUpdated}</p>

        <h4 className="mt-6 text-xs text-ink-subtle">Summary</h4>
        <EditableText value={row.topic.summary} label="Topic summary" multiline onCommit={(summary) => onSummary(summary)}
          className="mt-1.5 text-sm text-ink" />

        <h4 className="mt-6 text-xs text-ink-subtle">Details</h4>
        <ul className="mt-1.5 list-disc space-y-1.5 pl-6 text-sm leading-6 text-ink marker:text-ink-subtle">
          {row.memories.map((memory) => (
            <li key={memory.id} className="group/memory">
              <div className="flex items-start gap-2">
                <EditableText value={memory.text} label="Memory" multiline className="min-w-0 flex-1"
                  onCommit={(text) => onEditMemory(memory.id, text)} />
                <button onClick={() => onDeleteMemory(memory.id)} aria-label="Delete memory"
                  className="mt-1 shrink-0 rounded p-0.5 text-ink-subtle opacity-0 transition-opacity hover:text-danger group-hover/memory:opacity-100 focus-visible:opacity-100 focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted">
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>
            </li>
          ))}
          {adding && (
            <li>
              <EditableText value="" label="New memory" multiline startEditing onCommit={onAddMemory} onDone={() => setAdding(false)} />
            </li>
          )}
        </ul>
        {!adding && (
          <button onClick={() => setAdding(true)}
            className="mt-3 flex items-center gap-1.5 rounded-md px-1 py-0.5 text-xs text-ink-muted transition-colors hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted">
            <Plus className="h-3.5 w-3.5" />Add
          </button>
        )}
      </div>
    </div>
  )
}

/**
 * Text that turns into an input on click: Enter saves, Shift+Enter adds a line when `multiline`, Esc cancels.
 *
 * - `onCommit` runs only for a changed, non-empty value, a returned string is shown as the error and keeps editing
 * - `onDone` runs whenever editing ends, `startEditing` opens it in edit mode (new entries)
 */
function EditableText({ value, label, multiline = false, startEditing = false, className, onCommit, onDone }: {
  value: string
  label: string
  multiline?: boolean
  startEditing?: boolean
  className?: string
  onCommit: (next: string) => string | void
  onDone?: () => void
}) {
  const [editing, setEditing] = useState(startEditing)
  const [draft, setDraft] = useState(value)
  const [error, setError] = useState<string | null>(null)

  const stop = useCallback(() => {
    setEditing(false)
    setError(null)
    onDone?.()
  }, [onDone])
  useEscapeLayer(stop, editing)

  const commit = () => {
    const next = draft.trim()
    if (next && next !== value) {
      const failure = onCommit(next)
      if (failure) { setError(failure); return }
    }
    stop()
  }

  if (!editing) {
    return (
      <button onClick={() => { setDraft(value); setEditing(true) }} aria-label={`Edit ${label}`}
        className={cn("block w-full whitespace-pre-wrap rounded text-left focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted", className)}>
        {value || <span className="text-ink-faint">Empty</span>}
      </button>
    )
  }

  return (
    <div className={className}>
      <textarea
        autoFocus aria-label={label} value={draft} rows={Math.max(1, draft.split("\n").length)}
        onChange={(e) => { setDraft(e.target.value); setError(null) }}
        onFocus={(e) => e.currentTarget.setSelectionRange(e.currentTarget.value.length, e.currentTarget.value.length)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !(multiline && e.shiftKey)) { e.preventDefault(); commit() }
        }}
        className="block w-full resize-none rounded bg-transparent outline-none ring-1 ring-divider focus:ring-divider-strong"
      />
      {error && <p role="alert" className="mt-1 text-xs text-danger">{error}</p>}
    </div>
  )
}
