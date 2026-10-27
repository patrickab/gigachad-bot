"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { ArrowLeft, ChevronDown, ChevronRight, Loader2, Plus, Shuffle, Trash2, X } from "lucide-react"
import { FloatingWindow } from "@/components/ui/FloatingWindow"
import { InlineConfirm } from "@/components/ui/InlineConfirm"
import { StyledSelect } from "@/components/ui/StyledSelect"
import { useProject } from "@/contexts/ProjectContext"
import { useUndoDelete } from "@/contexts/UndoDeleteContext"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import { getMemories, getTopics, regroupMemories, saveMemories, saveTopics } from "@/lib/backend/api"
import { errorMessage, toggled, cn } from "@/lib/utils"
import { EditableText } from "@/features/memory/EditableText"
import { formatTopicHeading, groupByTopicOrder, newManualMemoryId, normalizeMemoryTopic, topicNameFromTitle } from "@/features/memory/memoryUtils"
import { useMemoryViewer } from "@/features/memory/MemoryViewerContext"
import type { MemoryScope, MemoryTopic, StoredMemory } from "@/lib/domain/types"

/** One memory scope: the global profile ("You") or one project. */
interface ScopeMemory {
  key: string
  scope: MemoryScope
  projectSlug: string | null
  label: string
  topics: MemoryTopic[]
  memories: StoredMemory[]
}

type ScopeContent = Pick<ScopeMemory, "topics" | "memories">
type ScopeChange = (scope: ScopeMemory) => Partial<ScopeContent>

interface TopicRow {
  topic: MemoryTopic
  memories: StoredMemory[]
  updatedAt: string | null
}

/** Regroup proposal of one scope, `proposal` null while the model works. */
interface RegroupState {
  scopeKey: string
  proposal: ScopeContent | null
}

type View = { kind: "overview" } | { kind: "topic"; scopeKey: string; name: string } | { kind: "regroup" } & RegroupState

const DAY_MS = 86_400_000
const SECTION_ACTION_CLASS = "rounded-md p-1 text-ink-subtle opacity-0 transition-opacity hover:bg-hover hover:text-ink group-hover/section:opacity-100 focus-visible:opacity-100 focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"
const BUTTON_CLASS = "shrink-0 rounded-md bg-surface-elevated px-3 py-1.5 text-sm text-ink transition-colors hover:bg-hover disabled:opacity-40 focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"

export function MemoryViewer() {
  const { isOpen, closeMemoryViewer } = useMemoryViewer()
  return isOpen ? <MemoryWindow onClose={closeMemoryViewer} /> : null
}

function latestUpdate(memories: StoredMemory[]): string | null {
  return memories.reduce<string | null>((latest, m) => {
    const stamp = m.updated_at ?? m.created_at ?? null
    return stamp && (!latest || stamp > latest) ? stamp : latest
  }, null)
}

/** Topics in stored order plus topics only referenced by memories, each with its memories. */
function topicRows(content: ScopeContent): TopicRow[] {
  return groupByTopicOrder(content.topics, content.memories).map(({ topic, items }) => ({ topic, memories: items, updatedAt: latestUpdate(items) }))
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

function topicNames(scope: ScopeMemory): string[] {
  return topicRows(scope).map((row) => row.topic.name)
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

  return { scopes, error, setError, reportError, change, save, update }
}

function MemoryWindow({ onClose }: { onClose: () => void }) {
  const library = useMemoryLibrary()
  const { schedule } = useUndoDelete()
  const [view, setView] = useState<View>({ kind: "overview" })
  const [expanded, setExpanded] = useState<Set<string> | null>(null)
  const { scopes, change, save, update, setError, reportError } = library
  const back = useCallback(() => setView({ kind: "overview" }), [])

  // Projects holding memories start expanded, empty ones collapsed.
  const expandedScopes = expanded ?? new Set(scopes?.filter((s) => s.memories.length > 0).map((s) => s.key))

  const scopeOf = useCallback((key: string) => scopes?.find((s) => s.key === key), [scopes])

  /** Remove content optimistically; the delete is written when the undo window ends, `restore` puts it back and writes again. */
  const removeWithUndo = useCallback((scopeKey: string, undoKey: string, label: string, remove: ScopeChange, restore: ScopeChange) => {
    change(scopeKey, remove)
    schedule(undoKey, label, () => save(scopeKey), () => {
      change(scopeKey, restore)
      save(scopeKey).catch(reportError)
    })
  }, [change, save, schedule, reportError])

  const deleteTopic = useCallback((scopeKey: string, name: string) => {
    const scope = scopeOf(scopeKey)
    if (!scope) return
    const topicIndex = scope.topics.findIndex((t) => t.name === name)
    const removedTopic = scope.topics[topicIndex]
    const removedMemories = scope.memories.filter((m) => normalizeMemoryTopic(m) === name)
    removeWithUndo(scopeKey, `memory-topic:${scopeKey}:${name}`, formatTopicHeading(name), (s) => ({
      topics: s.topics.filter((t) => t.name !== name),
      memories: s.memories.filter((m) => normalizeMemoryTopic(m) !== name),
    }), (s) => {
      const topics = [...s.topics]
      if (removedTopic) topics.splice(Math.min(topicIndex, topics.length), 0, removedTopic)
      return { topics, memories: [...s.memories, ...removedMemories] }
    })
    back()
  }, [scopeOf, removeWithUndo, back])

  const deleteMemory = useCallback((scopeKey: string, memoryId: string) => {
    const scope = scopeOf(scopeKey)
    const index = scope?.memories.findIndex((m) => m.id === memoryId) ?? -1
    if (!scope || index < 0) return
    const removed = scope.memories[index]
    removeWithUndo(scopeKey, `memory:${memoryId}`, "memory", (s) => ({ memories: s.memories.filter((m) => m.id !== memoryId) }), (s) => {
      const memories = [...s.memories]
      memories.splice(Math.min(index, memories.length), 0, removed)
      return { memories }
    })
  }, [scopeOf, removeWithUndo])

  const createTopic = useCallback((scopeKey: string, title: string): string | void => {
    const scope = scopeOf(scopeKey)
    const name = topicNameFromTitle(title)
    if (!scope || !name) return
    if (topicNames(scope).includes(name)) return "A topic with this name exists"
    update(scopeKey, (s) => ({ topics: [...s.topics, { name, summary: "" }] }))
    setView({ kind: "topic", scopeKey, name })
  }, [scopeOf, update])

  const startRegroup = useCallback((scopeKey: string) => {
    const scope = scopeOf(scopeKey)
    if (!scope) return
    setView({ kind: "regroup", scopeKey, proposal: null })
    regroupMemories(scope.scope, scope.projectSlug)
      .then((proposal) => setView((current) => current.kind === "regroup" && current.scopeKey === scopeKey ? { ...current, proposal } : current))
      .catch((e) => {
        setError(`Could not regroup: ${errorMessage(e, "unknown error")}`)
        setView((current) => current.kind === "regroup" && current.scopeKey === scopeKey ? { kind: "overview" } : current)
      })
  }, [scopeOf, setError])

  const activeScope = view.kind !== "overview" ? scopeOf(view.scopeKey) : undefined
  const activeRow = activeScope && view.kind === "topic" ? topicRows(activeScope).find((row) => row.topic.name === view.name) : undefined

  let content
  if (activeScope && view.kind === "regroup") {
    content = (
      <RegroupPage scope={activeScope} proposal={view.proposal} onBack={back}
        onApply={(proposal) => { update(activeScope.key, () => proposal); back() }} />
    )
  } else if (activeScope && activeRow) {
    const name = activeRow.topic.name
    content = (
      <TopicPage
        key={`${activeScope.key}:${name}`}
        scope={activeScope}
        row={activeRow}
        onBack={back}
        onRename={(next) => {
          update(activeScope.key, (s) => ({
            topics: patchTopic(s.topics, name, { name: next }),
            memories: s.memories.map((m) => normalizeMemoryTopic(m) === name ? { ...m, topic: next } : m),
          }))
          setView({ kind: "topic", scopeKey: activeScope.key, name: next })
        }}
        onMerge={(target) => {
          update(activeScope.key, (s) => ({
            topics: patchTopic(s.topics.filter((t) => t.name !== name), target, {}),
            memories: s.memories.map((m) => normalizeMemoryTopic(m) === name ? { ...m, topic: target } : m),
          }))
          setView({ kind: "topic", scopeKey: activeScope.key, name: target })
        }}
        onSummary={(summary) => update(activeScope.key, (s) => ({ topics: patchTopic(s.topics, name, { summary }) }))}
        onAddMemory={(text) => {
          const now = new Date().toISOString()
          const memory: StoredMemory = { id: newManualMemoryId(), text, topic: name, scope: activeScope.scope, created_at: now, updated_at: now }
          update(activeScope.key, (s) => ({ topics: patchTopic(s.topics, name, {}), memories: [...s.memories, memory] }))
        }}
        onEditMemory={(id, text) => update(activeScope.key, (s) => ({
          memories: s.memories.map((m) => m.id === id ? { ...m, text, updated_at: new Date().toISOString() } : m),
        }))}
        onMoveMemory={(id, topic) => update(activeScope.key, (s) => ({
          topics: patchTopic(s.topics, topic, {}),
          memories: s.memories.map((m) => m.id === id ? { ...m, topic } : m),
        }))}
        onDeleteMemory={(id) => deleteMemory(activeScope.key, id)}
        onDelete={() => deleteTopic(activeScope.key, name)}
      />
    )
  } else {
    content = (
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
            onToggle={(key) => setExpanded(toggled(expandedScopes, key))}
            onOpen={(scopeKey, name) => setView({ kind: "topic", scopeKey, name })}
            onCreate={createTopic}
            onDelete={deleteTopic}
            onRegroup={startRegroup}
          />
        ) : !library.error ? <Spinner label="Loading memory" /> : null}
      </>
    )
  }

  return (
    <FloatingWindow onClose={onClose} label="Memory" overlayClassName="z-[95]"
      panelClassName="h-auto max-h-[min(86vh,840px)] max-w-4xl bg-surface">
      {content}
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

function Overview({ scopes, expanded, onToggle, onOpen, onCreate, onDelete, onRegroup }: {
  scopes: ScopeMemory[]
  expanded: Set<string>
  onToggle: (scopeKey: string) => void
  onOpen: (scopeKey: string, name: string) => void
  onCreate: (scopeKey: string, title: string) => string | void
  onDelete: (scopeKey: string, name: string) => void
  onRegroup: (scopeKey: string) => void
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

  const actions = (scope: ScopeMemory) => (
    <div className="flex items-center gap-0.5">
      {scope.memories.length > 1 && (
        <button onClick={(e) => { e.stopPropagation(); onRegroup(scope.key) }} aria-label={`Regroup topics in ${scope.label}`} className={SECTION_ACTION_CLASS}>
          <Shuffle className="h-3.5 w-3.5" />
        </button>
      )}
      <button onClick={(e) => { e.stopPropagation(); setCreatingIn(scope.key) }} aria-label={`New topic in ${scope.label}`} className={SECTION_ACTION_CLASS}>
        <Plus className="h-3.5 w-3.5" />
      </button>
    </div>
  )

  return (
    <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-6">
      <section aria-label="You" className="group/section">
        <div className="flex items-center justify-between px-3 pb-2 pt-4">
          <h3 className="text-sm font-semibold text-ink">You</h3>
          {actions(you)}
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
                  {actions(project)}
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

/** Centered loading indicator of a memory view. */
function Spinner({ label }: { label: string }) {
  return (
    <div className="flex items-center justify-center py-16">
      <Loader2 className="h-4 w-4 animate-spin text-ink-muted" aria-label={label} />
    </div>
  )
}

function PageHeader({ scope, onBack }: { scope: ScopeMemory; onBack: () => void }) {
  return (
    <div className="flex shrink-0 items-center gap-2 px-5 pb-2 pt-5 text-sm">
      <button onClick={onBack}
        className="flex items-center gap-2 rounded-md px-1 py-0.5 text-ink-muted transition-colors hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted">
        <ArrowLeft className="h-4 w-4" />Memory
      </button>
      {scope.scope === "project" && <span className="truncate text-ink-subtle">/ {scope.label}</span>}
    </div>
  )
}

/** One topic: title, summary and facts, all editable in place. A title naming another topic offers to merge into it. */
function TopicPage({ scope, row, onBack, onRename, onMerge, onSummary, onAddMemory, onEditMemory, onMoveMemory, onDeleteMemory, onDelete }: {
  scope: ScopeMemory
  row: TopicRow
  onBack: () => void
  onRename: (name: string) => void
  onMerge: (target: string) => void
  onSummary: (summary: string) => void
  onAddMemory: (text: string) => void
  onEditMemory: (id: string, text: string) => void
  onMoveMemory: (id: string, topic: string) => void
  onDeleteMemory: (id: string) => void
  onDelete: () => void
}) {
  const [adding, setAdding] = useState(false)
  const [mergeTarget, setMergeTarget] = useState<string | null>(null)
  useEscapeLayer(onBack)
  const names = topicNames(scope)
  const topicOptions = names.map((name) => ({ value: name, label: formatTopicHeading(name) }))
  const lastUpdated = row.updatedAt
    ? new Date(row.updatedAt).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" })
    : "Never"

  const rename = (title: string) => {
    const name = topicNameFromTitle(title)
    if (!name || name === row.topic.name) return
    if (names.includes(name)) setMergeTarget(name)
    else onRename(name)
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <PageHeader scope={scope} onBack={onBack} />
      <div className="min-h-0 flex-1 overflow-y-auto px-6 pb-8">
        <div className="flex items-start justify-between gap-4 pt-3">
          <EditableText value={formatTopicHeading(row.topic.name)} label="Topic title" onCommit={rename}
            className="min-w-0 flex-1 text-lg font-semibold text-ink" />
          <button onClick={onDelete} className={BUTTON_CLASS}>Delete</button>
        </div>
        {mergeTarget && (
          <div className="mt-2 max-w-md">
            <InlineConfirm question={`Merge into ${formatTopicHeading(mergeTarget)}?`}
              onYes={() => onMerge(mergeTarget)} onNo={() => setMergeTarget(null)} />
          </div>
        )}

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
                <div className="flex shrink-0 items-center gap-1 opacity-0 transition-opacity focus-within:opacity-100 group-hover/memory:opacity-100">
                  {topicOptions.length > 1 && (
                    <StyledSelect variant="inline" ariaLabel="Move memory to topic" value={row.topic.name} options={topicOptions}
                      onChange={(topic) => onMoveMemory(memory.id, topic)} className="text-xs text-ink-subtle hover:text-ink" />
                  )}
                  <button onClick={() => onDeleteMemory(memory.id)} aria-label="Delete memory"
                    className="rounded p-0.5 text-ink-subtle hover:text-danger focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted">
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
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

/** Proposed regrouping of one scope: facts that change topic name their previous one. */
function RegroupPage({ scope, proposal, onBack, onApply }: {
  scope: ScopeMemory
  proposal: ScopeContent | null
  onBack: () => void
  onApply: (proposal: ScopeContent) => void
}) {
  useEscapeLayer(onBack)
  const previousTopic = new Map(scope.memories.map((m) => [m.id, normalizeMemoryTopic(m)]))

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <PageHeader scope={scope} onBack={onBack} />
      <div className="flex shrink-0 items-start justify-between gap-4 px-6 pt-3">
        <h2 className="text-lg font-semibold text-ink">Regroup topics</h2>
        <div className="flex items-center gap-2">
          <button onClick={onBack} className="px-2 py-1.5 text-sm text-ink-muted transition-colors hover:text-ink">Cancel</button>
          <button onClick={() => proposal && onApply(proposal)} disabled={!proposal} className={BUTTON_CLASS}>Apply</button>
        </div>
      </div>
      {proposal ? (
        <div className="min-h-0 flex-1 overflow-y-auto px-6 pb-8">
          {topicRows(proposal).map(({ topic, memories }) => (
            <section key={topic.name} className="mt-6">
              <h3 className="text-sm font-semibold text-ink">{formatTopicHeading(topic.name)}</h3>
              {topic.summary && <p className="mt-0.5 text-sm text-ink-muted">{topic.summary}</p>}
              <ul className="mt-1.5 list-disc space-y-1 pl-6 text-sm leading-6 text-ink marker:text-ink-subtle">
                {memories.map((memory) => {
                  const from = previousTopic.get(memory.id)
                  return (
                    <li key={memory.id}>
                      {memory.text}
                      {from && from !== topic.name && <span className="text-ink-subtle"> · from {formatTopicHeading(from)}</span>}
                    </li>
                  )
                })}
              </ul>
            </section>
          ))}
        </div>
      ) : <Spinner label="Regrouping" />}
    </div>
  )
}
