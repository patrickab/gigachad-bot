"use client"

import { useState } from "react"
import { ChevronDown, ChevronRight, Plus, Shuffle, Trash2 } from "lucide-react"
import { cn } from "@/lib/utils"
import { EditableText } from "@/features/memory/EditableText"
import { formatTopicHeading, formatUpdated } from "@/features/memory/memoryUtils"
import { topicRows, type ScopeMemory, type TopicRow } from "@/features/memory/scopeEdits"

const SECTION_ACTION_CLASS = "rounded-md p-1 text-ink-subtle opacity-0 transition-opacity hover:bg-hover hover:text-ink group-hover/section:opacity-100 focus-visible:opacity-100 focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"

/** Every scope with its topics: "You" first, then collapsible projects. */
export function Overview({ scopes, expanded, onToggle, onOpen, onCreate, onDelete, onRegroup }: {
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
