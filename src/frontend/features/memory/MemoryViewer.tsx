"use client"

import { useCallback, useState } from "react"
import { X } from "lucide-react"
import { FloatingWindow } from "@/components/ui/FloatingWindow"
import { regroupMemories } from "@/lib/backend/api"
import { errorMessage, toggled } from "@/lib/utils"
import { topicNameFromTitle } from "@/features/memory/memoryUtils"
import { useMemoryViewer } from "@/features/memory/MemoryViewerContext"
import { Spinner } from "@/features/memory/MemoryChrome"
import { Overview } from "@/features/memory/Overview"
import { RankingPage } from "@/features/memory/RankingPage"
import { RegroupPage } from "@/features/memory/RegroupPage"
import { TopicPage } from "@/features/memory/TopicPage"
import { createTopic, topicNames, topicRows, type ScopeContent } from "@/features/memory/scopeEdits"
import { useMemoryLibrary } from "@/features/memory/useMemoryLibrary"

/** Regroup proposal of one scope, `proposal` null while the model works. */
interface RegroupState {
  scopeKey: string
  proposal: ScopeContent | null
}

type View = { kind: "overview" } | { kind: "topic"; scopeKey: string; name: string } | { kind: "ranking"; scopeKey: string } | { kind: "regroup" } & RegroupState

export function MemoryViewer() {
  const { isOpen, closeMemoryViewer } = useMemoryViewer()
  return isOpen ? <MemoryWindow onClose={closeMemoryViewer} /> : null
}

function MemoryWindow({ onClose }: { onClose: () => void }) {
  const { scopes, error, setError, scopeOf, update, removeTopic, removeMemory } = useMemoryLibrary()
  const [view, setView] = useState<View>({ kind: "overview" })
  const [expanded, setExpanded] = useState<Set<string> | null>(null)
  const back = useCallback(() => setView({ kind: "overview" }), [])

  // Projects holding memories start expanded, empty ones collapsed.
  const expandedScopes = expanded ?? new Set(scopes?.filter((s) => s.memories.length > 0).map((s) => s.key))

  const deleteTopic = (scopeKey: string, name: string) => {
    removeTopic(scopeKey, name)
    back()
  }

  const addTopic = (scopeKey: string, title: string): string | void => {
    const scope = scopeOf(scopeKey)
    const name = topicNameFromTitle(title)
    if (!scope || !name) return
    if (topicNames(scope).includes(name)) return "A topic with this name exists"
    update(scopeKey, createTopic(name))
    setView({ kind: "topic", scopeKey, name })
  }

  const startRegroup = (scopeKey: string) => {
    const scope = scopeOf(scopeKey)
    if (!scope) return
    const settle = (next: (current: View & { kind: "regroup" }) => View) =>
      setView((current) => current.kind === "regroup" && current.scopeKey === scopeKey ? next(current) : current)
    setView({ kind: "regroup", scopeKey, proposal: null })
    regroupMemories(scope.scope, scope.projectSlug)
      .then((proposal) => settle((current) => ({ ...current, proposal })))
      .catch((e) => {
        setError(`Could not regroup: ${errorMessage(e, "unknown error")}`)
        settle(() => ({ kind: "overview" }))
      })
  }

  const activeScope = view.kind !== "overview" ? scopeOf(view.scopeKey) : undefined
  const activeRow = activeScope && view.kind === "topic" ? topicRows(activeScope).find((row) => row.topic.name === view.name) : undefined

  let content
  if (activeScope && view.kind === "regroup") {
    content = (
      <RegroupPage scope={activeScope} proposal={view.proposal} onBack={back}
        onApply={(proposal) => { update(activeScope.key, () => proposal); back() }} />
    )
  } else if (activeScope && view.kind === "ranking") {
    content = <RankingPage scope={activeScope} onBack={back} onEdit={(change) => update(activeScope.key, change)} />
  } else if (activeScope && activeRow) {
    content = (
      <TopicPage
        key={`${activeScope.key}:${activeRow.topic.name}`}
        scope={activeScope}
        row={activeRow}
        onBack={back}
        onEdit={(change) => update(activeScope.key, change)}
        onOpen={(name) => setView({ kind: "topic", scopeKey: activeScope.key, name })}
        onDeleteMemory={(id) => removeMemory(activeScope.key, id)}
        onDelete={() => deleteTopic(activeScope.key, activeRow.topic.name)}
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
            onCreate={addTopic}
            onDelete={deleteTopic}
            onRegroup={startRegroup}
            onRank={(scopeKey) => setView({ kind: "ranking", scopeKey })}
          />
        ) : !error ? <Spinner label="Loading memory" /> : null}
      </>
    )
  }

  return (
    <FloatingWindow onClose={onClose} label="Memory" overlayClassName="z-[95]"
      panelClassName="h-auto max-h-[min(86vh,840px)] max-w-4xl bg-surface">
      {content}
      {error && (
        <div role="alert" className="flex shrink-0 items-center justify-between gap-3 border-t border-divider/50 px-6 py-2 text-xs text-danger">
          <span>{error}</span>
          <button onClick={() => setError(null)} aria-label="Dismiss memory error" className="text-ink-muted hover:text-ink">
            <X className="h-3 w-3" />
          </button>
        </div>
      )}
    </FloatingWindow>
  )
}
