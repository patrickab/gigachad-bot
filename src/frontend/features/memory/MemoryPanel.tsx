"use client"

import { Loader2, Minus, Pencil, Plus, RotateCcw, X } from "lucide-react"
import { FloatingWindow } from "@/components/ui/FloatingWindow"
import { StyledSelect } from "@/components/ui/StyledSelect"
import { useEventListener } from "@/hooks/useEventListener"
import { cn, isModKey } from "@/lib/utils"
import { EditableText } from "@/features/memory/EditableText"
import { countOperations, type MemoryPanelState, type MemoryReviewActions, type ReviewOperation, type ReviewScope, type ReviewTopic } from "@/features/memory/useMemorizeFlow"
import { formatTopicHeading } from "@/features/memory/memoryUtils"
import { MEMORY_BUTTON_CLASS, MEMORY_TEXT_BUTTON_CLASS } from "@/features/memory/styles"
import type { MemoryScope, MemoryTopic } from "@/lib/domain/memory"

const SCOPE_LABELS: Record<MemoryScope, string> = { global: "You", project: "Project" }
const OP_ICONS = { add: Plus, update: Pencil, delete: Minus }
const OP_LABELS = { add: "Add", update: "Update", delete: "Delete" }
const ICON_BUTTON_CLASS = "rounded p-0.5 text-ink-subtle transition-colors hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"

/** Topics of one reviewed scope in display order: stored topics first, then new ones, each with its operations. */
function topicGroups(review: ReviewScope, stored: MemoryTopic[]) {
  const order = [...stored.map((t) => t.name), ...review.topics.map((t) => t.name), ...review.operations.map((o) => o.topic)]
  const names = [...new Set(order)].filter((name) => review.operations.some((o) => o.topic === name) || review.topics.some((t) => t.name === name))
  return names.map((name) => ({
    name,
    proposal: review.topics.find((t) => t.name === name),
    operations: review.operations.filter((o) => o.topic === name),
  }))
}

/** Review proposed memory changes in one screen: Ctrl+Enter commits accepted ones, Esc cancels. */
export function MemoryPanel({ state, actions }: { state: MemoryPanelState; actions: MemoryReviewActions }) {
  const reviewing = state.phase === "review"

  useEventListener("document", "keydown", (e) => {
    if (isModKey(e) && e.key === "Enter") {
      e.preventDefault()
      void actions.commit()
    }
  }, { active: reviewing })

  const acceptedCount = state.phase === "review" ? countOperations(state.scopes, true) : 0

  return (
    <FloatingWindow onClose={state.phase === "extracting" ? undefined : () => void actions.cancel()} label="Review memory"
      overlayClassName="z-[80]" panelClassName="h-auto max-h-[min(86vh,840px)] max-w-4xl bg-surface">
      {state.phase === "extracting" && (
        <div className="flex items-center gap-2 px-6 py-5 text-sm text-ink-muted">
          <Loader2 className="h-4 w-4 animate-spin" />Reading the conversation
        </div>
      )}
      {state.phase === "error" && (
        <div className="flex items-center justify-between gap-4 px-6 py-5">
          <p className="text-sm text-ink-muted">{state.error}</p>
          <button onClick={() => void actions.cancel()} aria-label="Close" className="rounded-md p-1 text-ink-subtle hover:bg-hover hover:text-ink">
            <X className="h-4 w-4" />
          </button>
        </div>
      )}
      {state.phase === "review" && (
        <>
          <div className="flex shrink-0 items-center justify-between gap-4 px-6 pb-2 pt-5">
            <h2 className="text-base font-semibold text-ink">Review memory</h2>
            <div className="flex items-center gap-2">
              <button onClick={() => void actions.cancel()} disabled={state.committing}
                className={`${MEMORY_TEXT_BUTTON_CLASS} disabled:opacity-40`}>Cancel</button>
              <button onClick={() => void actions.commit()} disabled={state.committing}
                className={`flex items-center gap-1.5 ${MEMORY_BUTTON_CLASS}`}>
                {state.committing && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
                {acceptedCount > 0 ? `Commit ${acceptedCount}` : "Commit"}
              </button>
            </div>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto px-6 pb-8">
            {(["global", "project"] as const).map((scope) => {
              const review = state.scopes[scope]
              if (!review) return null
              const stored = state.storedTopics[scope] ?? []
              const options = [...new Set([...stored.map((t) => t.name), ...review.topics.map((t) => t.name)])]
                .map((name) => ({ value: name, label: formatTopicHeading(name) }))
              return (
                <section key={scope} aria-label={SCOPE_LABELS[scope]} className="pt-4">
                  <h3 className="text-sm font-semibold text-ink">{SCOPE_LABELS[scope]}</h3>
                  {topicGroups(review, stored).map((group) => (
                    <TopicGroup key={group.name} scope={scope} name={group.name} proposal={group.proposal}
                      operations={group.operations} topicOptions={options} actions={actions} />
                  ))}
                </section>
              )
            })}
          </div>
        </>
      )}
    </FloatingWindow>
  )
}

function TopicGroup({ scope, name, proposal, operations, topicOptions, actions }: {
  scope: MemoryScope
  name: string
  proposal: ReviewTopic | undefined
  operations: ReviewOperation[]
  topicOptions: { value: string; label: string }[]
  actions: MemoryReviewActions
}) {
  const isNew = proposal?.previous_summary === null
  const revision = proposal && !isNew ? proposal : undefined

  return (
    <div className="mt-4 border-b border-divider/40 pb-4">
      <div className="flex items-baseline gap-2">
        {isNew ? (
          <EditableText value={formatTopicHeading(name)} label={`Title of new topic ${formatTopicHeading(name)}`}
            onCommit={(title) => actions.renameTopic(scope, name, title)} className="w-auto text-sm font-medium text-ink" />
        ) : (
          <span className="text-sm font-medium text-ink">{formatTopicHeading(name)}</span>
        )}
        {isNew && <span className="text-xs text-ink-subtle">new topic</span>}
      </div>

      {isNew && proposal && (
        <EditableText value={proposal.summary} label={`Summary of ${formatTopicHeading(name)}`} multiline
          onCommit={(summary) => actions.editTopicSummary(scope, name, summary)} className="mt-0.5 text-sm text-ink-muted" />
      )}
      {revision && (
        <div className={cn("group/row mt-1 flex items-start gap-2 text-sm", revision.excluded && "opacity-40")}>
          <div className="min-w-0 flex-1">
            <p className="text-ink-subtle line-through">{revision.previous_summary}</p>
            <EditableText value={revision.summary} label={`Summary of ${formatTopicHeading(name)}`} multiline
              onCommit={(summary) => actions.editTopicSummary(scope, name, summary)} className="text-ink-muted" />
          </div>
          <ExcludeButton excluded={revision.excluded} label={`summary of ${formatTopicHeading(name)}`} onToggle={() => actions.toggleTopic(scope, name)} />
        </div>
      )}

      <ul className="mt-2 space-y-1.5">
        {operations.map((operation) => (
          <OperationRow key={operation.id} scope={scope} operation={operation} topicOptions={topicOptions} actions={actions} />
        ))}
      </ul>
    </div>
  )
}

function OperationRow({ scope, operation, topicOptions, actions }: {
  scope: MemoryScope
  operation: ReviewOperation
  topicOptions: { value: string; label: string }[]
  actions: MemoryReviewActions
}) {
  const Icon = OP_ICONS[operation.op]
  return (
    <li className={cn("group/row flex items-start gap-2 text-sm leading-6", operation.excluded && "opacity-40")}>
      <Icon className="mt-1 h-3.5 w-3.5 shrink-0 text-ink-subtle" aria-label={OP_LABELS[operation.op]} />
      <div className="min-w-0 flex-1">
        {operation.target_text && (
          <p className="text-ink-subtle line-through">{operation.target_text}</p>
        )}
        {operation.op !== "delete" && (
          <EditableText value={operation.text} label="Memory" multiline className="text-ink"
            onCommit={(text) => actions.editOperation(scope, operation.id, text)} />
        )}
      </div>
      {operation.index !== null && (
        <span className="mt-0.5 shrink-0 text-xs tabular-nums text-ink-subtle" aria-label="Relevance rank">#{operation.index + 1}</span>
      )}
      <div className="flex shrink-0 items-center gap-1">
        {operation.op !== "delete" && topicOptions.length > 1 && (
          <div className="opacity-0 transition-opacity focus-within:opacity-100 group-hover/row:opacity-100">
            <StyledSelect variant="inline" ariaLabel="Move to topic" value={operation.topic} options={topicOptions}
              onChange={(topic) => actions.moveOperation(scope, operation.id, topic)} className="text-xs text-ink-subtle hover:text-ink" />
          </div>
        )}
        <ExcludeButton excluded={operation.excluded} label="change" onToggle={() => actions.toggleOperation(scope, operation.id)} />
      </div>
    </li>
  )
}

function ExcludeButton({ excluded, label, onToggle }: { excluded: boolean; label: string; onToggle: () => void }) {
  return (
    <button onClick={onToggle} aria-label={excluded ? `Include ${label}` : `Deny ${label}`} aria-pressed={excluded} className={ICON_BUTTON_CLASS}>
      {excluded ? <RotateCcw className="h-3.5 w-3.5" /> : <X className="h-3.5 w-3.5" />}
    </button>
  )
}
