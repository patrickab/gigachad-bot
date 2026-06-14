"use client"

import { useCallback, useEffect, useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Check, FolderOpen, GitCompare, Globe, Loader2, Plus, X } from "lucide-react"
import { FloatingWindow } from "@/components/FloatingWindow"
import { LaTeXMarkdown } from "@/components/LaTeXMarkdown"
import { computeLineDiff } from "@/lib/diff"
import type { BoundMemoryActions, MemoryPanelState } from "@/hooks/useCommandBar"
import type { MemoryStatus, PreviewMemory, ProposedMemory } from "@/lib/types"

const STATUS_STYLES: Record<MemoryStatus, string> = {
  "new": "bg-emerald-500/15 text-emerald-400",
  "combined": "bg-amber-500/15 text-amber-400",
  "pre-existing": "bg-surface-elevated text-ink-subtle",
}

// Mirrors the backend `render_memories_as_markdown` so live edits/removals in the
// review panel update the diff without another round-trip.
export function renderMemoriesMarkdown(memories: PreviewMemory[], title: string): string {
  const groups = new Map<string, string[]>()
  for (const m of memories) {
    const arr = groups.get(m.kind) ?? []
    arr.push(m.text)
    groups.set(m.kind, arr)
  }
  const parts = [`# ${title}`]
  for (const [kind, items] of groups) {
    parts.push(`\n## ${kind.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}`)
    for (const item of items) parts.push(`- ${item}`)
  }
  return parts.join("\n")
}

interface MemoryPanelProps {
  state: MemoryPanelState
  projectEnabled: boolean
  projectSlug?: string | null
  actions: BoundMemoryActions
}

// ---------------------------------------------------------------------------
// Shared memory card — editable on click, with optional accept/deny actions
// ---------------------------------------------------------------------------

type MemoryRecord = ProposedMemory | PreviewMemory

function memoryText(m: MemoryRecord): string {
  return "memory" in m ? m.memory : m.text
}

function LoadingWorkspace({ message }: { message: string }) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="flex h-full flex-col"
      onClick={(e) => e.stopPropagation()}
    >
      <div className="flex items-center border-b border-divider/50 px-4 py-2 shrink-0">
        <Loader2 className="h-3.5 w-3.5 animate-spin text-ink-muted mr-2" />
        <span className="text-xs font-medium text-ink-muted">{message}</span>
      </div>
      <div className="flex-1" />
    </motion.div>
  )
}

export function AddMemoryCard({ scope, disabled, onAddMemory, onCancel }: {
  scope: "global" | "project"
  disabled: boolean
  onAddMemory: (scope: "global" | "project", memory: string) => void
  onCancel: () => void
}) {
  const [value, setValue] = useState("")
  const ScopeIcon = scope === "global" ? Globe : FolderOpen
  const scopeLabel = scope === "global" ? "Global" : "Project"

  function submit() {
    const trimmed = value.trim()
    if (!trimmed || disabled) return
    onAddMemory(scope, trimmed)
    setValue("")
    onCancel()
  }

  return (
    <div className="rounded-lg border border-divider bg-surface p-4">
      <div className="mb-3 flex items-center gap-2">
        <ScopeIcon className="h-3.5 w-3.5 text-ink-muted" />
        <span className="text-xs font-medium text-ink-muted">{scopeLabel}</span>
        <span className="rounded-full bg-surface-elevated px-2 py-0.5 text-[10px] text-ink-subtle">manual</span>
      </div>
      <textarea
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
            e.preventDefault()
            submit()
          }
        }}
        disabled={disabled}
        rows={3}
        autoFocus
        className="w-full resize-none bg-transparent text-sm leading-6 text-ink outline-none placeholder:text-ink-faint disabled:opacity-50"
        placeholder="Write a memory to incorporate..."
      />
      <div className="mt-3 flex justify-end gap-2">
        <button onClick={onCancel} disabled={disabled}
          className="flex items-center gap-1.5 text-xs text-ink-muted hover:text-ink disabled:opacity-40 transition-colors">
          <X className="h-3 w-3" />Deny
        </button>
        <button onClick={submit} disabled={disabled || !value.trim()}
          className="flex items-center gap-1.5 text-xs text-ink-muted hover:text-ink disabled:opacity-40 transition-colors">
          <Check className="h-3 w-3" />Accept
        </button>
      </div>
    </div>
  )
}

function ErrorWorkspace({ error, onClose }: { error: string | null; onClose: () => void }) {
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      className="flex h-full flex-col" onClick={(e) => e.stopPropagation()}>
      <div className="flex items-center justify-between border-b border-divider/50 px-4 py-2 shrink-0">
        <span className="text-xs font-medium text-ink-muted">Memory Management</span>
        <button onClick={onClose}
          className="flex items-center gap-1.5 text-xs text-ink-muted hover:text-ink transition-colors">
          <X className="h-3 w-3" />Close
        </button>
      </div>
      <div className="flex flex-1 items-center justify-center p-8">
        <div className="w-full max-w-lg rounded-xl border border-divider bg-surface p-5">
          <div className="text-sm font-medium text-ink">Memory error</div>
          <div className="mt-2 whitespace-pre-wrap text-xs leading-5 text-ink-muted">
            {error || "Memory operation failed."}
          </div>
        </div>
      </div>
    </motion.div>
  )
}

interface MemoryCardProps {
  memory: MemoryRecord
  scope: "global" | "project"
  disabled?: boolean
  mode?: "review" | "display"
  onChange?: (memoryId: string, text: string) => void
  onAccept?: (memoryId: string) => void
  onCancel?: (memoryId: string) => void
  onRemove?: (memoryId: string) => void
}

export function MemoryCard({
  memory, scope, disabled = false, mode = "review",
  onChange, onAccept, onCancel, onRemove,
}: MemoryCardProps) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState("")
  const ScopeIcon = scope === "global" ? Globe : FolderOpen
  const scopeLabel = scope === "global" ? "Global" : "Project"
  const mKind = "kind" in memory ? (memory.kind || "note") : memory.kind
  const status: MemoryStatus | undefined = "status" in memory ? memory.status : undefined

  const handleEdit = useCallback(() => {
    setDraft(memoryText(memory))
    setEditing(true)
  }, [memory])

  const handleSave = useCallback(() => {
    const trimmed = draft.trim()
    if (trimmed && trimmed !== memoryText(memory)) {
      onChange?.(memory.id, trimmed)
    }
    setEditing(false)
  }, [draft, memory, onChange])

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === "Escape") {
      setEditing(false)
    }
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault()
      handleSave()
    }
  }, [handleSave])

  return (
    <div className="rounded-lg border border-divider bg-surface p-4 group/card">
      <div className="flex items-center gap-2 mb-3">
        <ScopeIcon className="h-3.5 w-3.5 text-ink-muted" />
        <span className="text-xs font-medium text-ink-muted">{scopeLabel}</span>
        <span className="rounded-full bg-surface-elevated px-2 py-0.5 text-[10px] text-ink-subtle">{mKind}</span>
        {mode === "display" && (
          <div className="ml-auto flex items-center gap-1.5">
            {status && (
              <span className={`rounded-full px-2 py-0.5 text-[10px] ${STATUS_STYLES[status]}`}>{status}</span>
            )}
            {onRemove && (
              <button onClick={() => onRemove(memory.id)} disabled={disabled} title="Remove memory"
                className="rounded-md p-0.5 text-ink-subtle hover:bg-hover hover:text-ink disabled:opacity-40 transition-colors">
                <X className="h-3 w-3" />
              </button>
            )}
          </div>
        )}
      </div>

      {editing ? (
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={handleKeyDown}
          onBlur={handleSave}
          rows={Math.max(2, draft.split("\n").length)}
          autoFocus
          disabled={disabled}
          className="w-full resize-none bg-transparent text-sm leading-6 text-ink outline-none placeholder:text-ink-faint disabled:opacity-50"
        />
      ) : (
        <div
          onClick={handleEdit}
          className="cursor-text text-sm leading-6 text-ink whitespace-pre-wrap select-none"
          title="Click to edit"
        >
          {memoryText(memory)}
        </div>
      )}

      {mode === "review" && onAccept && onCancel && (
        <div className="mt-3 flex justify-end gap-2">
          <button onClick={() => onCancel(memory.id)} disabled={disabled}
            className="flex items-center gap-1.5 text-xs text-ink-muted hover:text-ink disabled:opacity-40 transition-colors">
            <X className="h-3 w-3" />Deny
          </button>
          <button onClick={() => onAccept(memory.id)} disabled={disabled}
            className="flex items-center gap-1.5 text-xs text-ink-muted hover:text-ink disabled:opacity-40 transition-colors">
            <Check className="h-3 w-3" />Accept
          </button>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Candidate review — extracted memories, editable + accept/deny
// ---------------------------------------------------------------------------

function CandidateWorkspace({
  globalMemories,
  projectMemories,
  projectEnabled,
  onAcceptRemaining,
  onCancelRemaining,
  onAcceptMemory,
  onCancelMemory,
  onAddMemory,
  onEditMemory,
}: {
  globalMemories: ProposedMemory[]
  projectMemories: ProposedMemory[] | null
  projectEnabled: boolean
  onAcceptRemaining: () => void
  onCancelRemaining: () => void
  onAcceptMemory: (memoryId: string) => void
  onCancelMemory: (memoryId: string) => void
  onAddMemory: (scope: "global" | "project", memory: string) => void
  onEditMemory: (memoryId: string, text: string) => void
}) {
  const total = globalMemories.length + (projectMemories?.length ?? 0)
  const [addingScope, setAddingScope] = useState<"global" | "project" | null>(null)

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      className="h-full flex flex-col" onClick={(e) => e.stopPropagation()}>
      <div className="flex items-center justify-between border-b border-divider/50 px-4 py-2 shrink-0">
        <div className="flex items-center gap-3">
          <span className="text-xs font-medium text-ink-muted">Review memories</span>
          <span className="text-[10px] text-ink-faint">{total} candidate{total === 1 ? "" : "s"}</span>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={onCancelRemaining}
            className="flex items-center gap-1.5 text-xs text-ink-muted hover:text-ink transition-colors">
            <X className="h-3 w-3" />Deny remaining
          </button>
          <button onClick={onAcceptRemaining}
            className="flex items-center gap-1.5 rounded-md border border-divider-strong bg-surface-elevated px-3 py-1.5 text-xs font-medium text-ink hover:bg-hover transition-colors">
            <Check className="h-3 w-3" />Accept remaining
          </button>
        </div>
      </div>

      <div className="flex-1 flex min-h-0">
        <div className="w-1/2 min-w-0 border-r border-divider/50 flex flex-col">
          <div className="flex items-center justify-between border-b border-divider/30 px-3 py-1.5">
            <div className="flex items-center gap-2 text-[10px] font-medium uppercase tracking-wider text-ink-faint">
              <Globe className="h-3 w-3" />Global profile
            </div>
            <button onClick={() => setAddingScope(addingScope === "global" ? null : "global")}
              className="rounded-md p-1 text-ink-subtle hover:bg-hover hover:text-ink transition-colors" title="Add global memory">
              <Plus className="h-3.5 w-3.5" />
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {addingScope === "global" && (
              <AddMemoryCard scope="global" disabled={false} onAddMemory={onAddMemory} onCancel={() => setAddingScope(null)} />
            )}
            {globalMemories.length > 0 ? globalMemories.map((m) => (
              <MemoryCard key={m.id} memory={m} scope="global" mode="review"
                onChange={onEditMemory} onAccept={onAcceptMemory} onCancel={onCancelMemory} />
            )) : <div className="text-xs italic text-ink-faint">No global candidates.</div>}
          </div>
        </div>
        <div className="w-1/2 min-w-0 flex flex-col">
          <div className="flex items-center justify-between border-b border-divider/30 px-3 py-1.5">
            <div className="flex items-center gap-2 text-[10px] font-medium uppercase tracking-wider text-ink-faint">
              <FolderOpen className="h-3 w-3" />Project memory
            </div>
            {projectEnabled && (
              <button onClick={() => setAddingScope(addingScope === "project" ? null : "project")}
                className="rounded-md p-1 text-ink-subtle hover:bg-hover hover:text-ink transition-colors" title="Add project memory">
                <Plus className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {addingScope === "project" && projectEnabled && (
              <AddMemoryCard scope="project" disabled={false} onAddMemory={onAddMemory} onCancel={() => setAddingScope(null)} />
            )}
            {projectMemories && projectMemories.length > 0 ? projectMemories.map((m) => (
              <MemoryCard key={m.id} memory={m} scope="project" mode="review"
                onChange={onEditMemory} onAccept={onAcceptMemory} onCancel={onCancelMemory} />
            )) : <div className="text-xs italic text-ink-faint">No project candidates.</div>}
          </div>
        </div>
      </div>
    </motion.div>
  )
}

// ---------------------------------------------------------------------------
// Doc review — reconciled memory cards + diff
// ---------------------------------------------------------------------------

interface PreviewTab {
  scope: "global" | "project"
  icon: typeof Globe
  title: string
  existing_markdown: string
  revised_markdown: string
  existing_memories: PreviewMemory[]
  revised_memories: PreviewMemory[]
  loading: boolean
}

function DocumentWorkspace({
  previews,
  onCommitDocuments,
  onCancelRemaining,
  onEditMemory,
  onRemoveMemory,
}: {
  previews: PreviewTab[]
  onCommitDocuments: () => Promise<void>
  onCancelRemaining: () => void
  onEditMemory: (scope: "global" | "project", memoryId: string, text: string) => void
  onRemoveMemory: (scope: "global" | "project", memoryId: string) => void
}) {
  const [activeIdx, setActiveIdx] = useState(0)
  const [processing, setProcessing] = useState(false)
  const active = previews[activeIdx] ?? previews[0]

  const handleCommit = useCallback(async () => {
    setProcessing(true)
    try { await onCommitDocuments() } finally { setProcessing(false) }
  }, [onCommitDocuments])

  if (!active) return null

  const anyLoading = previews.some((p) => p.loading)
  const ActiveIcon = active.icon
  const memories = active.revised_memories
  const revisedMarkdown = renderMemoriesMarkdown(memories, active.title).trimEnd() + "\n"
  const diff = active.loading ? "" : computeLineDiff(active.existing_markdown || "", revisedMarkdown)

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      className="h-full flex flex-col" onClick={(e) => e.stopPropagation()}>
      <div className="flex items-center justify-between border-b border-divider/50 px-4 py-2 shrink-0">
        <div className="flex items-center gap-3">
          {(processing || anyLoading) && <Loader2 className="h-3.5 w-3.5 animate-spin text-ink-muted" />}
          <span className="text-xs font-medium text-ink-muted">{anyLoading ? "Preparing changes" : "Review changes"}</span>
          {previews.length > 1 && (
            <div className="flex items-center gap-1 rounded-lg border border-divider bg-surface p-1">
              {previews.map((p, i) => {
                const Icon = p.icon
                return (
                  <button key={p.scope} onClick={() => setActiveIdx(i)}
                    className={`flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs transition-colors ${i === activeIdx ? "bg-surface-elevated text-ink" : "text-ink-muted hover:text-ink hover:bg-hover"}`}>
                    <Icon className="h-3.5 w-3.5" />{p.title}
                  </button>
                )
              })}
            </div>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button onClick={onCancelRemaining} disabled={processing}
            className="flex items-center gap-1.5 text-xs text-ink-muted hover:text-ink disabled:opacity-40 transition-colors">
            <X className="h-3 w-3" />Cancel
          </button>
          <button onClick={handleCommit} disabled={processing || anyLoading}
            className="flex items-center gap-1.5 rounded-md border border-divider-strong bg-surface-elevated px-3 py-1.5 text-xs font-medium text-ink hover:bg-hover disabled:opacity-40 transition-colors">
            {processing ? <Loader2 className="h-3 w-3 animate-spin" /> : <Check className="h-3 w-3" />}
            Commit
          </button>
        </div>
      </div>

      {active.loading ? (
        <div className="flex flex-1 items-center justify-center gap-2 min-h-0">
          <Loader2 className="h-4 w-4 animate-spin text-ink-muted" />
          <span className="text-xs text-ink-muted">Reconciling memories…</span>
        </div>
      ) : (
        <div className="flex-1 flex min-h-0">
          <div className="w-1/2 min-w-0 border-r border-divider/50 flex flex-col">
            <div className="flex items-center justify-between border-b border-divider/30 px-3 py-1.5">
              <div className="flex items-center gap-2 text-[10px] font-medium uppercase tracking-wider text-ink-faint">
                <ActiveIcon className="h-3 w-3" />Merged memories
              </div>
              <span className="text-[10px] text-ink-faint">{memories.length} entries</span>
            </div>
            <div className="flex-1 overflow-y-auto p-4 space-y-3">
              {memories.length > 0 ? memories.map((m) => (
                <MemoryCard key={m.id} memory={m} scope={active.scope} mode="display"
                  onChange={(id, text) => onEditMemory(active.scope, id, text)}
                  onRemove={(id) => onRemoveMemory(active.scope, id)} />
              )) : <div className="text-xs italic text-ink-faint">No memories in this profile.</div>}
            </div>
          </div>
          <div className="w-1/2 min-w-0 flex flex-col">
            <div className="flex items-center gap-2 border-b border-divider/30 px-3 py-1.5 text-[10px] font-medium uppercase tracking-wider text-ink-faint">
              <GitCompare className="h-3 w-3" />Diff
            </div>
            <div className="flex-1 overflow-auto p-4">
              {diff ? (
                <LaTeXMarkdown content={"```diff\n" + diff + "\n```"} compact />
              ) : (
                <div className="text-xs italic text-ink-faint">No changes.</div>
              )}
            </div>
          </div>
        </div>
      )}
    </motion.div>
  )
}

// ---------------------------------------------------------------------------
// Main MemoryPanel
// ---------------------------------------------------------------------------

export function MemoryPanel({
  state, projectEnabled, projectSlug, actions,
}: MemoryPanelProps) {
  const extractingMode = state.phase === "extracting"
  const composingMode = state.phase === "composing"
  const errorMode = state.phase === "error"
  const candidatesMode = state.phase === "review"
  const documentsMode = state.phase === "doc-review"
  const locked = composingMode || extractingMode
  const globalMemories = candidatesMode ? state.globalMemories : []
  const projectMemories = candidatesMode ? state.projectMemories : null
  const error = errorMode ? state.error : null

  const previews: PreviewTab[] = []
  if (state.phase === "doc-review") {
    const tabDefs = [
      { scope: "global" as const, icon: Globe, title: "Global Profile", preview: state.globalPreview },
      { scope: "project" as const, icon: FolderOpen, title: "Project Memory", preview: state.projectPreview },
    ]
    for (const t of tabDefs) {
      if (t.preview) {
        previews.push({ scope: t.scope, icon: t.icon, title: t.title, ...t.preview, loading: false })
      } else if (state.loadingScopes.includes(t.scope)) {
        previews.push({
          scope: t.scope, icon: t.icon, title: t.title,
          existing_markdown: "", revised_markdown: "",
          existing_memories: [], revised_memories: [], loading: true,
        })
      }
    }
  }

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const target = e.target as HTMLElement | null
      const editable = !!target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)
      if (editable) return

      if (e.key === "Escape") {
        e.preventDefault()
        actions.cancelRemaining()
        return
      }
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault()
        if (documentsMode) actions.commitDocuments()
        else actions.acceptRemaining()
        return
      }
      if (candidatesMode) {
        const nextId = (globalMemories[0] ?? projectMemories?.[0])?.id
        if (!nextId) return
        if (e.key === "+" || e.key === "=") {
          e.preventDefault()
          actions.acceptMemory(nextId)
        } else if (e.key === "-") {
          e.preventDefault()
          actions.cancelMemory(nextId)
        }
      }
    }
    document.addEventListener("keydown", onKey)
    return () => document.removeEventListener("keydown", onKey)
  }, [actions, documentsMode, candidatesMode, globalMemories, projectMemories])

  return (
    <AnimatePresence>
      <FloatingWindow
        onClose={locked ? undefined : actions.cancelRemaining}
        overlayClassName="z-[80]"
        panelClassName={(extractingMode || composingMode || errorMode) ? "h-[420px] max-w-2xl" : "h-[86vh] max-h-[840px]"}
      >
        {(extractingMode || composingMode) ? (
          <LoadingWorkspace message={extractingMode ? "Reading conversation context" : "Preparing diff"} />
        ) : errorMode ? (
          <ErrorWorkspace error={error} onClose={actions.cancelRemaining} />
        ) : documentsMode ? (
          <DocumentWorkspace
            previews={previews}
            onCommitDocuments={actions.commitDocuments}
            onCancelRemaining={actions.cancelRemaining}
            onEditMemory={actions.editRevisedMemory}
            onRemoveMemory={actions.removeRevisedMemory}
          />
        ) : (
          <CandidateWorkspace
            globalMemories={globalMemories}
            projectMemories={projectMemories}
            projectEnabled={projectEnabled}
            onAcceptRemaining={actions.acceptRemaining}
            onCancelRemaining={actions.cancelRemaining}
            onAcceptMemory={actions.acceptMemory}
            onCancelMemory={actions.cancelMemory}
            onAddMemory={actions.addMemory}
            onEditMemory={actions.editMemory}
          />
        )}
      </FloatingWindow>
    </AnimatePresence>
  )
}
