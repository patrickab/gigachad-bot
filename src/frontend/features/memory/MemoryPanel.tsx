"use client"

import { useCallback, useEffect, useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Check, FolderOpen, GitCompare, Globe, Loader2, Plus, X } from "lucide-react"
import { FloatingWindow } from "@/components/ui/FloatingWindow"
import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"
import { AddMemoryCard, MemoryErrorCard } from "@/features/memory/MemoryCard"
import { MemoryBoard } from "@/features/memory/MemoryBoard"
import { computeLineDiff } from "@/lib/diff"
import type { BoundMemoryActions, MemoryPanelState } from "@/features/memory/useMemorizeFlow"
import type { CategoryDef, PreviewMemory, ProposedMemory } from "@/lib/domain/types"
import { groupByCategoryOrder, renderMemoriesMarkdown } from "@/features/memory/memoryUtils"

interface MemoryPanelProps {
  state: MemoryPanelState
  projectEnabled: boolean
  actions: BoundMemoryActions
  globalCategories: CategoryDef[]
  projectCategories: CategoryDef[]
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
      <MemoryErrorCard message={error || "Memory operation failed."} />
    </motion.div>
  )
}

// Candidate review: extracted memories, editable + accept/deny
function CandidateWorkspace({
  globalMemories,
  projectMemories,
  projectEnabled,
  globalCategories,
  projectCategories,
  onAcceptRemaining,
  onCancelRemaining,
  onAcceptMemory,
  onCancelMemory,
  onAddMemory,
  onEditMemory,
  onEditCategory,
}: {
  globalMemories: ProposedMemory[]
  projectMemories: ProposedMemory[] | null
  projectEnabled: boolean
  globalCategories: CategoryDef[]
  projectCategories: CategoryDef[]
  onAcceptRemaining: () => void
  onCancelRemaining: () => void
  onAcceptMemory: (memoryId: string) => void
  onCancelMemory: (memoryId: string) => void
  onAddMemory: (scope: "global" | "project", memory: string) => void
  onEditMemory: (memoryId: string, text: string) => void
  onEditCategory: (memoryId: string, category: string) => void
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
            <button aria-label="Add global memory" onClick={() => setAddingScope(addingScope === "global" ? null : "global")}
              className="rounded-md p-1 text-ink-subtle hover:bg-hover hover:text-ink transition-colors">
              <Plus className="h-3.5 w-3.5" />
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {addingScope === "global" && (
              <AddMemoryCard scope="global" onAddMemory={onAddMemory} onCancel={() => setAddingScope(null)} />
            )}
            {globalMemories.length > 0 || globalCategories.length > 0 ? (
              <MemoryBoard
                memories={globalMemories}
                categoryOrder={globalCategories}
                mode="review"
                onChange={onEditMemory}
                onChangeCategory={onEditCategory}
                onAccept={onAcceptMemory}
                onCancel={onCancelMemory}
              />
            ) : (
              <div className="text-xs italic text-ink-faint">No global candidates.</div>
            )}
          </div>
        </div>
        <div className="w-1/2 min-w-0 flex flex-col">
          <div className="flex items-center justify-between border-b border-divider/30 px-3 py-1.5">
            <div className="flex items-center gap-2 text-[10px] font-medium uppercase tracking-wider text-ink-faint">
              <FolderOpen className="h-3 w-3" />Project memory
            </div>
            {projectEnabled && (
              <button aria-label="Add project memory" onClick={() => setAddingScope(addingScope === "project" ? null : "project")}
                className="rounded-md p-1 text-ink-subtle hover:bg-hover hover:text-ink transition-colors">
                <Plus className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {addingScope === "project" && projectEnabled && (
              <AddMemoryCard scope="project" onAddMemory={onAddMemory} onCancel={() => setAddingScope(null)} />
            )}
            {projectEnabled && (projectMemories && projectMemories.length > 0 || projectCategories.length > 0) ? (
              <MemoryBoard
                memories={projectMemories ?? []}
                categoryOrder={projectCategories}
                mode="review"
                onChange={onEditMemory}
                onChangeCategory={onEditCategory}
                onAccept={onAcceptMemory}
                onCancel={onCancelMemory}
              />
            ) : (
              <div className="text-xs italic text-ink-faint">No project candidates.</div>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  )
}

// Doc review: reconciled memory cards + diff
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
  globalCategories,
  projectCategories,
  onCommitDocuments,
  onCancelRemaining,
  onEditMemory,
  onEditCategory,
  onRemoveMemory,
}: {
  previews: PreviewTab[]
  globalCategories: CategoryDef[]
  projectCategories: CategoryDef[]
  onCommitDocuments: () => Promise<void>
  onCancelRemaining: () => void
  onEditMemory: (scope: "global" | "project", memoryId: string, text: string) => void
  onEditCategory: (scope: "global" | "project", memoryId: string, category: string) => void
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
  const activeCategories = active.scope === "global" ? globalCategories : projectCategories
  const revisedMarkdown = renderMemoriesMarkdown(memories, active.title, activeCategories).trimEnd() + "\n"
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
            <div className="flex-1 overflow-y-auto p-4">
              <MemoryBoard
                memories={memories}
                categoryOrder={activeCategories}
                mode="display"
                onChange={(id, text) => onEditMemory(active.scope, id, text)}
                onChangeCategory={(id, cat) => onEditCategory(active.scope, id, cat)}
                onRemove={(id) => onRemoveMemory(active.scope, id)}
              />
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

const NO_MEMORIES: ProposedMemory[] = []

// Main MemoryPanel
export function MemoryPanel({
  state, projectEnabled, actions, globalCategories, projectCategories,
}: MemoryPanelProps) {
  const extractingMode = state.phase === "extracting"
  const errorMode = state.phase === "error"
  const candidatesMode = state.phase === "review"
  const documentsMode = state.phase === "doc-review"
  const globalMemories = candidatesMode ? state.globalMemories : NO_MEMORIES
  const projectMemories = candidatesMode ? state.projectMemories : null
  const error = errorMode ? state.error : null

  // Global tab first, then project. A scope still reconciling gets a loading placeholder tab.
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

      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault()
        if (documentsMode) actions.commitDocuments()
        else actions.acceptRemaining()
        return
      }
      // +/- act on the first candidate as displayed (global before project)
      if (candidatesMode) {
        const globalSections = groupByCategoryOrder(globalCategories, globalMemories)
        const projectSections = groupByCategoryOrder(projectCategories, projectMemories ?? [])
        const firstGlobal = globalSections.find(s => s.items.length > 0)?.items[0]
        const firstProject = projectSections.find(s => s.items.length > 0)?.items[0]
        const nextId = (firstGlobal ?? firstProject)?.id
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
  }, [actions, documentsMode, candidatesMode, globalCategories, projectCategories, globalMemories, projectMemories])

  return (
    <AnimatePresence>
      <FloatingWindow
        onClose={extractingMode ? undefined : actions.cancelRemaining}
        overlayClassName="z-[80]"
        panelClassName={(extractingMode || errorMode) ? "h-[420px] max-w-2xl" : "h-[86vh] max-h-[840px]"}
      >
        {extractingMode ? (
          <LoadingWorkspace message="Reading conversation context" />
        ) : errorMode ? (
          <ErrorWorkspace error={error} onClose={actions.cancelRemaining} />
        ) : documentsMode ? (
          <DocumentWorkspace
            previews={previews}
            globalCategories={globalCategories}
            projectCategories={projectEnabled ? projectCategories : []}
            onCommitDocuments={actions.commitDocuments}
            onCancelRemaining={actions.cancelRemaining}
            onEditMemory={actions.editRevisedMemory}
            onEditCategory={actions.editRevisedMemoryCategory}
            onRemoveMemory={actions.removeRevisedMemory}
          />
        ) : (
          <CandidateWorkspace
            globalMemories={globalMemories}
            projectMemories={projectMemories}
            projectEnabled={projectEnabled}
            globalCategories={globalCategories}
            projectCategories={projectEnabled ? projectCategories : []}
            onAcceptRemaining={actions.acceptRemaining}
            onCancelRemaining={actions.cancelRemaining}
            onAcceptMemory={actions.acceptMemory}
            onCancelMemory={actions.cancelMemory}
            onAddMemory={actions.addMemory}
            onEditMemory={actions.editMemory}
            onEditCategory={actions.editMemoryCategory}
          />
        )}
      </FloatingWindow>
    </AnimatePresence>
  )
}
