"use client"

import { useEffect, useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Check, FolderOpen, GitCompare, Globe, Loader2, Plus, X } from "lucide-react"
import { ConsoleEditor } from "@/components/ConsoleEditor"
import { FloatingWindow } from "@/components/FloatingWindow"
import { LaTeXMarkdown } from "@/components/LaTeXMarkdown"
import { computeLineDiff } from "@/lib/diff"
import type { BoundMemoryActions, MemoryPanelState } from "@/hooks/useCommandBar"
import type { MemoryProfileMeta, ProposedMemory } from "@/lib/types"

interface MemoryPanelProps {
  state: MemoryPanelState
  projectEnabled: boolean
  projectSlug?: string | null
  actions: BoundMemoryActions
}

function LoadingWorkspace() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="flex h-full flex-col"
      onClick={(e) => e.stopPropagation()}
    >
      <div className="flex items-center justify-between border-b border-divider/50 px-4 py-2 shrink-0">
        <div className="flex items-center gap-3">
          <span className="text-xs font-medium text-ink-muted">Memory Management</span>
        </div>
      </div>
      <div className="flex flex-1 items-center justify-center p-8">
        <div className="max-w-md text-center">
          <Loader2 className="mx-auto h-6 w-6 animate-spin text-ink-muted" />
          <div className="mt-4 text-sm font-medium text-ink">Reading conversation context</div>
          <div className="mt-2 text-xs leading-5 text-ink-muted">
            Candidate memories will appear here for accept/deny review before anything is committed.
          </div>
        </div>
      </div>
    </motion.div>
  )
}

function AddMemoryCard({ scope, disabled, onAddMemory, onCancel }: {
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
        <button
          onClick={onCancel}
          disabled={disabled}
          className="flex items-center gap-1.5 text-xs text-ink-muted hover:text-ink disabled:opacity-40 disabled:hover:text-ink-muted transition-colors"
        >
          <X className="h-3 w-3" />
          Deny
        </button>
        <button
          onClick={submit}
          disabled={disabled || !value.trim()}
          className="flex items-center gap-1.5 text-xs text-ink-muted hover:text-ink disabled:opacity-40 disabled:hover:text-ink-muted transition-colors"
        >
          <Check className="h-3 w-3" />
          Accept
        </button>
      </div>
    </div>
  )
}

function ErrorWorkspace({ error, onClose }: { error: string | null; onClose: () => void }) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="flex h-full flex-col"
      onClick={(e) => e.stopPropagation()}
    >
      <div className="flex items-center justify-between border-b border-divider/50 px-4 py-2 shrink-0">
        <span className="text-xs font-medium text-ink-muted">Memory Management</span>
        <button
          onClick={onClose}
          className="flex items-center gap-1.5 text-xs text-ink-muted hover:text-ink transition-colors"
        >
          <X className="h-3 w-3" />
          Close
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

function MemoryCard({ memory, scope, disabled, onAccept, onCancel }: {
  memory: ProposedMemory
  scope: "global" | "project"
  disabled: boolean
  onAccept: (memoryId: string) => void
  onCancel: (memoryId: string) => void
}) {
  const ScopeIcon = scope === "global" ? Globe : FolderOpen
  const scopeLabel = scope === "global" ? "Global" : "Project"

  return (
    <div className="rounded-lg border border-divider bg-surface p-4">
      <div className="flex items-center gap-2 mb-3">
        <ScopeIcon className="h-3.5 w-3.5 text-ink-muted" />
        <span className="text-xs font-medium text-ink-muted">{scopeLabel}</span>
        {memory.kind && <span className="rounded-full bg-surface-elevated px-2 py-0.5 text-[10px] text-ink-subtle">{memory.kind}</span>}
      </div>
      <div className="text-sm text-ink">
        <LaTeXMarkdown content={memory.memory} compact />
      </div>
      <div className="mt-3 flex justify-end gap-2">
        <button
          onClick={() => onCancel(memory.id)}
          disabled={disabled}
          className="flex items-center gap-1.5 text-xs text-ink-muted hover:text-ink disabled:opacity-40 disabled:hover:text-ink-muted transition-colors"
        >
          <X className="h-3 w-3" />
          Deny
        </button>
        <button
          onClick={() => onAccept(memory.id)}
          disabled={disabled}
          className="flex items-center gap-1.5 text-xs text-ink-muted hover:text-ink disabled:opacity-40 disabled:hover:text-ink-muted transition-colors"
        >
          <Check className="h-3 w-3" />
          Accept
        </button>
      </div>
    </div>
  )
}

function CandidateWorkspace({
  globalMemories,
  projectMemories,
  isComposing,
  projectEnabled,
  onAcceptRemaining,
  onCancelRemaining,
  onAcceptMemory,
  onCancelMemory,
  onAddMemory,
}: {
  globalMemories: ProposedMemory[]
  projectMemories: ProposedMemory[] | null
  isComposing: boolean
  projectEnabled: boolean
  onAcceptRemaining: () => void
  onCancelRemaining: () => void
  onAcceptMemory: (memoryId: string) => void
  onCancelMemory: (memoryId: string) => void
  onAddMemory: (scope: "global" | "project", memory: string) => void
}) {
  const total = globalMemories.length + (projectMemories?.length ?? 0)
  const [addingScope, setAddingScope] = useState<"global" | "project" | null>(null)

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="h-full flex flex-col"
      onClick={(e) => e.stopPropagation()}
    >
      <div className="flex items-center justify-between border-b border-divider/50 px-4 py-2 shrink-0">
        <div className="flex items-center gap-3">
          <span className="text-xs font-medium text-ink-muted">Review memories</span>
          {isComposing && (
            <span className="flex items-center gap-1.5 text-[10px] text-ink-muted">
              <Loader2 className="h-3 w-3 animate-spin" />
              Updating memory docs
            </span>
          )}
          <span className="text-[10px] text-ink-faint">{total} candidate{total === 1 ? "" : "s"}</span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={onCancelRemaining}
            disabled={isComposing}
            className="flex items-center gap-1.5 text-xs text-ink-muted hover:text-ink disabled:opacity-40 disabled:hover:text-ink-muted transition-colors"
          >
            <X className="h-3 w-3" />
            Deny remaining
          </button>
          <button
            onClick={onAcceptRemaining}
            disabled={isComposing}
            className="flex items-center gap-1.5 rounded-md border border-divider-strong bg-surface-elevated px-3 py-1.5 text-xs font-medium text-ink hover:bg-hover disabled:opacity-50 disabled:hover:bg-surface-elevated transition-colors"
          >
            <Check className="h-3 w-3" />
            Accept remaining
          </button>
        </div>
      </div>

      <div className="flex-1 flex min-h-0">
        <div className="w-1/2 min-w-0 border-r border-divider/50 flex flex-col">
          <div className="flex items-center justify-between border-b border-divider/30 px-3 py-1.5">
            <div className="flex items-center gap-2 text-[10px] font-medium uppercase tracking-wider text-ink-faint">
              <Globe className="h-3 w-3" />
              Global profile candidates
            </div>
            <button
              onClick={() => setAddingScope(addingScope === "global" ? null : "global")}
              disabled={isComposing}
              className="rounded-md p-1 text-ink-subtle hover:bg-hover hover:text-ink disabled:opacity-40 disabled:hover:bg-transparent disabled:hover:text-ink-subtle transition-colors"
              title="Add global memory"
            >
              <Plus className="h-3.5 w-3.5" />
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {addingScope === "global" && (
              <AddMemoryCard scope="global" disabled={isComposing} onAddMemory={onAddMemory} onCancel={() => setAddingScope(null)} />
            )}
            {globalMemories.length > 0 ? globalMemories.map((m) => (
              <MemoryCard key={m.id} memory={m} scope="global" disabled={isComposing} onAccept={onAcceptMemory} onCancel={onCancelMemory} />
            )) : <div className="text-xs italic text-ink-faint">No global candidates.</div>}
          </div>
        </div>
        <div className="w-1/2 min-w-0 flex flex-col">
          <div className="flex items-center justify-between border-b border-divider/30 px-3 py-1.5">
            <div className="flex items-center gap-2 text-[10px] font-medium uppercase tracking-wider text-ink-faint">
              <FolderOpen className="h-3 w-3" />
              Project memory candidates
            </div>
            {projectEnabled && (
              <button
                onClick={() => setAddingScope(addingScope === "project" ? null : "project")}
                disabled={isComposing}
                className="rounded-md p-1 text-ink-subtle hover:bg-hover hover:text-ink disabled:opacity-40 disabled:hover:bg-transparent disabled:hover:text-ink-subtle transition-colors"
                title="Add project memory"
              >
                <Plus className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {addingScope === "project" && projectEnabled && (
              <AddMemoryCard scope="project" disabled={isComposing} onAddMemory={onAddMemory} onCancel={() => setAddingScope(null)} />
            )}
            {!projectEnabled ? <div className="text-xs italic text-ink-faint">Open a project to add project memories.</div> : projectMemories && projectMemories.length > 0 ? projectMemories.map((m) => (
              <MemoryCard key={m.id} memory={m} scope="project" disabled={isComposing} onAccept={onAcceptMemory} onCancel={onCancelMemory} />
            )) : <div className="text-xs italic text-ink-faint">No project candidates.</div>}
          </div>
        </div>
      </div>
    </motion.div>
  )
}

type DocumentId = "global" | "project"

interface MemoryDocument {
  id: DocumentId
  title: string
  subtitle: string
  icon: typeof Globe
  diff: string | null
  value: string
  onChange: (value: string) => void
}

function DocumentWorkspace({
  documents,
  acceptedCount,
  onCommitDocuments,
  onCancelRemaining,
  projectSlug,
  actions,
}: {
  documents: MemoryDocument[]
  acceptedCount: number
  onCommitDocuments: () => void
  onCancelRemaining: () => void
  projectSlug?: string | null
  actions: BoundMemoryActions
}) {
  const [activeId, setActiveId] = useState<DocumentId>(documents[0]?.id ?? "global")
  const active = documents.find((doc) => doc.id === activeId) ?? documents[0]
  const [profiles, setProfiles] = useState<MemoryProfileMeta[]>([])

  useEffect(() => {
    import("@/lib/api").then(({ listMemoryProfiles }) => {
      listMemoryProfiles(projectSlug).then((res) => {
        setProfiles(res.profiles)
      }).catch(() => {})
    })
  }, [projectSlug, activeId])

  const activeScopeProfiles = profiles.filter((p) => {
    if (activeId === "global") {
      return p.filepath.startsWith("memory/global-profile")
    } else {
      return projectSlug ? p.filepath.startsWith(`${projectSlug}/memory/memory`) : false
    }
  })

  if (!active) return null
  const ActiveIcon = active.icon

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="h-full flex flex-col"
      onClick={(e) => e.stopPropagation()}
    >
      <div className="flex items-center justify-between border-b border-divider/50 px-4 py-2 shrink-0">
        <div className="flex items-center gap-3">
          <span className="text-xs font-medium text-ink-muted">Memory Documents</span>
          {acceptedCount > 0 && (
            <span className="text-[10px] text-ink-faint">{acceptedCount} accepted memor{acceptedCount === 1 ? "y" : "ies"}</span>
          )}
          <div className="flex items-center gap-1 rounded-lg border border-divider bg-surface p-1">
            {documents.map((doc) => {
              const Icon = doc.icon
              const activeDoc = doc.id === active.id
              return (
                <button
                  key={doc.id}
                  onClick={() => setActiveId(doc.id)}
                  className={`flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs transition-colors ${activeDoc ? "bg-surface-elevated text-ink" : "text-ink-muted hover:text-ink hover:bg-hover"}`}
                >
                  <Icon className="h-3.5 w-3.5" />
                  {doc.title}
                </button>
              )
            })}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={onCancelRemaining}
            className="flex items-center gap-1.5 text-xs text-ink-muted hover:text-ink transition-colors"
          >
            <X className="h-3 w-3" />
            Cancel
          </button>
          <button
            onClick={onCommitDocuments}
            className="flex items-center gap-1.5 rounded-md border border-divider-strong bg-surface-elevated px-3 py-1.5 text-xs font-medium text-ink hover:bg-hover transition-colors"
          >
            <Check className="h-3 w-3" />
            Commit documents
          </button>
        </div>
      </div>

      <div className="flex-1 flex min-h-0">
        <div className="w-1/2 min-w-0 border-r border-divider/50 flex flex-col">
          <div className="flex items-center justify-between border-b border-divider/30 px-3 py-1.5 min-h-[37px]">
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1.5 text-[10px] font-medium uppercase tracking-wider text-ink-faint">
                <ActiveIcon className="h-3 w-3" />
                Edit
              </div>
              {activeScopeProfiles.length > 0 && (
                <select
                  value={activeScopeProfiles.find(p => p.filepath.endsWith(activeId === "global" ? "global-profile.md" : "memory.md"))?.filepath || activeScopeProfiles[0]?.filepath}
                  onChange={(e) => actions.loadProfile(e.target.value, activeId)}
                  className="bg-transparent border border-divider/50 rounded px-1.5 py-0.5 text-[10px] text-ink outline-none"
                >
                  {activeScopeProfiles.map((p) => (
                    <option key={p.filepath} value={p.filepath} className="bg-paper text-ink">
                      {p.title}
                    </option>
                  ))}
                </select>
              )}
            </div>
            <span className="text-[10px] text-ink-faint">{active.subtitle}</span>
          </div>
          <ConsoleEditor
            value={active.value}
            onChange={active.onChange}
            language="markdown"
            placeholder="Edit memory document..."
          />
        </div>
        <div className="w-1/2 min-w-0 flex flex-col">
          <div className="flex items-center gap-2 border-b border-divider/30 px-3 py-1.5 text-[10px] font-medium uppercase tracking-wider text-ink-faint">
            <GitCompare className="h-3 w-3" />
            Proposed diff
          </div>
          <div className="flex-1 overflow-auto p-4">
            {active.diff ? (
              <div className="rounded-lg border border-divider bg-surface/50 p-4 text-xs">
                <LaTeXMarkdown content={`\`\`\`diff\n${active.diff}\n\`\`\``} compact />
              </div>
            ) : (
              <div className="text-xs italic text-ink-faint">No generated diff for this document.</div>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  )
}

export function MemoryPanel({
  state,
  projectEnabled,
  projectSlug,
  actions,
}: MemoryPanelProps) {
  const documentsMode = state.phase === "doc-review"
  const extractingMode = state.phase === "extracting"
  const errorMode = state.phase === "error"
  const candidatesMode = state.phase === "review" || state.phase === "composing"
  const isComposing = state.phase === "composing"
  const locked = isComposing || extractingMode
  const acceptedCount = candidatesMode || documentsMode ? state.acceptedCount : 0
  const globalMemories = candidatesMode ? state.globalMemories : []
  const projectMemories = candidatesMode ? state.projectMemories : null
  const globalDocument = documentsMode ? state.globalDocument : null
  const projectDocument = documentsMode ? state.projectDocument : null
  const globalDiff = documentsMode && state.globalDocument && state.originalGlobalDocument
    ? computeLineDiff(state.originalGlobalDocument, state.globalDocument)
    : null
  const projectDiff = documentsMode && state.projectDocument && state.originalProjectDocument
    ? computeLineDiff(state.originalProjectDocument, state.projectDocument)
    : null
  const error = errorMode ? state.error : null
  const documents: MemoryDocument[] = [
    ...(globalDocument !== null ? [{
      id: "global" as const,
      title: "Global profile",
      subtitle: "Cross-project user context",
      icon: Globe,
      diff: globalDiff,
      value: globalDocument,
      onChange: actions.setGlobalDocument,
    }] : []),
    ...(projectDocument !== null ? [{
      id: "project" as const,
      title: "Project memory",
      subtitle: "Active project context",
      icon: FolderOpen,
      diff: projectDiff,
      value: projectDocument,
      onChange: actions.setProjectDocument,
    }] : []),
  ]

  useEffect(() => {
    function handler(e: KeyboardEvent) {
      if (e.key === "Escape") {
        if (locked) return
        e.preventDefault()
        actions.cancelRemaining()
      }
      if (e.key === "Enter" && !documentsMode && !locked && !errorMode) {
        e.preventDefault()
        actions.acceptRemaining()
      }
    }
    document.addEventListener("keydown", handler)
    return () => document.removeEventListener("keydown", handler)
  }, [actions, documentsMode, errorMode, locked])

  return (
    <AnimatePresence>
      <FloatingWindow
        onClose={locked ? undefined : actions.cancelRemaining}
        overlayClassName="z-[80]"
        panelClassName={extractingMode || errorMode ? "h-[420px] max-w-2xl" : "h-[86vh] max-h-[840px]"}
      >
        {extractingMode ? (
          <LoadingWorkspace />
        ) : errorMode ? (
          <ErrorWorkspace error={error} onClose={actions.cancelRemaining} />
        ) : documentsMode ? (
          <DocumentWorkspace
            documents={documents}
            acceptedCount={acceptedCount}
            onCommitDocuments={actions.commitDocuments}
            onCancelRemaining={actions.cancelRemaining}
            projectSlug={projectSlug}
            actions={actions}
          />
        ) : (
          <CandidateWorkspace
            globalMemories={globalMemories}
            projectMemories={projectMemories}
            isComposing={isComposing}
            projectEnabled={projectEnabled}
            onAcceptRemaining={actions.acceptRemaining}
            onCancelRemaining={actions.cancelRemaining}
            onAcceptMemory={actions.acceptMemory}
            onCancelMemory={actions.cancelMemory}
            onAddMemory={actions.addMemory}
          />
        )}
      </FloatingWindow>
    </AnimatePresence>
  )
}
