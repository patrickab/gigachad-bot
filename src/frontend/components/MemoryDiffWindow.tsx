"use client"

import { useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Check, X, Globe, FolderOpen } from "lucide-react"
import { LaTeXMarkdown } from "@/components/LaTeXMarkdown"
import type { ProposedMemory } from "@/lib/types"

interface MemoryDiffWindowProps {
  globalMemories: ProposedMemory[]
  projectMemories: ProposedMemory[] | null
  onAcceptRemaining: () => void
  onCancelRemaining: () => void
  onAcceptMemory: (memoryId: string) => void
  onCancelMemory: (memoryId: string) => void
}

function MemoryCard({ memory, scope, onAccept, onCancel }: {
  memory: ProposedMemory
  scope: "global" | "project"
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
        {memory.categories && memory.categories.length > 0 && (
          <div className="flex gap-1 ml-auto">
            {memory.categories.map((cat) => (
              <span
                key={cat}
                className="rounded-full bg-surface-elevated px-2 py-0.5 text-[10px] text-ink-subtle"
              >
                {cat}
              </span>
            ))}
          </div>
        )}
      </div>
      <div className="text-sm text-ink">
        <LaTeXMarkdown content={memory.memory} compact />
      </div>
      <div className="mt-3 flex justify-end gap-2">
        <button
          onClick={() => onCancel(memory.id)}
          className="flex items-center gap-1.5 text-xs text-ink-muted hover:text-ink transition-colors"
        >
          <X className="h-3 w-3" />
          Cancel
        </button>
        <button
          onClick={() => onAccept(memory.id)}
          className="flex items-center gap-1.5 text-xs text-ink-muted hover:text-ink transition-colors"
        >
          <Check className="h-3 w-3" />
          Accept
        </button>
      </div>
    </div>
  )
}

export function MemoryDiffWindow({
  globalMemories,
  projectMemories,
  onAcceptRemaining,
  onCancelRemaining,
  onAcceptMemory,
  onCancelMemory,
}: MemoryDiffWindowProps) {
  const total = globalMemories.length + (projectMemories?.length ?? 0)

  useEffect(() => {
    function handler(e: KeyboardEvent) {
      if (e.key === "Escape") {
        e.preventDefault()
        onCancelRemaining()
      }
      if (e.key === "Enter") {
        e.preventDefault()
        onAcceptRemaining()
      }
    }
    document.addEventListener("keydown", handler)
    return () => document.removeEventListener("keydown", handler)
  }, [onAcceptRemaining, onCancelRemaining])

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.15 }}
        className="absolute inset-0 z-[80] flex items-center justify-center"
        onClick={onCancelRemaining}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.96, y: 8 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.96, y: 8 }}
          transition={{ duration: 0.2, ease: "easeOut" }}
          onClick={(e) => e.stopPropagation()}
          className="w-full max-w-xl max-h-[70vh] mx-4 flex flex-col rounded-xl border border-divider bg-paper shadow-[var(--shadow-xl)] overflow-hidden"
        >
          <div className="shrink-0 px-5 pt-5 pb-3 flex items-center justify-between">
            <div className="text-sm font-medium text-ink">
              {total} memor{total !== 1 ? "ies" : "y"} proposed
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={onCancelRemaining}
                className="flex items-center gap-1.5 text-xs text-ink-muted hover:text-ink transition-colors"
              >
                <X className="h-3 w-3" />
                Cancel remaining
              </button>
              <button
                onClick={onAcceptRemaining}
                className="flex items-center gap-1.5 text-xs text-ink-muted hover:text-ink transition-colors"
              >
                <Check className="h-3 w-3" />
                Accept remaining
              </button>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto px-5 pb-5 space-y-3">
            {globalMemories.length > 0 && (
              <div className="space-y-3">
                {globalMemories.map((m) => (
                  <MemoryCard key={m.id} memory={m} scope="global" onAccept={onAcceptMemory} onCancel={onCancelMemory} />
                ))}
              </div>
            )}
            {projectMemories && projectMemories.length > 0 && (
              <div className="space-y-3">
                {projectMemories.map((m) => (
                  <MemoryCard key={m.id} memory={m} scope="project" onAccept={onAcceptMemory} onCancel={onCancelMemory} />
                ))}
              </div>
            )}
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  )
}
