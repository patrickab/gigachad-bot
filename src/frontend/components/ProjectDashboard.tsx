"use client"

import { useState, useRef, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Plus, X } from "lucide-react"
import { cn } from "@/lib/utils"
import { useProject } from "@/contexts/ProjectContext"
import type { KanbanColumnId } from "@/lib/types"

const COLUMNS: { id: KanbanColumnId; label: string }[] = [
  { id: "backlog", label: "Backlog" },
  { id: "doing", label: "Doing" },
  { id: "done", label: "Done" },
]

const NEXT_STATE: Record<KanbanColumnId, KanbanColumnId> = {
  backlog: "doing",
  doing: "done",
  done: "backlog",
}

interface AddCardModalProps {
  open: boolean
  onClose: () => void
  onAdd: (title: string, description: string, state: KanbanColumnId) => void
  defaultState: KanbanColumnId
}

function AddCardModal({ open, onClose, onAdd, defaultState }: AddCardModalProps) {
  const [title, setTitle] = useState("")
  const [desc, setDesc] = useState("")
  const titleRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (open) {
      setTitle("")
      setDesc("")
      setTimeout(() => titleRef.current?.focus(), 50)
    }
  }, [open])

  useEffect(() => {
    if (!open) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose()
    }
    document.addEventListener("keydown", handler)
    return () => document.removeEventListener("keydown", handler)
  }, [open, onClose])

  const submit = () => {
    if (!title.trim()) return
    onAdd(title.trim(), desc.trim(), defaultState)
    onClose()
  }

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.15 }}
          className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-[2px]"
          onClick={onClose}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.96, y: 8 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: 8 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            onClick={(e) => e.stopPropagation()}
            className="w-full max-w-sm mx-4 rounded-xl border border-zinc-700/60 bg-zinc-800 shadow-2xl overflow-hidden"
          >
            <div className="px-5 pt-5 pb-3">
              <div className="flex items-center gap-2 text-sm font-medium text-zinc-200">
                <Plus className="h-4 w-4 text-cyan-400" />
                Add Card
              </div>
            </div>

            <form
              onSubmit={(e) => { e.preventDefault(); submit() }}
              className="px-5 pb-5 space-y-3"
            >
              <input
                ref={titleRef}
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                className={cn(
                  "w-full rounded-lg border border-zinc-700 bg-zinc-900/60 px-3 py-2.5 text-sm text-zinc-200 placeholder-zinc-500",
                  "outline-none transition-all duration-200",
                  "focus:border-cyan-500/40 focus:shadow-[0_0_20px_rgba(6,182,212,0.06)]"
                )}
                autoComplete="off"
                spellCheck={false}
                placeholder="Title"
              />
              <input
                type="text"
                value={desc}
                onChange={(e) => setDesc(e.target.value)}
                className={cn(
                  "w-full rounded-lg border border-zinc-700 bg-zinc-900/60 px-3 py-2 text-sm text-zinc-300 placeholder-zinc-500",
                  "outline-none transition-all duration-200",
                  "focus:border-cyan-500/40 focus:shadow-[0_0_20px_rgba(6,182,212,0.06)]"
                )}
                autoComplete="off"
                spellCheck={false}
                placeholder="Description (optional)"
              />
              <button type="submit" className="hidden" />
            </form>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

interface CardTooltipProps {
  description: string
  children: React.ReactNode
}

function CardTooltip({ description, children }: CardTooltipProps) {
  const [visible, setVisible] = useState(false)

  if (!description) return <>{children}</>

  return (
    <div
      className="relative"
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
    >
      {children}
      <AnimatePresence>
        {visible && (
          <motion.div
            initial={{ opacity: 0, y: 2 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 2 }}
            transition={{ duration: 0.1 }}
            className="absolute left-0 right-0 top-full z-50 mt-0.5 rounded-md border border-zinc-600/50 bg-zinc-700 shadow-lg px-2.5 py-2 max-h-[30%] overflow-y-auto"
          >
            <p className="text-[11px] text-zinc-200 whitespace-pre-wrap leading-relaxed">{description}</p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export function ProjectDashboard() {
  const { projectData, activeProject, addCard, moveCard, deleteCard, setDashboardOpen } = useProject()
  const [addModalState, setAddModalState] = useState<KanbanColumnId | null>(null)

  useEffect(() => {
    if (addModalState !== null) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") setDashboardOpen(false)
    }
    document.addEventListener("keydown", handler)
    return () => document.removeEventListener("keydown", handler)
  }, [setDashboardOpen, addModalState])

  if (!projectData || !activeProject) return null

  const cardsByColumn = (col: KanbanColumnId) =>
    projectData.kanban.filter((c) => c.state === col)

  return (
    <>
      <AddCardModal
        open={addModalState !== null}
        onClose={() => setAddModalState(null)}
        onAdd={(title, description, state) => addCard(title, description, state)}
        defaultState={addModalState ?? "backlog"}
      />

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.15 }}
        className="fixed inset-0 z-[90] flex items-center justify-center bg-black/40 backdrop-blur-sm"
        onClick={() => setDashboardOpen(false)}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.96 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.96 }}
          transition={{ duration: 0.2, ease: "easeOut" }}
          onClick={(e) => e.stopPropagation()}
          className="w-1/2 h-1/2 min-w-[600px] min-h-[400px] rounded-xl border border-zinc-600/60 bg-zinc-800 shadow-2xl flex flex-col overflow-hidden"
        >
          <div className="px-4 py-3 border-b border-zinc-700/50">
            <h2 className="text-sm font-medium text-zinc-100">{activeProject}</h2>
          </div>

          <div className="flex-1 flex gap-3 p-4 overflow-hidden">
            {COLUMNS.map((col) => {
              const cards = cardsByColumn(col.id)
              return (
                <div key={col.id} className="flex-1 flex flex-col min-w-0">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">
                      {col.label} <span className="text-zinc-500 font-normal">({cards.length})</span>
                    </span>
                    <button
                      onClick={() => setAddModalState(col.id)}
                      className="p-1 rounded hover:bg-zinc-700 text-zinc-500 hover:text-cyan-400 transition-colors"
                    >
                      <Plus className="h-3.5 w-3.5" />
                    </button>
                  </div>
                  <div className="flex-1 overflow-y-auto space-y-2 pr-1">
                    {cards.map((card) => (
                      <CardTooltip key={card.id} description={card.description}>
                        <button
                          onClick={() => moveCard(card.id, NEXT_STATE[col.id])}
                          className="w-full text-left p-2.5 rounded-lg border border-zinc-600/40 bg-zinc-700/50 hover:bg-zinc-600/50 hover:border-zinc-500/40 transition-colors group"
                        >
                          <div className="flex items-start justify-between gap-1">
                            <span className="text-xs text-zinc-100 break-words leading-snug">{card.title}</span>
                            <span
                              onClick={async (e) => { e.stopPropagation(); await deleteCard(card.id) }}
                              className="shrink-0 opacity-0 group-hover:opacity-100 p-0.5 rounded hover:bg-zinc-500/40 text-zinc-500 hover:text-red-400 transition-all"
                            >
                              <X className="h-3 w-3" />
                            </span>
                          </div>
                          {card.description && (
                            <p className="text-[11px] text-zinc-400 mt-1 truncate">{card.description}</p>
                          )}
                        </button>
                      </CardTooltip>
                    ))}
                    {cards.length === 0 && (
                      <div className="py-4 text-center text-[11px] text-zinc-500">Empty</div>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        </motion.div>
      </motion.div>
    </>
  )
}