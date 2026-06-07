"use client"

import { useState, useRef, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Plus, Layers, Activity, CheckCircle2, ArrowRight, X } from "lucide-react"
import { cn } from "@/lib/utils"
import { useProject } from "@/contexts/ProjectContext"
import type { KanbanColumnId } from "@/lib/types"
import { ElevationProvider, ElevatedContainer } from "./ElevatedContainer"

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

const PREV_STATE: Record<KanbanColumnId, KanbanColumnId> = {
  backlog: "done",
  doing: "backlog",
  done: "doing",
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

  const handleKeyDown = (e: React.KeyboardEvent<HTMLFormElement>) => {
    if (e.key === "Enter") {
      e.preventDefault()
      submit()
    }
  }

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.15 }}
          className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-md"
          onClick={onClose}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.96, y: 8 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: 8 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            onClick={(e) => e.stopPropagation()}
            className="w-full max-w-sm mx-4 rounded-2xl border border-zinc-700 bg-zinc-900 shadow-[0_8px_30px_rgba(0,0,0,0.08)] overflow-hidden"
          >
            <div className="px-5 pt-5 pb-3">
              <div className="flex items-center gap-2 text-sm font-medium text-zinc-300">
                <Plus className="h-4 w-4 text-cyan-400" />
                Add Card to <span className="text-zinc-400 font-semibold uppercase text-xs">{defaultState}</span>
              </div>
            </div>

            <form
              onSubmit={(e) => {
                e.preventDefault()
                submit()
              }}
              onKeyDown={handleKeyDown}
              className="px-5 pb-5 space-y-3"
            >
              <input
                ref={titleRef}
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                className={cn(
                  "w-full rounded-lg border border-zinc-700 bg-zinc-800/60 px-3 py-2.5 text-sm text-zinc-200 placeholder-zinc-600",
                  "outline-none transition-all duration-200",
                  "focus:border-cyan-500/30 focus:shadow-[0_8px_30px_rgba(6,182,212,0.03)]"
                )}
                placeholder="Title"
                required
                autoComplete="off"
                spellCheck={false}
              />
              <input
                type="text"
                value={desc}
                onChange={(e) => setDesc(e.target.value)}
                className={cn(
                  "w-full rounded-lg border border-zinc-700 bg-zinc-800/60 px-3 py-2.5 text-sm text-zinc-200 placeholder-zinc-600",
                  "outline-none transition-all duration-200",
                  "focus:border-cyan-500/30 focus:shadow-[0_8px_30px_rgba(6,182,212,0.03)]"
                )}
                placeholder="Description (optional)"
                autoComplete="off"
                spellCheck={false}
              />
              <button type="submit" className="hidden" />
            </form>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

export function ProjectDashboard() {
  const { projectData, activeProject, addCard, moveCard, deleteCard, setDashboardOpen } = useProject()
  const [addModalState, setAddModalState] = useState<KanbanColumnId | null>(null)
  const [draggedCardId, setDraggedCardId] = useState<string | null>(null)
  const [dragOverCol, setDragOverCol] = useState<KanbanColumnId | null>(null)

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
        className="fixed inset-0 z-[90] flex items-center justify-center bg-black/80 backdrop-blur-sm"
        onClick={() => setDashboardOpen(false)}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.98, y: 10 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.98, y: 10 }}
          transition={{ duration: 0.25, ease: "easeOut" }}
          onClick={(e) => e.stopPropagation()}
          className="w-11/12 max-w-6xl h-[85vh] max-h-[800px] rounded-3xl border border-zinc-800/50 overflow-hidden flex flex-col shadow-2xl"
        >
          <ElevationProvider darkColor="var(--color-zinc-950)" brightColor="var(--color-zinc-900)" numLevels={3}>
            <ElevatedContainer className="w-full h-full flex flex-col overflow-hidden">
              {/* Board Header */}
              <div className="pt-6 px-8 flex items-center justify-between border-b border-zinc-800/20 pb-4 select-none">
                <div className="flex items-center gap-3">
                  <h2 className="text-base font-bold text-zinc-100">
                    {projectData.name || activeProject} Dashboard
                  </h2>
                </div>
              </div>

              {/* Board Canvas */}
              <div className="flex-1 p-10 pt-8 flex gap-10 overflow-hidden">
                {COLUMNS.map((col) => {
                  const cards = cardsByColumn(col.id)
                  const isBacklog = col.id === "backlog"
                  const isDoing = col.id === "doing"
                  const isDone = col.id === "done"
                  const isOver = dragOverCol === col.id

                  return (
                    <div key={col.id} className="flex-1 flex flex-col min-w-0 min-h-0">
                      {/* Column Header */}
                      <div className="flex items-center justify-between mb-4 px-2 select-none">
                        <div className="flex items-center gap-2">
                          <span className="text-[11px] font-bold text-zinc-400 uppercase tracking-[0.2em]">
                            {col.label}
                          </span>
                          <span className={cn(
                            "text-[10px] font-mono font-bold px-2 py-0.5 rounded-md transition-colors",
                            isBacklog && "bg-amber-500/10 text-amber-400",
                            isDoing && "bg-cyan-500/10 text-cyan-400",
                            isDone && "bg-emerald-500/10 text-emerald-400"
                          )}>
                            {cards.length}
                          </span>
                        </div>
                        <button
                          onClick={() => setAddModalState(col.id)}
                          className="p-1 rounded-md text-zinc-500 hover:text-cyan-400 hover:bg-cyan-500/10 transition-all duration-200"
                          title={`Add task to ${col.label}`}
                        >
                          <Plus className="h-4 w-4" />
                        </button>
                      </div>

                      {/* Drop Zone / Cards List */}
                      <div
                        onDragOver={(e) => {
                          e.preventDefault()
                          setDragOverCol(col.id)
                        }}
                        onDragLeave={() => setDragOverCol(null)}
                        onDrop={(e) => {
                          const cardId = e.dataTransfer.getData("text/plain")
                          setDragOverCol(null)
                          setDraggedCardId(null)
                          if (cardId) moveCard(cardId, col.id)
                        }}
                        className={cn(
                          "flex-1 overflow-y-auto space-y-3 p-[18px] rounded-2xl border border-zinc-800/30 bg-zinc-900/35 border-l-[3px] backdrop-blur-md scrollbar-thin transition-all duration-200",
                          isBacklog && "border-l-amber-500/35",
                          isDoing && "border-l-cyan-500/35",
                          isDone && "border-l-emerald-500/35",
                          isOver && "bg-zinc-800/15 border-zinc-700/40"
                        )}
                      >
                        <AnimatePresence mode="popLayout" initial={false}>
                          {cards.map((card) => (
                            <motion.div
                              key={card.id}
                              layout
                              initial={{ opacity: 0, scale: 0.95 }}
                              animate={{ 
                                opacity: draggedCardId === card.id ? 0 : 1,
                                scale: 1 
                              }}
                              exit={{ opacity: 0, scale: 0.95 }}
                              transition={{ duration: 0.2 }}
                            >
                              <div
                                draggable
                                onDragStart={(e) => {
                                  e.dataTransfer.setData("text/plain", card.id)
                                  setTimeout(() => setDraggedCardId(card.id), 0)
                                }}
                                onDragEnd={() => {
                                  setDraggedCardId(null)
                                  setDragOverCol(null)
                                }}
                                onClick={() => moveCard(card.id, NEXT_STATE[col.id])}
                                onContextMenu={(e) => {
                                  e.preventDefault()
                                  moveCard(card.id, PREV_STATE[col.id])
                                }}
                                className="group relative w-full p-4 rounded-2xl border border-zinc-600/15 bg-zinc-800 hover:border-cyan-500/25 text-zinc-100 overflow-hidden cursor-pointer transition-all duration-300 shadow-[0_4px_16px_rgba(0,0,0,0.08)] hover:shadow-[0_12px_24px_rgba(0,0,0,0.12)]"
                              >
                                <div className="flex flex-col gap-1 pr-6">
                                  <span className="text-sm font-semibold text-zinc-100 group-hover:text-cyan-400 transition-colors duration-200">
                                    {card.title}
                                  </span>
                                  {card.description && (
                                    <span className="text-xs text-zinc-400 leading-relaxed mt-1 line-clamp-2">
                                      {card.description}
                                    </span>
                                  )}
                                </div>

                                <button
                                  onClick={async (e) => {
                                    e.stopPropagation()
                                    await deleteCard(card.id)
                                  }}
                                  className="absolute top-4 right-3 shrink-0 opacity-0 group-hover:opacity-100 p-1.5 rounded-lg hover:bg-red-500/10 text-zinc-600 hover:text-red-400 transition-all"
                                >
                                  <X className="h-3.5 w-3.5" />
                                </button>
                              </div>
                            </motion.div>
                          ))}
                        </AnimatePresence>

                        {/* Skeleton Preview */}
                        {isOver && draggedCardId && (
                          <div className="w-full p-4 rounded-xl border border-dashed border-zinc-800 bg-zinc-900/40 h-16" />
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>
            </ElevatedContainer>
          </ElevationProvider>
        </motion.div>
      </motion.div>
    </>
  )
}
