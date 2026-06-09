"use client"

import { useState, useRef, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Plus, X } from "lucide-react"
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
          className="fixed inset-0 z-[100] flex items-center justify-center bg-backdrop backdrop-blur-md"
          onClick={onClose}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.96, y: 8 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: 8 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            onClick={(e) => e.stopPropagation()}
            className="w-full max-w-sm mx-4 rounded-2xl border border-divider-strong bg-surface shadow-[var(--shadow-lg)] overflow-hidden"
          >
            <div className="px-5 pt-5 pb-3">
              <div className="flex items-center gap-2 text-sm font-medium text-ink">
                <Plus className="h-4 w-4 text-ink" />
                Add Card to <span className="text-ink-muted font-semibold uppercase text-xs">{defaultState}</span>
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
                  "w-full rounded-lg border border-divider-strong bg-surface-elevated/60 px-3 py-2.5 text-sm text-ink placeholder-ink-faint",
                   "outline-none transition-all duration-200 focus:border-ink-muted"
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
                   "w-full rounded-lg border border-divider-strong bg-surface-elevated/60 px-3 py-2.5 text-sm text-ink placeholder-ink-faint",
                   "outline-none transition-all duration-200 focus:border-ink-muted"
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
        className="fixed inset-0 z-[90] flex items-center justify-center bg-backdrop backdrop-blur-sm"
        onClick={() => setDashboardOpen(false)}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.98, y: 10 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.98, y: 10 }}
          transition={{ duration: 0.25, ease: "easeOut" }}
          onClick={(e) => e.stopPropagation()}
          className="w-11/12 max-w-6xl h-[85vh] max-h-[800px] rounded-3xl border border-divider/50 overflow-hidden flex flex-col shadow-[var(--shadow-xl)]"
        >
          <ElevationProvider darkColor="var(--paper)" brightColor="var(--surface-elevated)" numLevels={3}>
            <ElevatedContainer className="w-full h-full flex flex-col overflow-hidden">
              {/* Board Header */}
              <div className="pt-6 px-8 flex items-center justify-between border-b border-divider/20 pb-4 select-none shrink-0">
                <div className="flex items-center gap-3">
                   <h2 className="text-lg font-semibold tracking-tight text-ink">
                     {projectData.name || activeProject} Dashboard
                   </h2>
                </div>
              </div>

              {/* Board Canvas */}
              <div className="flex-1 p-10 pt-8 flex gap-6 overflow-hidden">
                {COLUMNS.map((col) => {
                  const cards = cardsByColumn(col.id)
                  const isBacklog = col.id === "backlog"
                  const isDoing = col.id === "doing"
                  const isDone = col.id === "done"
                  const isOver = dragOverCol === col.id

                  return (
                    <ElevatedContainer
                      key={col.id}
                      className={cn(
                        "flex-1 flex flex-col min-w-0 min-h-0 rounded-2xl border overflow-hidden transition-all duration-200",
                        isOver ? "border-divider-strong" : "border-divider/40",
                      )}
                    >
                      {/* Column Header */}
                      <div className="flex items-center justify-between px-4 pt-4 pb-3 select-none shrink-0">
                        <div className="flex items-center gap-2">
                           <span className="text-[11px] font-bold text-ink-muted uppercase tracking-[0.2em] tabular-nums">
                             {col.label}
                           </span>
                           <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded-md bg-surface-elevated text-ink tabular-nums">
                            {cards.length}
                          </span>
                        </div>
                        <button
                          onClick={() => setAddModalState(col.id)}
                          className="p-1 rounded-md text-ink-subtle hover:text-ink hover:bg-hover transition-all duration-200"
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
                        className="flex-1 overflow-y-auto space-y-2.5 px-4 pb-4 scrollbar-thin transition-all duration-200"
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
                              <ElevatedContainer
                                draggable
                                onDragStart={(e: any) => {
                                  e.dataTransfer.setData("text/plain", card.id)
                                  setTimeout(() => setDraggedCardId(card.id), 0)
                                }}
                                onDragEnd={() => {
                                  setDraggedCardId(null)
                                  setDragOverCol(null)
                                }}
                                onClick={() => moveCard(card.id, NEXT_STATE[col.id])}
                                onContextMenu={(e: any) => {
                                  e.preventDefault()
                                  moveCard(card.id, PREV_STATE[col.id])
                                }}
                                hoverLift
                                className="group relative w-full rounded-xl border border-divider/40 overflow-hidden cursor-grab active:cursor-grabbing select-none transition-all duration-300 shadow-[var(--shadow-md)]"
                              >
                                <div className="p-4 flex flex-col gap-1 pr-6">
                                  <span className="text-sm font-semibold text-ink">
                                    {card.title}
                                  </span>
                                  {card.description && (
                                    <span className="text-xs text-ink-muted leading-relaxed mt-1 line-clamp-2">
                                      {card.description}
                                    </span>
                                  )}
                                </div>

                                <button
                                  onClick={async (e: any) => {
                                    e.stopPropagation()
                                    await deleteCard(card.id)
                                  }}
                                  className="absolute top-3 right-3 shrink-0 opacity-0 group-hover:opacity-100 p-1.5 rounded-lg hover:bg-danger-soft text-ink-faint hover:text-danger transition-all"
                                >
                                  <X className="h-3.5 w-3.5" />
                                </button>
                              </ElevatedContainer>
                            </motion.div>
                          ))}
                        </AnimatePresence>

                        {/* Skeleton Preview */}
                        {isOver && draggedCardId && (
                          <div className="w-full p-4 rounded-xl border border-dashed border-divider bg-overlay/60 h-16" />
                        )}
                      </div>
                    </ElevatedContainer>
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
