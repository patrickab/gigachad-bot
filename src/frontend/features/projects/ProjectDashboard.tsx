"use client"

import { useCallback, useEffect, useState, useRef } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Plus, X } from "lucide-react"
import { cn } from "@/lib/utils"
import { loadProject, addProjectCard, moveProjectCard, deleteProjectCard } from "@/lib/backend/api"
import { useUndoDelete } from "@/contexts/UndoDeleteContext"
import { Dialog } from "@/components/ui/Dialog"
import { useResetOnOpen } from "@/hooks/useResetOnChange"
import type { KanbanColumnId, ProjectData } from "@/lib/domain/types"
import { ElevationProvider, ElevatedContainer } from "@/components/ui/ElevatedContainer"
import { FloatingWindow } from "@/components/ui/FloatingWindow"

const COLUMNS: { id: KanbanColumnId; label: string }[] = [
  { id: "backlog", label: "Backlog" },
  { id: "doing", label: "Doing" },
  { id: "done", label: "Done" },
]

// Click cycles forward, right-click cycles backward
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

const INPUT_CLASS = cn(
  "w-full rounded-lg border border-divider-strong bg-surface-elevated/60 px-3 py-2.5 text-sm text-ink placeholder-ink-faint",
  "outline-none transition-all duration-200 focus:border-ink-muted"
)

const emptyCounters = () => ({ backlog: 0, doing: 0, done: 0 })

type Card = ProjectData["kanban"][number]

/** `data` with card `cardId` replaced by `next(card)`. */
const patchCard = (data: ProjectData | null, cardId: string, next: (card: Card) => Card): ProjectData | null =>
  data ? { ...data, kanban: data.kanban.map((card) => card.id === cardId ? next(card) : card) } : data

interface AddCardModalProps {
  open: boolean
  onClose: () => void
  onAdd: (title: string, description: string, state: KanbanColumnId) => void
  defaultState: KanbanColumnId
}

/** Modal to add a card to `defaultState`. Enter submits. */
function AddCardModal({ open, onClose, onAdd, defaultState }: AddCardModalProps) {
  const [title, setTitle] = useState("")
  const [desc, setDesc] = useState("")
  const titleRef = useRef<HTMLInputElement>(null)

  // Start each opening with an empty form.
  useResetOnOpen(open, () => {
    setTitle("")
    setDesc("")
  })

  const handleKeyDown = (e: React.KeyboardEvent<HTMLFormElement>) => {
    if (e.key === "Enter") {
      e.preventDefault()
      if (!title.trim()) return
      onAdd(title.trim(), desc.trim(), defaultState)
      onClose()
    }
  }

  return (
    <Dialog
      open={open}
      onClose={onClose}
      icon={Plus}
      title={<>Add Card to <span className="text-ink-muted font-semibold uppercase text-xs">{defaultState}</span></>}
      initialFocusRef={titleRef}
      panelClassName="rounded-2xl border border-divider-strong bg-surface shadow-[var(--shadow-lg)]"
    >
      <form onKeyDown={handleKeyDown} className="px-5 pb-5 space-y-3">
        <input
          ref={titleRef}
          type="text"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          className={INPUT_CLASS}
          placeholder="Title"
          required
          autoComplete="off"
          spellCheck={false}
        />
        <input
          type="text"
          value={desc}
          onChange={(e) => setDesc(e.target.value)}
          className={INPUT_CLASS}
          placeholder="Description (optional)"
          autoComplete="off"
          spellCheck={false}
        />
      </form>
    </Dialog>
  )
}

/** Load one project's board and apply card changes to it. Remount per `slug` (via `key`) so a late response never lands on another project. */
function useProjectBoard(slug: string, onLoadFailed: () => void) {
  const [projectData, setProjectData] = useState<ProjectData | null>(null)
  const onLoadFailedRef = useRef(onLoadFailed)
  useEffect(() => { onLoadFailedRef.current = onLoadFailed }, [onLoadFailed])

  useEffect(() => {
    let stale = false
    loadProject(slug).then(
      (data) => { if (!stale) setProjectData(data) },
      () => { if (!stale) onLoadFailedRef.current() },
    )
    return () => { stale = true }
  }, [slug])

  const addCard = useCallback(async (title: string, description: string, state: string = "backlog") => {
    const card = await addProjectCard(slug, title, description, state)
    setProjectData((prev) => prev ? { ...prev, kanban: [...prev.kanban, card] } : prev)
  }, [slug])

  const moveCard = useCallback(async (cardId: string, state: KanbanColumnId) => {
    // Optimistic move. `originalCard` is captured inside the updater so a failed call can restore it.
    let originalCard: Card | undefined
    setProjectData((prev) => {
      originalCard = prev?.kanban.find((c) => c.id === cardId)
      return patchCard(prev, cardId, (card) => ({ ...card, state }))
    })
    try {
      const updated = await moveProjectCard(slug, cardId, state)
      setProjectData((prev) => patchCard(prev, cardId, () => updated))
    } catch {
      const original = originalCard
      if (original) setProjectData((prev) => patchCard(prev, cardId, () => original))
    }
  }, [slug])

  const deleteCard = useCallback(async (cardId: string) => {
    await deleteProjectCard(slug, cardId)
    setProjectData((prev) => prev ? { ...prev, kanban: prev.kanban.filter((c) => c.id !== cardId) } : prev)
  }, [slug])

  return { projectData, addCard, moveCard, deleteCard }
}

/** Kanban board for the project `slug`. Cards drag between columns, click to advance, right-click to go back. */
export function ProjectDashboard({ slug, onClose }: { slug: string; onClose: () => void }) {
  const { projectData, addCard, moveCard, deleteCard } = useProjectBoard(slug, onClose)
  const { schedule, pending } = useUndoDelete()
  const [addModalState, setAddModalState] = useState<KanbanColumnId | null>(null)
  const [draggedCardId, setDraggedCardId] = useState<string | null>(null)
  const [dragOverCol, setDragOverCol] = useState<KanbanColumnId | null>(null)
  // Per-column drag-enter counter to avoid false dragLeave on child elements.
  const dragCounters = useRef<Record<string, number>>(emptyCounters())

  if (!projectData) return null

  // Cards with a pending undoable delete are hidden
  const cardsByColumn = (col: KanbanColumnId) =>
    projectData.kanban.filter((card) => card.state === col && !pending(`card:${slug}:${card.id}`))

  return (
    <>
      <AddCardModal
        open={addModalState !== null}
        onClose={() => setAddModalState(null)}
        onAdd={addCard}
        defaultState={addModalState ?? "backlog"}
      />

      <FloatingWindow onClose={onClose} panelClassName="w-11/12 bg-transparent backdrop-blur-none">
          <ElevationProvider>
            <ElevatedContainer className="w-full h-full flex flex-col overflow-hidden">
              <div className="pt-6 px-8 flex items-center justify-between border-b border-divider/20 pb-4 select-none shrink-0">
                <div className="flex items-center gap-3">
                   <h2 className="text-lg font-semibold tracking-tight text-ink">
                     {projectData.name || slug} Dashboard
                   </h2>
                </div>
              </div>

              <div className="flex-1 p-10 pt-8 flex gap-6 overflow-hidden">
                {COLUMNS.map((col) => {
                  const cards = cardsByColumn(col.id)
                  const isOver = dragOverCol === col.id

                  return (
                    <ElevatedContainer
                      key={col.id}
                      className={cn(
                        "flex-1 flex flex-col min-w-0 min-h-0 rounded-2xl border overflow-hidden transition-all duration-200",
                        isOver ? "border-divider-strong" : "border-divider/40",
                      )}
                    >
                      <div className="flex items-center justify-between px-4 pt-4 pb-3 select-none shrink-0">
                        <div className="flex items-center gap-2">
                           <span className="text-[11px] font-bold text-ink-muted uppercase tracking-[0.2em] tabular-nums">
                             {col.label}
                           </span>
                           <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded-md bg-surface-elevated text-ink tabular-nums">
                            {cards.length}
                          </span>
                        </div>
                        <button aria-label={`Add card to ${col.label}`}
                          onClick={() => setAddModalState(col.id)}
                          className="p-1 rounded-md text-ink-subtle hover:text-ink hover:bg-hover transition-all duration-200"
                        >
                          <Plus className="h-4 w-4" />
                        </button>
                      </div>

                      <div
                        onDragOver={(e) => { e.preventDefault() }}
                        onDragEnter={(e) => {
                          e.preventDefault()
                          dragCounters.current[col.id] = (dragCounters.current[col.id] ?? 0) + 1
                          setDragOverCol(col.id)
                        }}
                        onDragLeave={() => {
                          dragCounters.current[col.id] = Math.max(0, (dragCounters.current[col.id] ?? 1) - 1)
                          if (dragCounters.current[col.id] === 0) setDragOverCol(null)
                        }}
                        onDrop={(e) => {
                          const cardId = e.dataTransfer.getData("text/plain")
                          dragCounters.current[col.id] = 0
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
                                onDragStart={(e) => {
                                  e.dataTransfer.setData("text/plain", card.id)
                                  // Defer so the drag image is captured before the original fades out
                                  setTimeout(() => setDraggedCardId(card.id), 0)
                                }}
                                onDragEnd={() => {
                                  dragCounters.current = emptyCounters()
                                  setDraggedCardId(null)
                                  setDragOverCol(null)
                                }}
                                // Click advances the card one column, cycling
                                onClick={() => moveCard(card.id, NEXT_STATE[col.id])}
                                // Right-click moves the card back one column
                                onContextMenu={(e) => {
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
                                  aria-label={`Delete ${card.title}`}
                                  onClick={(event) => {
                                    event.stopPropagation()
                                    schedule(`card:${slug}:${card.id}`, card.title, () => deleteCard(card.id))
                                  }}
                                  className="absolute top-3 right-3 shrink-0 opacity-0 group-hover:opacity-100 p-1.5 rounded-lg hover:bg-danger-soft text-ink-faint hover:text-danger transition-all"
                                >
                                  <X className="h-3.5 w-3.5" />
                                </button>
                              </ElevatedContainer>
                            </motion.div>
                          ))}
                        </AnimatePresence>

                        {/* Placeholder showing where the dragged card will land */}
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
      </FloatingWindow>
    </>
  )
}
