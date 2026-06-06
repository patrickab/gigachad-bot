"use client"

import { useState, useRef, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Plus, Layers, Activity, CheckCircle2, ArrowRight, X } from "lucide-react"
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
            className="w-full max-w-sm mx-4 rounded-xl border border-zinc-800 bg-zinc-950 shadow-2xl overflow-hidden"
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
                  "w-full rounded-lg border border-zinc-800 bg-zinc-900/60 px-3 py-2.5 text-sm text-zinc-200 placeholder-zinc-600",
                  "outline-none transition-all duration-200",
                  "focus:border-cyan-500/40 focus:shadow-[0_0_20px_rgba(6,182,212,0.06)]"
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
                  "w-full rounded-lg border border-zinc-800 bg-zinc-900/60 px-3 py-2.5 text-sm text-zinc-200 placeholder-zinc-600",
                  "outline-none transition-all duration-200",
                  "focus:border-cyan-500/40 focus:shadow-[0_0_20px_rgba(6,182,212,0.06)]"
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
            initial={{ opacity: 0, scale: 0.95, y: 4 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 4 }}
            transition={{ duration: 0.15, ease: "easeOut" }}
            className="absolute left-1/2 -translate-x-1/2 bottom-full z-50 mb-2.5 w-64 rounded-lg border border-zinc-700 bg-zinc-800 shadow-[0_12px_24px_rgba(0,0,0,0.5)] p-3 text-left pointer-events-none"
          >
            {/* Tiny arrow pointing down */}
            <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-1 border-4 border-transparent border-t-zinc-800" />
            <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-[5px] border-4 border-transparent border-t-zinc-700 -z-10" />
            
            <p className="text-[11px] text-zinc-300 font-normal whitespace-pre-wrap leading-relaxed">{description}</p>
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
        className="fixed inset-0 z-[90] flex items-center justify-center bg-black/60 backdrop-blur-md"
        onClick={() => setDashboardOpen(false)}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.96 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.96 }}
          transition={{ duration: 0.2, ease: "easeOut" }}
          onClick={(e) => e.stopPropagation()}
          className="w-11/12 max-w-5xl h-[75vh] max-h-[720px] rounded-2xl border-none bg-zinc-950 shadow-[0_35px_100px_rgba(0,0,0,0.95)] flex flex-col overflow-hidden p-6"
        >
          {/* Columns Canvas */}
          <div className="flex-1 flex gap-4 overflow-hidden bg-zinc-950">
            {COLUMNS.map((col) => {
              const cards = cardsByColumn(col.id)
              
              // Dynamic state colors for colormix columns
              const isBacklog = col.id === "backlog"
              const isDoing = col.id === "doing"
              const isDone = col.id === "done"

              return (
                <div
                  key={col.id}
                  className={cn(
                    "flex-1 flex flex-col min-w-0 rounded-2xl border bg-assistant-message p-4 h-full shadow-[0_4px_20px_rgba(0,0,0,0.4)]",
                    isBacklog && "border-zinc-800/40 border-t-2 border-t-amber-500/35",
                    isDoing && "border-zinc-800/40 border-t-2 border-t-cyan-500/45",
                    isDone && "border-zinc-800/40 border-t-2 border-t-emerald-500/35"
                  )}
                >
                  {/* Column Header */}
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                      <span className={cn(
                        "p-1 rounded-md",
                        isBacklog && "bg-amber-500/10 text-amber-400",
                        isDoing && "bg-cyan-500/10 text-cyan-400",
                        isDone && "bg-emerald-500/10 text-emerald-400"
                      )}>
                        {isBacklog && <Layers className="h-3.5 w-3.5" />}
                        {isDoing && <Activity className="h-3.5 w-3.5" />}
                        {isDone && <CheckCircle2 className="h-3.5 w-3.5" />}
                      </span>
                      <span className="text-xs font-semibold text-zinc-200 tracking-wide uppercase">
                        {col.label}
                      </span>
                      <span className={cn(
                        "text-[10px] font-mono px-1.5 py-0.5 rounded-full border",
                        isBacklog && "bg-amber-500/5 text-amber-400/80 border-amber-500/15",
                        isDoing && "bg-cyan-500/5 text-cyan-400/80 border-cyan-500/15",
                        isDone && "bg-emerald-500/5 text-emerald-400/80 border-emerald-500/15"
                      )}>
                        {cards.length}
                      </span>
                    </div>
                  </div>

                  {/* Cards List */}
                  <div
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={(e) => {
                      const cardId = e.dataTransfer.getData("text/plain")
                      if (cardId) moveCard(cardId, col.id)
                    }}
                    className="flex-1 overflow-y-auto space-y-3 pr-1 scrollbar-thin"
                  >
                    {cards.map((card) => (
                      <CardTooltip key={card.id} description={card.description}>
                        <button
                          draggable
                          onDragStart={(e) => {
                            e.dataTransfer.setData("text/plain", card.id)
                          }}
                          onClick={() => moveCard(card.id, NEXT_STATE[col.id])}
                          onContextMenu={(e) => {
                            e.preventDefault()
                            moveCard(card.id, PREV_STATE[col.id])
                          }}
                          className={cn(
                            "w-full text-left p-3.5 rounded-xl border border-zinc-800/40 bg-zinc-900 hover:bg-zinc-850/60 shadow-sm hover:shadow-[0_8px_20px_rgba(0,0,0,0.35)] relative hover:z-10 transition-all duration-200 group cursor-pointer overflow-hidden"
                          )}
                        >
                          <div className="flex items-start justify-between gap-2">
                            <span className="text-xs font-medium text-zinc-100 break-words leading-relaxed">{card.title}</span>
                            <span
                              onClick={async (e) => { e.stopPropagation(); await deleteCard(card.id) }}
                              className="shrink-0 opacity-0 group-hover:opacity-100 p-1 rounded-lg hover:bg-zinc-800 text-zinc-500 hover:text-red-400 transition-all"
                              title="Delete Card"
                            >
                              <X className="h-3 w-3" />
                            </span>
                          </div>
                          
                          {card.description && (
                            <p className="text-[11px] text-zinc-400 mt-1.5 line-clamp-2 leading-relaxed font-normal">{card.description}</p>
                          )}
                        </button>
                      </CardTooltip>
                    ))}
                    {cards.length === 0 && (
                      <button
                        onClick={() => setAddModalState(col.id)}
                        className="w-full flex items-center justify-center gap-2 py-3.5 px-3.5 rounded-xl border border-dashed border-zinc-800/60 hover:border-zinc-700/80 hover:bg-zinc-950/10 text-zinc-500 hover:text-zinc-400 transition-all duration-200 group cursor-pointer"
                      >
                        <Plus className="h-4 w-4 stroke-[2] group-hover:text-zinc-300 transition-colors" />
                        <span className="text-xs font-medium">Click to Add Card</span>
                      </button>
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
