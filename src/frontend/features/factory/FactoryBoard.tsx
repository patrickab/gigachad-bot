"use client"

import { useState } from "react"
import { AnimatePresence, motion } from "framer-motion"
import { Plus, Settings } from "lucide-react"
import { ElevatedContainer, ElevationProvider } from "@/components/ui/ElevatedContainer"
import { cn } from "@/lib/utils"
import { fetchFactoryTasks } from "@/lib/backend/api"
import { FactorySettingsDialog } from "./FactorySettingsDialog"
import { NewTaskDialog } from "./NewTaskDialog"
import { BOARD_COLUMNS, PHASE_LABELS, phaseProgress, statusTone, taskStatusLine } from "./states"
import { StatusDot } from "./StatusDot"
import { ICON_BUTTON_CLASS, PRIMARY_BUTTON_CLASS } from "./styles"
import type { FactoryTask, Phase } from "@/lib/domain/types"
import { useFactoryPolling } from "./useFactoryPolling"

/** Done tasks shown before "Show all", the column otherwise grows without bound. */
const DONE_LIMIT = 10

function newestFirst(left: FactoryTask, right: FactoryTask): number {
  return right.updated_at.localeCompare(left.updated_at)
}

function workflowPhases(task: FactoryTask): string[] {
  const phases = task.config.phases
  return Array.isArray(phases) ? phases.filter((phase): phase is string => typeof phase === "string") : []
}

/** One segment per workflow phase: finished ones filled, the running one half-filled. */
function PhaseTrack({ task }: { task: FactoryTask }) {
  const phases = workflowPhases(task)
  if (phases.length === 0) return null
  const { finished, active } = phaseProgress(phases, task.state, task.phase)
  const label = phases.map((phase) => PHASE_LABELS[phase as Phase] ?? phase).join(", ")
  return (
    <div className="flex gap-1" role="img" aria-label={`Phases: ${label}, ${finished} finished`}>
      {phases.map((phase, index) => (
        <span
          key={phase}
          className={cn(
            "h-1 flex-1 rounded-full transition-colors duration-300",
            index < finished ? "bg-ink-muted" : index === active ? "bg-ink-faint" : "bg-divider/60",
          )}
        />
      ))}
    </div>
  )
}

function TaskCard({ task, onOpen }: { task: FactoryTask; onOpen: () => void }) {
  const tone = statusTone(task.state)
  return (
    <ElevatedContainer hoverLift className="rounded-xl border border-divider/40 shadow-[var(--shadow-md)]">
      <button
        type="button"
        onClick={onOpen}
        className="flex w-full flex-col gap-2 rounded-xl p-4 text-left focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"
      >
        <span className="line-clamp-2 text-sm font-semibold text-ink">{task.title}</span>
        {task.request && <span className="line-clamp-2 text-xs leading-relaxed text-ink-muted">{task.request}</span>}
        <PhaseTrack task={task} />
        <span className="flex min-w-0 items-center gap-2 text-[11px]">
          <StatusDot tone={tone} />
          <span className={cn("truncate", tone === "error" ? "text-danger" : "text-ink-subtle")}>
            {taskStatusLine(task.state, task.phase, task.round)}
          </span>
        </span>
        {tone === "error" && task.error && <span className="line-clamp-2 text-[11px] text-ink-subtle">{task.error}</span>}
      </button>
    </ElevatedContainer>
  )
}

/** Kanban board of the factory's tasks, kept current by polling the event log. */
export function FactoryBoard({ repo, onOpenTask }: { repo: string; onOpenTask: (taskId: string) => void }) {
  const [tasks, setTasks] = useState<FactoryTask[] | null>(null)
  const [newTaskOpen, setNewTaskOpen] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [showAllDone, setShowAllDone] = useState(false)

  const refresh = async () => setTasks(await fetchFactoryTasks(repo))
  const error = useFactoryPolling(repo, refresh)

  return (
    <ElevationProvider>
      <div className="flex h-full flex-col">
        <div className="flex shrink-0 items-center justify-between border-b border-divider/20 px-8 pb-4 pt-6 select-none">
          <h2 className="text-lg font-semibold tracking-tight text-ink">Factory · {repo}</h2>
          <div className="flex items-center gap-2">
            <button type="button" aria-label="Factory settings" onClick={() => setSettingsOpen(true)} className={ICON_BUTTON_CLASS}>
              <Settings className="h-4 w-4" />
            </button>
            <button type="button" onClick={() => setNewTaskOpen(true)} className={cn(PRIMARY_BUTTON_CLASS, "flex items-center gap-1.5")}>
              <Plus className="h-3.5 w-3.5" />
              New task
            </button>
          </div>
        </div>
        {error && <p role="alert" className="px-8 pt-3 text-xs text-danger">{error}</p>}
        <div className="flex min-h-0 flex-1 gap-4 overflow-x-auto px-8 pb-8 pt-6 scrollbar-thin">
          {BOARD_COLUMNS.map((column) => {
            const columnTasks = (tasks ?? []).filter((task) => column.states.includes(task.state)).sort(newestFirst)
            const capped = column.id === "done" && !showAllDone && columnTasks.length > DONE_LIMIT
            const shown = capped ? columnTasks.slice(0, DONE_LIMIT) : columnTasks
            return (
              <ElevatedContainer
                key={column.id}
                role="region"
                aria-label={column.title}
                className="flex min-h-0 min-w-[15rem] flex-1 flex-col overflow-hidden rounded-2xl border border-divider/40"
              >
                <div className="flex shrink-0 items-center justify-between px-4 pb-3 pt-4 select-none">
                  <div className="flex items-center gap-2">
                    <h3 className="text-[11px] font-bold uppercase tracking-[0.2em] text-ink-muted">{column.title}</h3>
                    <span className="rounded-md bg-surface-elevated px-2 py-0.5 font-mono text-[10px] font-bold tabular-nums text-ink">
                      {columnTasks.length}
                    </span>
                  </div>
                  {column.id === "backlog" && (
                    <button type="button" aria-label="New task" onClick={() => setNewTaskOpen(true)} className={ICON_BUTTON_CLASS}>
                      <Plus className="h-4 w-4" />
                    </button>
                  )}
                </div>
                <div className="flex min-h-0 flex-1 flex-col gap-2.5 overflow-y-auto px-4 pb-4 scrollbar-thin">
                  <AnimatePresence mode="popLayout" initial={false}>
                    {shown.map((task) => (
                      <motion.div
                        key={task.id}
                        layout
                        layoutId={task.id}
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.95 }}
                        transition={{ duration: 0.2 }}
                      >
                        <TaskCard task={task} onOpen={() => onOpenTask(task.id)} />
                      </motion.div>
                    ))}
                  </AnimatePresence>
                  {tasks !== null && columnTasks.length === 0 && (
                    <div className="rounded-xl border border-dashed border-divider/60 px-4 py-6 text-center text-[11px] text-ink-faint">No tasks</div>
                  )}
                  {capped && (
                    <button
                      type="button"
                      onClick={() => setShowAllDone(true)}
                      className="rounded px-1 py-1 text-left text-[11px] text-ink-subtle hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"
                    >
                      Show all {columnTasks.length}
                    </button>
                  )}
                </div>
              </ElevatedContainer>
            )
          })}
        </div>
      </div>
      <NewTaskDialog repo={repo} open={newTaskOpen} onClose={() => setNewTaskOpen(false)} />
      <FactorySettingsDialog open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </ElevationProvider>
  )
}
