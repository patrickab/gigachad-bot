"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { GitBranch, Plus, Settings, Trash2 } from "lucide-react"
import { BoardColumn } from "@/components/ui/BoardColumn"
import { ElevatedContainer, ElevationProvider } from "@/components/ui/ElevatedContainer"
import { ErrorText } from "@/components/ui/ErrorText"
import { IconButton } from "@/components/ui/IconButton"
import { InlineConfirm } from "@/components/ui/InlineConfirm"
import { StatusDot } from "@/components/ui/StatusDot"
import { useAsyncAction } from "@/hooks/useAsyncAction"
import { deleteFactoryTask } from "@/lib/backend/api"
import type { FactoryTask } from "@/lib/domain/factory"
import { baseName } from "@/lib/paths"
import { cn } from "@/lib/utils"
import { FactorySettingsDialog } from "./FactorySettingsDialog"
import { NewTaskDialog } from "./NewTaskDialog"
import { BOARD_COLUMNS, STATE_ORDER, allowedActions, shownState, taskStatusLine, taskTone } from "./states"
import { ICON_BUTTON_CLASS, PRIMARY_BUTTON_CLASS } from "./styles"

/** Finished tasks shown before "Show all", the column otherwise grows without bound. */
const FINISHED_LIMIT = 10

/** Column order of two tasks: by state as the board orders them, then the one whose state changed last. */
function boardOrder(left: FactoryTask, right: FactoryTask): number {
  const byState = STATE_ORDER.indexOf(shownState(left)) - STATE_ORDER.indexOf(shownState(right))
  return byState || (right.factory?.at ?? right.created_at).localeCompare(left.factory?.at ?? left.created_at)
}

function TaskCard({ task, onOpen, onDelete }: { task: FactoryTask; onOpen: () => void; onDelete: () => Promise<unknown> }) {
  const [confirming, setConfirming] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const state = shownState(task)
  const failed = state === "blocked" || state === "interrupted"
  const remove = async () => {
    setDeleting(true)
    await onDelete()
    setDeleting(false)
    setConfirming(false)
  }
  return (
    <ElevatedContainer hoverLift className="group relative rounded-xl border border-divider/40 shadow-[var(--shadow-md)]">
      <button
        type="button"
        onClick={onOpen}
        className="flex w-full flex-col gap-2 rounded-xl p-4 text-left focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"
      >
        <span className="line-clamp-2 pr-6 text-sm font-semibold text-ink">{task.title}</span>
        <span className="flex min-w-0 items-center gap-2 text-[11px]">
          <StatusDot tone={taskTone(task)} />
          <span className={cn("truncate", failed ? "text-danger" : "text-ink-subtle")}>{taskStatusLine(task)}</span>
        </span>
        {state === "blocked" && task.factory?.reason && <span className="truncate text-[11px] text-ink-subtle">{task.factory.reason}</span>}
        {/* The delete question covers this row, so asking never resizes the card. */}
        <span className={cn("flex h-6 min-w-0 items-center gap-1 text-[11px] text-ink-faint", confirming && "invisible")}>
          {task.branch !== null && <GitBranch aria-hidden className="h-3 w-3 shrink-0" />}
          <span className="truncate">{task.branch ?? "Current checkout"}</span>
        </span>
      </button>
      {allowedActions(state, task.created_at).delete && !confirming && (
        <IconButton
          icon={Trash2}
          label={`Delete ${task.title}`}
          onClick={() => setConfirming(true)}
          className="absolute right-2 top-2 opacity-0 focus-visible:opacity-100 group-hover:opacity-100"
        />
      )}
      {confirming && (
        <div className="absolute inset-x-4 bottom-4 grid h-6 items-center">
          <InlineConfirm question="Delete this task?" busy={deleting} onYes={() => void remove()} onNo={() => setConfirming(false)} />
        </div>
      )}
    </ElevatedContainer>
  )
}

/** Kanban board of the tasks of the repo at path `repo`: what waits on the user first, then what runs, then what is finished. */
export function FactoryBoard({ repo, tasks, error, refresh, onOpenTask }: {
  repo: string
  tasks: FactoryTask[]
  /** Why the last read of the tasks failed, null when it did not. */
  error: string | null
  /** Read the tasks again, after a change this board made. */
  refresh: () => Promise<void>
  onOpenTask: (taskId: string) => void
}) {
  const [newTaskOpen, setNewTaskOpen] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [showAllFinished, setShowAllFinished] = useState(false)
  const { run: deleteTask, error: deleteError } = useAsyncAction(async (id: string) => {
    await deleteFactoryTask(id)
    await refresh()
  }, "Could not delete the task")

  return (
    // Lanes start at --surface, the level of the sidebar beside them
    <ElevationProvider base={1}>
      <div className="flex h-full flex-col">
        <div className="flex shrink-0 items-center justify-between px-8 pb-2 pt-6 select-none">
          <h2 className="text-lg font-semibold tracking-tight text-ink">Factory <span className="font-normal text-ink-muted">· {baseName(repo)}</span></h2>
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
        <ErrorText className="px-8 pt-3">{error ?? deleteError}</ErrorText>
        <div className="flex min-h-0 flex-1 gap-4 overflow-x-auto px-8 pb-8 pt-4 scrollbar-thin">
          {BOARD_COLUMNS.map((column) => {
            const columnTasks = tasks.filter((task) => column.states.includes(shownState(task))).sort(boardOrder)
            const capped = column.id === "finished" && !showAllFinished && columnTasks.length > FINISHED_LIMIT
            const shown = capped ? columnTasks.slice(0, FINISHED_LIMIT) : columnTasks
            return (
              <BoardColumn
                key={column.id}
                role="region"
                aria-label={column.title}
                title={column.title}
                titleAs="h3"
                count={columnTasks.length}
                marker={<StatusDot tone={columnTasks.length === 0 ? "idle" : column.tone} />}
                className={cn("min-w-[15rem] border-divider/30", column.id === "finished" && "opacity-60 hover:opacity-100 focus-within:opacity-100")}
                bodyProps={{ className: "flex min-h-0 flex-1 flex-col gap-2.5" }}
                footer={<>
                  {capped && (
                    <button
                      type="button"
                      onClick={() => setShowAllFinished(true)}
                      className="rounded px-1 py-1 text-left text-[11px] text-ink-subtle hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"
                    >
                      Show all {columnTasks.length}
                    </button>
                  )}
                </>}
              >
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
                    <TaskCard task={task} onOpen={() => onOpenTask(task.id)} onDelete={() => deleteTask(task.id)} />
                  </motion.div>
                ))}
              </BoardColumn>
            )
          })}
        </div>
      </div>
      <NewTaskDialog repo={repo} open={newTaskOpen} onClose={() => setNewTaskOpen(false)} onCreated={() => void refresh()} />
      <FactorySettingsDialog repo={repo} open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </ElevationProvider>
  )
}
