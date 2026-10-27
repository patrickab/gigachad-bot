"use client"

import { useEffect, useState } from "react"
import { ArrowLeft } from "lucide-react"
import { ErrorText } from "@/components/ui/ErrorText"
import { SidePanelFrame } from "@/components/ui/SidePanel"
import { DiffViewer } from "@/features/editor/diff/DiffViewer"
import { discardReviewNotes } from "@/features/editor/diff/review"
import { cn } from "@/lib/utils"
import { fetchFactoryTask, sendFactoryMessage } from "@/lib/backend/api"
import { AgentPanel } from "./AgentPanel"
import { PreviewPane } from "./PreviewPane"
import { FINISHED_STATES, allowedActions, statusTone, taskStatusLine } from "./states"
import { StatusDot } from "./StatusDot"
import { ICON_BUTTON_CLASS } from "./styles"
import type { FactoryTaskDetail } from "@/lib/domain/types"
import { useFactoryPolling } from "./useFactoryPolling"

type TaskPane = "preview" | "changes"

const PANES: readonly { id: TaskPane; label: string }[] = [
  { id: "preview", label: "Preview" },
  { id: "changes", label: "Changes" },
]

const DEFAULT_PANEL_WIDTH = 380

/** Changes only move when a phase run ends, so the diff reloads then and keeps its scroll position in between. */
function changesKey(task: FactoryTaskDetail): string {
  return `${task.worktree}:${task.phases.filter((phaseRun) => phaseRun.status !== "running").length}`
}

/** One task: preview or changes on the left, the agent panel on the right. */
export function TaskView({ repo, taskId, onBack }: { repo: string; taskId: string; onBack: () => void }) {
  const [task, setTask] = useState<FactoryTaskDetail | null>(null)
  const [chosenPane, setChosenPane] = useState<TaskPane | null>(null)
  const [panelWidth, setPanelWidth] = useState(DEFAULT_PANEL_WIDTH)

  const refresh = async (changedTaskIds: ReadonlySet<string> | null) => {
    if (changedTaskIds === null || changedTaskIds.has(taskId)) setTask(await fetchFactoryTask(taskId))
  }
  const error = useFactoryPolling(repo, refresh)
  const reload = () => { refresh(null).catch(() => {}) }

  // A task that was opened by id may be another's: ignore detail loaded for a previous id.
  const current = task?.id === taskId ? task : null
  const pane: TaskPane = chosenPane ?? (current?.preview ? "preview" : "changes")
  const hasWorktree = current !== null && current.worktree !== null && !FINISHED_STATES.includes(current.state)
  const finishedWorktree = current?.worktree && FINISHED_STATES.includes(current.state) ? current.worktree : null

  // Pending review notes outlive the diff view, but not the worktree they point into.
  useEffect(() => {
    if (finishedWorktree) discardReviewNotes(finishedWorktree)
  }, [finishedWorktree])

  return (
    <div className="flex h-full min-h-0">
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex h-12 shrink-0 items-center gap-3 border-b border-divider/20 px-4">
          <button type="button" aria-label="Back to board" onClick={onBack} className={ICON_BUTTON_CLASS}>
            <ArrowLeft className="h-4 w-4" />
          </button>
          <h2 className="min-w-0 flex-1 truncate text-sm font-semibold tracking-tight text-ink">{current?.title ?? "Task"}</h2>
          {current && (
            <span className="flex shrink-0 items-center gap-2 text-[11px] text-ink-subtle">
              <StatusDot tone={statusTone(current.state)} />
              {taskStatusLine(current.state, current.phase, current.round)}
            </span>
          )}
          <div role="tablist" aria-label="Task view" className="flex shrink-0 gap-1">
            {PANES.map(({ id, label }) => (
              <button
                key={id}
                id={`task-tab-${id}`}
                type="button"
                role="tab"
                aria-selected={pane === id}
                aria-controls="task-pane"
                onClick={() => setChosenPane(id)}
                className={cn(
                  "rounded-full px-3 py-1 text-[11px] font-medium transition-colors focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted",
                  pane === id ? "bg-surface-elevated text-ink" : "text-ink-muted hover:bg-surface/50 hover:text-ink",
                )}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
        <ErrorText className="px-3 py-1">{error}</ErrorText>
        <div id="task-pane" role="tabpanel" aria-labelledby={`task-tab-${pane}`} className="min-h-0 flex-1">
          {!current ? (
            <div className="flex h-full items-center justify-center text-xs text-ink-faint">Loading task</div>
          ) : pane === "preview" ? (
            <PreviewPane task={current} onChanged={reload} />
          ) : hasWorktree ? (
            <DiffViewer
              key={changesKey(current)}
              root={current.worktree!}
              review={{
                canSend: allowedActions(current.state).message,
                send: async (text) => {
                  await sendFactoryMessage(current.id, text)
                  reload()
                },
              }}
            />
          ) : (
            <div className="flex h-full items-center justify-center text-xs text-ink-faint">No worktree to show changes for</div>
          )}
        </div>
      </div>
      {current && (
        <SidePanelFrame width={panelWidth} onWidthChange={setPanelWidth} className="border-l border-divider/50 bg-surface">
          <AgentPanel task={current} onChanged={reload} />
        </SidePanelFrame>
      )}
    </div>
  )
}
