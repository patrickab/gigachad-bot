"use client"

import { useState } from "react"
import { ArrowLeft } from "lucide-react"
import { ErrorText } from "@/components/ui/ErrorText"
import { InlineConfirm } from "@/components/ui/InlineConfirm"
import { PanelToggleButton } from "@/components/ui/PanelToggleButton"
import { PillTabs } from "@/components/ui/PillTabs"
import { StatusDot } from "@/components/ui/StatusDot"
import { useTabWorkingDirectory } from "@/contexts/tabWorkingDirectory"
import { useAsyncAction } from "@/hooks/useAsyncAction"
import { callFactory } from "@/lib/backend/api"
import type { FactoryTask, FactoryTaskAction, FactoryTaskCommand } from "@/lib/domain/factory"
import { cn } from "@/lib/utils"
import { AgentPanel } from "./AgentPanel"
import { PreviewPane } from "./PreviewPane"
import { TaskHandoff, handoffOf } from "./TaskHandoff"
import { TaskSession } from "./TaskSession"
import { shownState, statusTone, taskStatusLine } from "./states"
import { ICON_BUTTON_CLASS, PRIMARY_BUTTON_CLASS, QUIET_BUTTON_CLASS } from "./styles"
import { TaskDiff } from "./TaskDiff"
import { useTaskReview } from "./taskReview"
import { useFactoryTask, useFactoryTasks } from "./useFactoryTasks"
import { useTaskDiff } from "./useTaskDiff"
import { useTaskSession } from "./useTaskSession"
import { useTaskTimeline } from "./useTaskTimeline"

type TaskPane = "session" | "handoff" | "changes" | "preview"

const PANES: readonly { value: TaskPane; label: string }[] = [
  { value: "session", label: "Session" },
  { value: "handoff", label: "Handoff" },
  { value: "changes", label: "Changes" },
  { value: "preview", label: "Preview" },
]

/** The commands the header offers: those of the task `actions` the core accepts now. */
export function taskCommands(actions: readonly FactoryTaskAction[]): Record<FactoryTaskCommand, boolean> {
  return { approve: actions.includes("tasks.approve"), retry: actions.includes("tasks.retry"), cancel: actions.includes("tasks.cancel") }
}

/** What moves when the task's session grows: its state, its runner and its usage. */
function taskRevision(task: FactoryTask | undefined): string {
  if (!task) return ""
  return `${task.state ?? ""}|${task.round ?? ""}|${task.at ?? ""}|${Number(task.running)}|${task.usage?.output ?? ""}`
}

const commandFailure = (command: FactoryTaskCommand) => `Could not ${command} the task`

/**
 * One task: its diff as the surface (or its preview), its timeline and a message box to its coder in the tab's file panel.
 *
 * - The task comes from its repo's task list (`useFactoryTasks` keeps it current), undefined until the list has it
 * - The timeline is read on whenever the task moves and, while a runner holds it, on a poll; the diff whenever the
 *   timeline grows
 * - A command or message answers the task as the core has it then, the list is read again after it
 * - Review comments on the diff go out as one message with "Send review", the round it opens arrives the same way
 * - `onOpenTask` opens another task of the repo: one the coder started from the timeline
 */
export function TaskView({ repo, taskId, onBack, onOpenTask }: {
  repo: string
  taskId: string
  onBack: () => void
  onOpenTask: (taskId: string) => void
}) {
  const { error: tasksError, refresh } = useFactoryTasks(repo)
  const task = useFactoryTask(taskId)
  useTabWorkingDirectory(task?.worktree)
  const timeline = useTaskTimeline(taskId, taskRevision(task), task?.running ?? false)
  const diff = useTaskDiff(taskId, timeline.entries.length, task !== undefined && task.state !== null)
  const session = useTaskSession(task?.sessionId ?? "", taskRevision(task), task?.running ?? false)
  const handoff = task?.running ? undefined : handoffOf(timeline.entries)
  const [chosen, setPane] = useState<TaskPane>("session")
  const pane = chosen === "handoff" && handoff === undefined ? "session" : chosen
  const panes = PANES.filter((one) => one.value !== "handoff" || handoff !== undefined)
  const [panelOpen, setPanelOpen] = useState(true)
  const [confirmingCancel, setConfirmingCancel] = useState(false)
  const { run, pending, error: commandError } = useAsyncAction(async (command: FactoryTaskCommand) => {
    try {
      await callFactory<FactoryTask>(`tasks.${command}`, { id: taskId })
      await refresh()
    } finally {
      setConfirmingCancel(false)
    }
  }, commandFailure)

  const state = task && shownState(task)
  const commands = task && taskCommands(task.actions)
  const canMessage = task?.actions.includes("tasks.message") ?? false
  const review = useTaskReview(taskId, state, canMessage, diff.files)

  return (
    <div className="flex h-full min-h-0">
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex h-12 shrink-0 items-center gap-3 border-b border-divider/20 px-4">
          <button type="button" aria-label="Back to board" onClick={onBack} className={ICON_BUTTON_CLASS}>
            <ArrowLeft className="h-4 w-4" />
          </button>
          <h2 className="min-w-0 flex-1 truncate text-sm font-semibold tracking-tight text-ink">{task?.title ?? "Task"}</h2>
          {task && state && (
            <span aria-label="Task state" className="flex min-w-0 shrink items-center gap-2 text-[11px] text-ink-subtle">
              <StatusDot tone={statusTone(state)} />
              <span className="truncate">{taskStatusLine(task)} · {task.branch}</span>
            </span>
          )}
          {review.notes.length > 0 && <button
            type="button"
            disabled={!review.canSend || review.sending}
            onClick={() => void review.send()}
            className={cn(PRIMARY_BUTTON_CLASS, "shrink-0")}
          >Send review <span className="tabular-nums opacity-70">{review.notes.length}</span></button>}
          {commands && (confirmingCancel
            ? <div className="w-48 shrink-0"><InlineConfirm question="Cancel this task?" busy={pending} onYes={() => void run("cancel")} onNo={() => setConfirmingCancel(false)} /></div>
            : <>
              {commands.approve && <button type="button" disabled={pending} onClick={() => void run("approve")} className={PRIMARY_BUTTON_CLASS}>Approve</button>}
              {commands.retry && <button type="button" disabled={pending} onClick={() => void run("retry")} className={PRIMARY_BUTTON_CLASS}>Retry</button>}
              {commands.cancel && <button type="button" disabled={pending} onClick={() => setConfirmingCancel(true)} className={QUIET_BUTTON_CLASS}>Cancel</button>}
            </>)}
          <PillTabs
            className="shrink-0"
            options={panes}
            value={pane}
            onChange={setPane}
            tabs={{ label: "Task view", idPrefix: "task-tab", panelId: "task-pane" }}
          />
          {!panelOpen && <PanelToggleButton open={false} onClick={() => setPanelOpen(true)} openLabel="Open agent" closeLabel="Close agent" />}
        </div>
        <ErrorText className="px-3 py-1">{commandError || review.sendError || tasksError || diff.error || session.error}</ErrorText>
        <div id="task-pane" role="tabpanel" aria-labelledby={`task-tab-${pane}`} className="min-h-0 flex-1">
          {!task ? (
            <div className="flex h-full items-center justify-center text-xs text-ink-faint">Loading task</div>
          ) : pane === "preview" ? (
            <PreviewPane taskId={task.id} />
          ) : pane === "session" ? (
            <TaskSession transcript={session.transcript} />
          ) : pane === "handoff" && handoff !== undefined ? (
            <TaskHandoff task={task} text={handoff} files={diff.files ?? []} onShowChanges={() => setPane("changes")} />
          ) : (
            <TaskDiff files={diff.files} root={task.worktree} review={review} />
          )}
        </div>
      </div>
      {task && state && panelOpen && <AgentPanel
        task={task}
        entries={timeline.entries}
        timelineError={timeline.error}
        canMessage={canMessage}
        onSent={() => void refresh()}
        onOpenTask={onOpenTask}
        onClose={() => setPanelOpen(false)}
      />}
    </div>
  )
}
