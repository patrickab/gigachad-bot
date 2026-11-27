// What the factory screens show of a task's state. Consumes the core's task (`state`, `interrupted`, `running`),
// provides the shown state, its label and tone and the board columns. Which actions a task takes comes from the
// core (`FactoryTask.actions`).
import type { StatusTone } from "@/components/ui/StatusDot"
import type { FactoryTask, FactoryTaskState } from "@/lib/domain/factory"

/** A task's state as shown: one of the factory's, or `starting` (none yet) or `interrupted`. */
export type FactoryShownState = FactoryTaskState | "starting" | "interrupted"

/**
 * The state the screens show for `task`.
 *
 * - `starting`: its session has no factory state yet, shown as "Launching session" until the agent works
 * - `interrupted`: the core reports the task interrupted (running or merging, no runner lives)
 */
export function shownState(task: Pick<FactoryTask, "state" | "interrupted">): FactoryShownState {
  if (task.state === null) return "starting"
  return task.interrupted ? "interrupted" : task.state
}

export const STATE_LABELS: Record<FactoryShownState, string> = {
  starting: "Launching session",
  running: "Working",
  waiting_approve: "Ready for review",
  waiting_answer: "Waiting for supervisor",
  reviewing: "In wave review",
  blocked: "Blocked",
  interrupted: "Interrupted",
  merging: "Merging",
  done: "Done",
  cancelled: "Cancelled",
}

interface BoardColumn {
  id: string
  title: string
  states: readonly FactoryShownState[]
  /** Tone of the column's dot while it holds a task. */
  tone: StatusTone
}

/** States that wait on the supervisor or the user: the Waiting block at the top of Working. */
export const WAITING_STATES: readonly FactoryShownState[] = ["waiting_answer", "waiting_approve", "blocked", "interrupted"]

/** Board columns in display order, every shown state belongs to exactly one. Working holds the waiting states too, in its Waiting block. */
export const BOARD_COLUMNS: readonly BoardColumn[] = [
  { id: "working", title: "Working", states: [...WAITING_STATES, "starting", "running", "reviewing"], tone: "ready" },
  { id: "merging", title: "Merging", states: ["merging"], tone: "ready" },
  { id: "finished", title: "Finished", states: ["done", "cancelled"], tone: "done" },
]

/** States of a finished task: its diff no longer moves. */
export const FINISHED_STATES: readonly FactoryShownState[] = ["done", "cancelled"]

/** One-line status of a task: the shown state, and the round once the first one started. */
export function taskStatusLine(task: Pick<FactoryTask, "state" | "interrupted" | "round">): string {
  const label = STATE_LABELS[shownState(task)]
  const round = task.round ?? 0
  return round > 0 ? `${label} · round ${round}` : label
}

/** Visual weight of a state: what is running, what waits on the user, what failed. */
export function statusTone(state: FactoryShownState): StatusTone {
  if (state === "starting" || state === "running" || state === "reviewing" || state === "merging") return "active"
  if (state === "waiting_approve" || state === "waiting_answer") return "attention"
  if (state === "blocked" || state === "interrupted") return "error"
  if (state === "done") return "done"
  return "idle"
}

/** Tone of a task's dot: its state's, pulsing only while a runner holds the task. */
export function taskTone(task: Pick<FactoryTask, "state" | "interrupted" | "running">): StatusTone {
  if (task.running) return "active"
  const tone = statusTone(shownState(task))
  return tone === "active" ? "ready" : tone
}
