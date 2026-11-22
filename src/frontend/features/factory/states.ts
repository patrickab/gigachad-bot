// What the factory screens show of a task's state. Consumes a task's `factory` and `live`, provides the shown state,
// its label and tone, the board columns and the actions a state allows.
import type { StatusTone } from "@/components/ui/StatusDot"
import type { FactoryTask, FactoryTaskState } from "@/lib/domain/factory"

/** A task's state as shown: one of the six the session records, or one derived from a missing state or child. */
export type FactoryShownState = FactoryTaskState | "starting" | "interrupted"

/**
 * The state the screens show for `task`.
 *
 * - `starting`: its session has no factory state yet, shown as "Launching session" until the agent works
 * - `interrupted`: the state says running or merging, but no Pi child runs the session
 */
export function shownState(task: Pick<FactoryTask, "factory" | "live">): FactoryShownState {
  if (task.factory === null) return "starting"
  const { state } = task.factory
  return (state === "running" || state === "merging") && task.live === null ? "interrupted" : state
}

export const STATE_LABELS: Record<FactoryShownState, string> = {
  starting: "Launching session",
  running: "Working",
  waiting_approve: "Ready for review",
  blocked: "Blocked",
  interrupted: "Interrupted",
  merging: "Merging",
  done: "Done",
  cancelled: "Cancelled",
}

/** Board order: what waits on the user first, then what runs, then what is finished. */
export const STATE_ORDER: readonly FactoryShownState[] = ["waiting_approve", "blocked", "interrupted", "starting", "running", "merging", "done", "cancelled"]

interface BoardColumn {
  id: string
  title: string
  states: readonly FactoryShownState[]
  /** Tone of the column's dot while it holds a task. */
  tone: StatusTone
}

/** Board columns in display order, every shown state belongs to exactly one. */
export const BOARD_COLUMNS: readonly BoardColumn[] = [
  { id: "needs-you", title: "Needs you", states: ["waiting_approve", "blocked", "interrupted"], tone: "attention" },
  { id: "working", title: "Working", states: ["starting", "running"], tone: "ready" },
  { id: "merging", title: "Merging", states: ["merging"], tone: "ready" },
  { id: "finished", title: "Finished", states: ["done", "cancelled"], tone: "done" },
]

/** States no command but a retry or a message leaves. */
export const FINISHED_STATES: readonly FactoryShownState[] = ["done", "cancelled"]

/** States whose task the backend deletes: it refuses one that runs or merges in a live session. */
export const DELETABLE_STATES: readonly FactoryShownState[] = ["blocked", "interrupted", "done", "cancelled"]

/** How long a task may stay `starting` before it is taken as a session pi-host lost, which can be deleted. */
export const LOST_AFTER_MS = 60_000

/** What the factory accepts for a task shown as `state`, created at `createdAt` (ISO time, for `starting` only). */
export function allowedActions(state: FactoryShownState, createdAt?: string, now = Date.now()) {
  const lost = state === "starting" && createdAt !== undefined && now - Date.parse(createdAt) > LOST_AFTER_MS
  return {
    approve: state === "waiting_approve",
    retry: state === "blocked" || state === "cancelled" || state === "interrupted",
    cancel: state === "running" || state === "merging" || state === "waiting_approve",
    message: state !== "starting",
    delete: lost || DELETABLE_STATES.includes(state),
  }
}

/** One-line status of a task: the shown state, and the round once the first one started. */
export function taskStatusLine(task: Pick<FactoryTask, "factory" | "live">): string {
  const label = STATE_LABELS[shownState(task)]
  const round = task.factory?.round ?? 0
  return round > 0 ? `${label} · round ${round}` : label
}

/** Visual weight of a state: what is running, what waits on the user, what failed. */
export function statusTone(state: FactoryShownState): StatusTone {
  if (state === "starting" || state === "running" || state === "merging") return "active"
  if (state === "waiting_approve") return "attention"
  if (state === "blocked" || state === "interrupted") return "error"
  if (state === "done") return "done"
  return "idle"
}

/** Tone of a task's dot: its state's, pulsing only while its Pi child runs a turn. */
export function taskTone(task: Pick<FactoryTask, "factory" | "live">): StatusTone {
  if (task.live?.running) return "active"
  const tone = statusTone(shownState(task))
  return tone === "active" ? "ready" : tone
}
