import type { Phase, TaskState } from "@/lib/domain/factory"

interface BoardColumn {
  id: string
  title: string
  states: readonly TaskState[]
}

/** Board columns in display order; cancelled tasks belong to none and stay hidden. */
export const BOARD_COLUMNS: readonly BoardColumn[] = [
  { id: "backlog", title: "Backlog", states: ["backlog"] },
  { id: "todo", title: "Todo", states: ["todo"] },
  { id: "working", title: "Working", states: ["in_progress", "rework"] },
  { id: "needs-you", title: "Needs you", states: ["human_review", "blocked"] },
  { id: "merging", title: "Merging", states: ["merging"] },
  { id: "done", title: "Done", states: ["done"] },
]

export const STATE_LABELS: Record<TaskState, string> = {
  backlog: "Backlog",
  todo: "Queued",
  in_progress: "Working",
  human_review: "Ready for review",
  rework: "Reworking",
  blocked: "Blocked",
  merging: "Merging",
  done: "Done",
  cancelled: "Cancelled",
}

export const PHASE_LABELS: Record<Phase, string> = {
  prepare: "Prepare",
  scout: "Scout",
  plan: "Plan",
  code: "Code",
  hooks: "Hooks",
  gates: "Gates",
  review: "Review",
  preview: "Preview",
  merge: "Merge",
}

/** States whose worktree no longer exists, so there is nothing to diff or preview. */
export const FINISHED_STATES: readonly TaskState[] = ["done", "cancelled"]

/** Actions the engine allows for a task in `state`. */
export function allowedActions(state: TaskState) {
  return {
    start: state === "backlog",
    approve: state === "human_review",
    retry: state === "blocked",
    cancel: !FINISHED_STATES.includes(state),
    message: state === "human_review" || state === "blocked",
    restartPreview: state === "human_review",
  }
}

/** One-line status of a task card: state, plus the running phase and round while the agent works. */
export function taskStatusLine(state: TaskState, phase: Phase | null, round: number): string {
  const working = state === "in_progress" || state === "rework"
  if (!working || !phase) return STATE_LABELS[state]
  return `${STATE_LABELS[state]} · ${PHASE_LABELS[phase]}${round > 0 ? ` · round ${round}` : ""}`
}

export type StatusTone = "idle" | "active" | "attention" | "error" | "done"

/** Visual weight of a state: what is running, what waits on the user, what failed. */
export function statusTone(state: TaskState): StatusTone {
  if (state === "in_progress" || state === "rework" || state === "merging") return "active"
  if (state === "human_review") return "attention"
  if (state === "blocked") return "error"
  if (state === "done") return "done"
  return "idle"
}

/** Pipeline phases a running phase counts toward: hooks and gates belong to code. */
const PHASE_OWNER: Partial<Record<Phase, string>> = { hooks: "code", gates: "code" }

/** Workflow phases of a task and how many are finished, `active` is the index of the running one or -1. */
export function phaseProgress(phases: readonly string[], state: TaskState, phase: Phase | null): { finished: number; active: number } {
  if (state === "backlog" || state === "todo" || state === "cancelled") return { finished: 0, active: -1 }
  const working = state === "in_progress" || state === "rework" || state === "blocked"
  if (!working || phase === null || phase === "preview") return { finished: phases.length, active: -1 }
  if (phase === "prepare") return { finished: 0, active: -1 }
  const active = phases.indexOf(PHASE_OWNER[phase] ?? phase)
  return active === -1 ? { finished: phases.length, active: -1 } : { finished: active, active }
}
