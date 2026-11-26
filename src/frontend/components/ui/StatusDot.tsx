import { cn } from "@/lib/utils"

export type StatusTone = "idle" | "active" | "attention" | "error" | "done" | "ready"

const TONE_CLASS: Record<StatusTone, string> = {
  idle: "bg-ink-faint",
  active: "bg-ink-muted animate-pulse",
  attention: "bg-ink",
  error: "bg-danger",
  done: "bg-diff-added",
  ready: "bg-ink-subtle",
}

const TONE_RING: Record<StatusTone, string> = {
  idle: "ring-ink-faint",
  active: "ring-ink-muted animate-pulse",
  attention: "ring-ink",
  error: "ring-danger",
  done: "ring-diff-added",
  ready: "ring-ink-subtle",
}

/** Small state dot: faint when nothing runs, pulsing while something works, solid ink when it waits on the user, danger on failure. `hollow` draws only its ring. */
export function StatusDot({ tone, hollow = false }: { tone: StatusTone; hollow?: boolean }) {
  return <span aria-hidden className={cn("h-1.5 w-1.5 shrink-0 rounded-full", hollow ? cn("ring-1 ring-inset", TONE_RING[tone]) : TONE_CLASS[tone])} />
}
