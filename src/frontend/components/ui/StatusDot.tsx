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

/** Small state dot: faint when nothing runs, pulsing while something works, solid ink when it waits on the user, danger on failure. */
export function StatusDot({ tone }: { tone: StatusTone }) {
  return <span aria-hidden className={cn("h-1.5 w-1.5 shrink-0 rounded-full", TONE_CLASS[tone])} />
}
