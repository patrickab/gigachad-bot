import { cn } from "@/lib/utils"
import type { StatusTone } from "./states"

const TONE_CLASS: Record<StatusTone, string> = {
  idle: "bg-ink-faint",
  active: "bg-ink-muted animate-pulse",
  attention: "bg-ink",
  error: "bg-danger",
  done: "bg-diff-added",
}

/** Small state dot: pulses while an agent works, solid ink when the task waits on the user. */
export function StatusDot({ tone }: { tone: StatusTone }) {
  return <span aria-hidden className={cn("h-1.5 w-1.5 shrink-0 rounded-full", TONE_CLASS[tone])} />
}
