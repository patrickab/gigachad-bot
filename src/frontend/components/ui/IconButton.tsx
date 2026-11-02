import type { LucideIcon } from "lucide-react"
import { cn } from "@/lib/utils"

interface IconButtonProps {
  icon: LucideIcon
  /** Accessible name, the button has no visible text. */
  label: string
  onClick: () => void
  disabled?: boolean
  /** Merged over the quiet default look, e.g. another hover fill on a raised panel. */
  className?: string
}

/** Small square icon button: quiet ink, a hover fill, a keyboard focus ring and a dimmed disabled state. */
export function IconButton({ icon: Icon, label, onClick, disabled, className }: IconButtonProps) {
  return <button
    type="button"
    aria-label={label}
    disabled={disabled}
    onClick={onClick}
    className={cn("flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-ink-subtle transition-colors hover:bg-hover hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted disabled:pointer-events-none disabled:opacity-40", className)}
  >
    <Icon className="h-3.5 w-3.5" aria-hidden="true" />
  </button>
}
