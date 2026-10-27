import type { ReactNode } from "react"

/** `fill` fits a small embedded frame (full height, tiny text), the default pads a document view. */
type LoadStateSize = "panel" | "fill"

/** Load error with a Retry button. */
export function LoadFailure({ message, onRetry, size = "panel" }: { message: ReactNode; onRetry: () => void; size?: LoadStateSize }) {
  const className = size === "fill"
    ? "flex h-full flex-col items-center justify-center gap-2 text-[10px] text-ink-faint"
    : "flex flex-col items-center justify-center gap-2 py-6 text-xs text-ink-faint"
  return (
    <div className={className}>
      <span>{message}</span>
      <button onClick={() => onRetry()} className="rounded px-2 py-1 text-ink-subtle hover:text-ink hover:bg-hover transition-colors">
        Retry
      </button>
    </div>
  )
}

/** Placeholder while a document loads. */
export function LoadingPlaceholder({ size = "panel" }: { size?: LoadStateSize }) {
  return size === "fill"
    ? <div className="flex h-full items-center justify-center text-[10px] text-ink-faint">Loading…</div>
    : <div className="flex items-center justify-center py-6 text-xs text-ink-faint">Loading...</div>
}
