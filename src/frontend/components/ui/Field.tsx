import type { ReactNode } from "react"

const FIELD_SHAPES = {
  /** Dialog forms such as the factory dialogs. */
  regular: { wrapper: "flex flex-col gap-1", label: "text-[11px] text-ink-subtle" },
  /** Dense popover panels such as the settings menu. */
  compact: { wrapper: "space-y-1", label: "text-[10px] text-ink-faint" },
} as const

/** Labelled form control: a small caption above `children`. */
export function Field({ label, children, compact = false }: { label: string; children: ReactNode; compact?: boolean }) {
  const shape = FIELD_SHAPES[compact ? "compact" : "regular"]
  return (
    <div className={shape.wrapper}>
      <span className={shape.label}>{label}</span>
      {children}
    </div>
  )
}
