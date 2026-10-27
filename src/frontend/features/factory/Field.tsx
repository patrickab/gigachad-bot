import type { ReactNode } from "react"

/** Labelled form control of the factory dialogs. */
export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[11px] text-ink-subtle">{label}</span>
      {children}
    </div>
  )
}
