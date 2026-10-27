import { cn } from "@/lib/utils"
import type { EditorView } from "./useEditorView"

function ViewSwitch({ view, onViewChange, options, className, buttonClass, inactiveClass }: {
  view: EditorView
  onViewChange: (v: EditorView) => void
  options: readonly (readonly [EditorView, string])[]
  className: string
  buttonClass: string
  inactiveClass: string
}) {
  return (
    <div role="group" aria-label="Show as" className={className}>
      {options.map(([option, label]) => (
        <button
          key={option}
          type="button"
          aria-pressed={view === option}
          onClick={() => onViewChange(option)}
          className={cn(buttonClass, view === option ? "bg-surface-elevated text-ink" : inactiveClass)}
        >
          {label}
        </button>
      ))}
    </div>
  )
}

export function ViewPills({ view, onViewChange }: { view: EditorView; onViewChange: (v: EditorView) => void }) {
  return (
    <ViewSwitch
      view={view}
      onViewChange={onViewChange}
      options={[["edit", "Edit"], ["preview", "Preview"]]}
      className="absolute top-2 left-3 z-10 flex items-center gap-px rounded-md bg-surface/80 backdrop-blur-sm p-0.5 opacity-0 hover:opacity-100 transition-opacity"
      buttonClass="px-2 py-0.5 rounded text-[10px] font-medium transition-colors"
      inactiveClass="text-ink-subtle hover:text-ink"
    />
  )
}

/** C4 "Show as" switch for tool artifacts: rendered view first, source text second. */
export function ShowAsToggle({ view, onViewChange }: { view: EditorView; onViewChange: (v: EditorView) => void }) {
  return (
    <ViewSwitch
      view={view}
      onViewChange={onViewChange}
      options={[["preview", "Render"], ["edit", "Text"]]}
      className="flex gap-px rounded bg-surface p-px"
      buttonClass="rounded px-1.5 py-px transition-colors"
      inactiveClass="hover:text-ink"
    />
  )
}
