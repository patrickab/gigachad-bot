import { ViewSwitch } from "@/components/ui/ViewSwitch"
import type { EditorView } from "./useEditorView"

export function ViewPills({ view, onViewChange }: { view: EditorView; onViewChange: (v: EditorView) => void }) {
  return (
    <ViewSwitch
      value={view}
      onChange={onViewChange}
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
      value={view}
      onChange={onViewChange}
      options={[["preview", "Render"], ["edit", "Text"]]}
      className="flex gap-px rounded bg-surface p-px"
      buttonClass="rounded px-1.5 py-px transition-colors"
      inactiveClass="hover:text-ink"
    />
  )
}
