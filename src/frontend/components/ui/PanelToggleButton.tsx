import { PanelRight, PanelRightClose } from "lucide-react"

/** Icon button toggling a right-hand panel: PanelRightClose while `open`, PanelRight otherwise. */
export function PanelToggleButton({ open, onClick, openLabel, closeLabel }: {
  open: boolean
  onClick: () => void
  openLabel: string
  closeLabel: string
}) {
  return (
    <button
      aria-label={open ? closeLabel : openLabel}
      onClick={() => onClick()}
      className="shrink-0 rounded p-1 text-ink-subtle hover:text-ink hover:bg-hover transition-colors"
    >
      {open ? <PanelRightClose className="h-4 w-4" /> : <PanelRight className="h-4 w-4" />}
    </button>
  )
}
