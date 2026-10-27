import { PanelLeft, PanelLeftClose, PanelRight, PanelRightClose } from "lucide-react"

const ICONS = {
  left: { open: PanelLeftClose, closed: PanelLeft },
  right: { open: PanelRightClose, closed: PanelRight },
}

/** Icon button toggling the panel on `side`: its close icon while `open`, its open icon otherwise. */
export function PanelToggleButton({ open, onClick, openLabel, closeLabel, side = "right" }: {
  open: boolean
  onClick: () => void
  openLabel: string
  closeLabel: string
  side?: "left" | "right"
}) {
  const Icon = open ? ICONS[side].open : ICONS[side].closed
  return (
    <button
      aria-label={open ? closeLabel : openLabel}
      onClick={() => onClick()}
      className="shrink-0 rounded p-1 text-ink-subtle hover:text-ink hover:bg-hover transition-colors"
    >
      <Icon className="h-4 w-4" />
    </button>
  )
}
