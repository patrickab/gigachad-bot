import { ChevronsLeftRight, ChevronsRightLeft } from "lucide-react"
import { InTopBar } from "@/features/navigation/TopBarSlots"

/** The top bar's wide mode toggle of a conversation column (a chat, a Pi tab), shown on hover or focus. Nothing where the bar has no `wideToggle` slot. */
export function WideModeToggle({ wide, onToggle }: { wide: boolean; onToggle: () => void }) {
  return (
    <InTopBar name="wideToggle">
      <button
        type="button"
        onClick={onToggle}
        aria-label={wide ? "Exit wide mode" : "Enter wide mode"}
        className="rounded-lg p-1.5 text-ink-subtle opacity-0 transition-opacity duration-200 hover:text-ink hover:opacity-100 focus-visible:opacity-100"
      >
        {wide ? <ChevronsRightLeft className="h-4 w-4" /> : <ChevronsLeftRight className="h-4 w-4" />}
      </button>
    </InTopBar>
  )
}
