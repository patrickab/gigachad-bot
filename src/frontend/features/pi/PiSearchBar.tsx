import { ChevronDown, ChevronUp, X } from "lucide-react"
import { IconButton } from "@/components/ui/IconButton"
import { clearPiSearchHit, stepPiSearchHit, type PiSearchFocus } from "./searchFocus"

/** Position of the shown search hit among its session's, with previous, next and close, at the transcript's top edge. */
export function PiSearchBar({ search }: { search: PiSearchFocus }) {
  return (
    <div className="sticky top-2 z-10 flex h-0 justify-end pr-3">
      <div role="group" aria-label="Search hits" className="flex h-7 items-center rounded-md bg-surface-elevated text-xs tabular-nums text-ink-muted">
        <span className="px-2">{search.index + 1} / {search.matches.length}</span>
        <IconButton icon={ChevronUp} label="Previous hit" onClick={() => stepPiSearchHit(-1)} />
        <IconButton icon={ChevronDown} label="Next hit" onClick={() => stepPiSearchHit(1)} />
        <IconButton icon={X} label="Clear search" onClick={clearPiSearchHit} />
      </div>
    </div>
  )
}
