import { Archive, ArchiveRestore } from "lucide-react"
import { PopoverSelector } from "@/components/ui/PopoverSelector"

export interface ArchiveEntry {
  key: string
  name: string
  archived: boolean
}

/** Header popover of every entry of a section, selecting an archived one restores it. */
export function ArchiveSelector({ noun, entries, onRestore }: { noun: string; entries: ArchiveEntry[]; onRestore: (key: string) => void }) {
  return <PopoverSelector
    triggerLabel={`Show all ${noun}`} triggerContent={<Archive className="h-3 w-3" />}
    triggerClassName="h-auto rounded p-1 text-ink-faint opacity-0 group-hover/row:opacity-100 group-focus-within/row:opacity-100 aria-expanded:opacity-100 focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"
    label={`All ${noun}`} placement="below" panelClassName="left-auto right-0 w-56"
    categories={[{ key: noun, label: `All ${noun}`, items: entries }]}
    itemKey={(entry) => entry.key} isSelected={(entry) => !entry.archived}
    renderItem={(entry) => <>
      <span className="min-w-0 flex-1 truncate">{entry.name}</span>
      {entry.archived ? <span className="flex shrink-0 items-center gap-1.5 text-xs text-ink-subtle"><ArchiveRestore className="h-3 w-3" />Restore</span>
        : <span className="text-xs text-ink-faint">Active</span>}
    </>}
    onSelect={(entry) => { if (entry.archived) onRestore(entry.key) }}
    emptyText={`No ${noun}`}
  />
}
