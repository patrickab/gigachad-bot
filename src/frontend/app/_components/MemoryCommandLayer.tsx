"use client"

import { CommandMenu, type CommandMenuItem } from "@/features/navigation/CommandMenu"
import { ContentCentered } from "@/components/ui/ContentArea"
import { MemoryPanel } from "@/features/memory/MemoryPanel"
import { useTab } from "@/contexts/TabContext"
import type { MemoryCommandBar } from "@/features/memory/useMemoryCommandBar"

/** Show workspace commands independently of sidebar navigation, plus memory review. */
export function MemoryCommandLayer({ bar, items, onRun }: {
  bar: MemoryCommandBar
  items: CommandMenuItem[]
  onRun: (command: string) => void
}) {
  const { isActive } = useTab()

  return (
    <>
      {isActive && bar.menuOpen && <div className="absolute inset-0 z-[70]" onClick={bar.closeMenu}>
        <ContentCentered className="top-14 px-4">
          <div role="dialog" aria-label="Commands" className="w-80 max-w-full overflow-hidden rounded-xl border border-divider bg-surface p-2 shadow-[var(--shadow-lg)]" onClick={(event) => event.stopPropagation()}>
            <input
              autoFocus
              aria-label="Search commands"
              placeholder="Find a command…"
              value={bar.query}
              onChange={(event) => bar.setQuery(event.target.value)}
              spellCheck={false}
              className="w-full rounded-md bg-paper px-2 py-2 font-mono text-xs text-ink outline-none focus-visible:ring-1 focus-visible:ring-ink-faint"
            />
            <CommandMenu items={items} query={bar.query} onRun={onRun} onClose={bar.closeMenu} />
          </div>
        </ContentCentered>
      </div>}
      {isActive && bar.panelState && <MemoryPanel state={bar.panelState} actions={bar.panelActions} />}
    </>
  )
}
