"use client"

import { CommandMenu, type CommandMenuItem } from "@/features/navigation/CommandMenu"
import { MemoryPanel } from "@/features/memory/MemoryPanel"
import { useTab } from "@/contexts/TabContext"
import type { MemoryCommandBar } from "@/features/memory/useMemoryCommandBar"

/** Show workspace commands independently of sidebar navigation, plus memory review. */
export function MemoryCommandLayer({ bar, items, onRun }: {
  bar: MemoryCommandBar
  items: CommandMenuItem[]
  onRun: (command: string) => void
}) {
  const { tab, isActive } = useTab()

  return (
    <>
      {isActive && bar.menuOpen && <div className="absolute inset-0 z-[70]" onClick={bar.closeMenu}>
        <div role="dialog" aria-label="Commands" className="absolute left-1/2 top-14 w-80 max-w-[calc(100%-2rem)] -translate-x-1/2 overflow-hidden rounded-xl border border-divider bg-surface p-2 shadow-[var(--shadow-lg)]" onClick={(event) => event.stopPropagation()}>
          <input
            autoFocus
            aria-label="Search commands"
            placeholder="Find a command…"
            value={bar.query}
            onChange={(event) => bar.setQuery(event.target.value)}
            spellCheck={false}
            className="w-full rounded-md bg-paper px-2 py-2 font-mono text-xs text-ink outline-none focus-visible:ring-1 focus-visible:ring-ink-faint"
          />
          <CommandMenu open items={items} query={bar.query} onRun={onRun} onClose={bar.closeMenu} />
        </div>
      </div>}
      {isActive && bar.panelState && (
        <MemoryPanel
          state={bar.panelState}
          projectEnabled={!!tab.projectSlug}
          actions={bar.panelActions}
          globalCategories={bar.categories.globalCategories}
          projectCategories={bar.categories.projectCategories}
        />
      )}
    </>
  )
}
