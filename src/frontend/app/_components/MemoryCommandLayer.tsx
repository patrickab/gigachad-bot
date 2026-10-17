"use client"

import { useEffect, useRef } from "react"
import { createPortal } from "react-dom"
import { CommandMenu, type CommandMenuItem } from "@/features/navigation/CommandMenu"
import { MemoryPanel } from "@/features/memory/MemoryPanel"
import { useProject } from "@/contexts/ProjectContext"
import { useTab } from "@/contexts/TabContext"
import type { MemoryCommandBar } from "@/features/memory/useMemoryCommandBar"

/** The slash-command input and menu in the tab row, the click-away overlay, and the memory review panel. */
export function MemoryCommandLayer({ bar, items, onRun }: {
  bar: MemoryCommandBar
  items: CommandMenuItem[]
  onRun: (command: string) => void
}) {
  const { isActive } = useTab()
  const { activeProject } = useProject()
  const inputRef = useRef<HTMLInputElement>(null)
  const { menuOpen } = bar

  useEffect(() => {
    if (!menuOpen) return
    // Defer so the portaled input exists before focusing.
    const timer = setTimeout(() => inputRef.current?.focus(), 0)
    return () => clearTimeout(timer)
  }, [menuOpen])

  return (
    <>
      {isActive && bar.menuOpen && bar.labelSlot && createPortal(
        <input
          ref={inputRef}
          autoFocus
          value={bar.query}
          onChange={(e) => bar.setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Escape") { e.preventDefault(); bar.closeMenu() }
          }}
          onClick={(e) => e.stopPropagation()}
          spellCheck={false}
          className="flex-1 min-w-0 bg-transparent font-mono text-xs text-ink outline-none"
        />,
        bar.labelSlot,
      )}
      <CommandMenu
        open={isActive && bar.menuOpen}
        items={items}
        query={bar.query}
        onRun={onRun}
        onClose={bar.closeMenu}
        container={bar.menuSlot}
      />
      {bar.menuOpen && (
        <div
          className="absolute inset-0 z-[70]"
          onClick={bar.closeMenu}
        />
      )}
      {isActive && bar.panelState && (
        <MemoryPanel
          state={bar.panelState}
          projectEnabled={!!activeProject}
          actions={bar.panelActions}
          globalCategories={bar.categories.globalCategories}
          projectCategories={bar.categories.projectCategories}
        />
      )}
    </>
  )
}
