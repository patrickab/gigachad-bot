"use client"

import { useMemo } from "react"
import { ListPlus } from "lucide-react"
import type { ArchitectureElement, ArchitectureViewNode } from "@/lib/domain/architecture"
import { cn } from "@/lib/utils"
import { toolButton } from "@/features/editor/canvas/CanvasToolbar"
import { ArchitectureMenu } from "./ArchitectureMenu"

// Adds to a saved view from the whole model, grouped by the file declaring each
// element. "Add all" frames a file's systems with everything inside them.
export function ArchitectureViewPicker({ elements, shown, onInclude }: {
  elements: ArchitectureElement[]
  shown: ArchitectureViewNode[]
  onInclude: (elements: string[], descendants: boolean) => void
}) {
  const groups = useMemo(() => {
    const visible = new Set(shown.map((node) => node.id))
    const byFile = new Map<string, ArchitectureElement[]>()
    for (const element of elements) {
      if (!element.file) continue
      byFile.set(element.file, [...(byFile.get(element.file) ?? []), element])
    }
    return [...byFile.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([file, declared]) => {
      const here = new Set(declared.map((element) => element.id))
      return {
        file,
        roots: declared.filter((element) => !element.parent || !here.has(element.parent)).map((element) => element.id),
        rows: declared.sort((a, b) => a.id.localeCompare(b.id)).map((element) => ({ element, depth: element.id.split(".").length - 1, visible: visible.has(element.id) })),
      }
    })
  }, [elements, shown])
  return (
    <ArchitectureMenu
      label="Add to view" align="left" menuStyle={{ width: 260, maxHeight: 320, overflowY: "auto" }}
      trigger={(open, toggle) => (
        <button type="button" onClick={toggle} aria-haspopup="menu" aria-expanded={open} className={cn(toolButton(open), "flex items-center gap-1 text-xs")}>
          <ListPlus className="h-3.5 w-3.5" /> Add to view
        </button>
      )}
    >
      {(close) => {
        const include = (ids: string[], descendants: boolean) => { onInclude(ids, descendants); close() }
        return <>
          {groups.length === 0 && <p className="px-2 py-1.5 text-[11px] text-ink-faint">Nothing to add yet: create an architecture file first.</p>}
          {groups.map((group) => (
            <div key={group.file} className="py-0.5">
              <div className="flex items-center gap-2 px-2 py-1">
                <span className="min-w-0 flex-1 truncate text-[10px] uppercase tracking-wider text-ink-faint">{group.file}</span>
                <button type="button" role="menuitem" className="architecture-diagram-button" style={{ minHeight: 22 }} onClick={() => include(group.roots, true)}>Add all</button>
              </div>
              {group.rows.map(({ element, depth, visible }) => (
                <button key={element.id} type="button" role="menuitem" disabled={visible} onClick={() => include([element.id], false)} className="architecture-diagram-menu-item disabled:opacity-50" style={{ display: "flex", alignItems: "center", gap: 8, paddingLeft: 8 + depth * 12 }}>
                  <span className="min-w-0 flex-1 truncate">{element.title}</span>
                  <span className="shrink-0 text-[10px] text-ink-faint">{visible ? "shown" : element.kind}</span>
                </button>
              ))}
            </div>
          ))}
        </>
      }}
    </ArchitectureMenu>
  )
}
