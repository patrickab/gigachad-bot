"use client"

import { useCallback, useMemo } from "react"
import type { CategoryDef, PreviewMemory, ProposedMemory } from "@/lib/types"
import { buildBoardSections, formatCategoryName, normalizeMemoryCategory } from "@/lib/memoryUtils"
import { ElevationProvider, ElevatedContainer } from "@/components/ElevatedContainer"
import { MemoryCard } from "@/components/MemoryPanel"
import { useDropZone } from "@/hooks/useDropZone"
import { cn } from "@/lib/utils"

type MemoryRecord = ProposedMemory | PreviewMemory

const PLACEHOLDER_DELAY_MS = 300

export interface MemoryBoardProps {
  memories: MemoryRecord[]
  categoryOrder: CategoryDef[]
  scope: "global" | "project"
  disabled?: boolean
  mode?: "review" | "display"
  onChange?: (memoryId: string, text: string) => void
  onChangeCategory: (memoryId: string, category: string) => void
  onRemove?: (memoryId: string) => void
  onAccept?: (memoryId: string) => void
  onCancel?: (memoryId: string) => void
}

export function MemoryBoard({
  memories,
  categoryOrder,
  scope,
  disabled = false,
  mode = "display",
  onChange,
  onChangeCategory,
  onRemove,
  onAccept,
  onCancel,
}: MemoryBoardProps) {
  const memoryById = useMemo(
    () => new Map(memories.map((m) => [m.id, m])),
    [memories],
  )

  const isValidTarget = useCallback((catName: string, itemId: string) => {
    const mem = memoryById.get(itemId)
    return mem ? normalizeMemoryCategory(mem) !== catName : false
  }, [memoryById])

  const handleDrop = useCallback((catName: string, itemId: string) => {
    const mem = memoryById.get(itemId)
    if (mem && normalizeMemoryCategory(mem) !== catName) {
      onChangeCategory(itemId, catName)
    }
  }, [memoryById, onChangeCategory])

  const drop = useDropZone({
    onDrop: handleDrop,
    isValidTarget,
    placeholderDelayMs: PLACEHOLDER_DELAY_MS,
  })

  const sections = useMemo(
    () => buildBoardSections(categoryOrder, memories),
    [categoryOrder, memories],
  )

  if (sections.length === 0) {
    return <div className="text-xs italic text-ink-faint">No memories in this profile.</div>
  }

  return (
    <ElevationProvider darkColor="var(--paper)" brightColor="var(--surface-elevated)" numLevels={3}>
      <ElevatedContainer className="rounded-2xl border border-divider/30 p-3">
        <div className="space-y-3">
          {sections.map(({ category, items }) => {
            const zone = drop.bindZone(category.name)

            return (
              <ElevatedContainer
                key={category.name}
                onDragOver={zone.onDragOver}
                onDragEnter={zone.onDragEnter}
                onDragLeave={zone.onDragLeave}
                onDrop={zone.onDrop}
                className={cn(
                  "rounded-xl border overflow-hidden transition-all duration-150",
                  zone.isHovered ? "border-divider-strong" : "border-divider/40",
                )}
              >
                <div className="flex items-center gap-2 px-3 pt-3 pb-2 select-none">
                  <span className="text-[11px] font-bold text-ink-muted uppercase tracking-[0.15em] truncate">
                    {formatCategoryName(category.name)}
                  </span>
                  <span className="text-[10px] font-mono font-bold px-1.5 py-0.5 rounded-md bg-surface-elevated text-ink tabular-nums shrink-0">
                    {items.length}
                  </span>
                </div>

                <div className="space-y-2 px-3 pb-3">
                  {items.length === 0 && !zone.showPlaceholder && (
                    <div className={cn(
                      "h-10 rounded-lg border border-dashed transition-colors duration-150",
                      zone.isHovered ? "border-divider/60 bg-overlay/30" : "border-divider/25",
                    )} />
                  )}

                  {items.map((m) => {
                    const item = drop.bindItem(m.id)
                    return (
                      <ElevatedContainer
                        key={m.id}
                        data-memory-card
                        hoverLift={!disabled && !drop.isDragging}
                        draggable={!disabled && item.draggable}
                        onDragStart={!disabled ? item.onDragStart : undefined}
                        onDragEnd={!disabled ? item.onDragEnd : undefined}
                        className={cn(
                          "rounded-xl border border-divider/40 overflow-hidden shadow-[var(--shadow-md)] transition-all duration-200",
                          !disabled && "cursor-grab active:cursor-grabbing select-none",
                          item.isDragging && "opacity-0 pointer-events-none",
                        )}
                      >
                        <MemoryCard
                          memory={m}
                          scope={scope}
                          disabled={disabled}
                          mode={mode}
                          embedded
                          onChange={onChange}
                          onAccept={onAccept}
                          onCancel={onCancel}
                          onRemove={onRemove}
                        />
                      </ElevatedContainer>
                    )
                  })}

                  {zone.showPlaceholder && (
                    <div
                      className="rounded-xl border border-dashed border-divider/60 bg-overlay/40"
                      style={{ height: drop.cardHeight }}
                    />
                  )}
                </div>
              </ElevatedContainer>
            )
          })}
        </div>
      </ElevatedContainer>
    </ElevationProvider>
  )
}
