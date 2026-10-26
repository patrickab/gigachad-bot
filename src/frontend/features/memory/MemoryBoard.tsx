"use client"

import { useCallback, useMemo } from "react"
import type { MemoryTopic } from "@/lib/domain/types"
import { type MemoryRecord, groupByTopicOrder, formatTopicName, normalizeMemoryTopic } from "@/features/memory/memoryUtils"
import { ElevationProvider, ElevatedContainer } from "@/components/ui/ElevatedContainer"
import { MemoryCard } from "@/features/memory/MemoryCard"
import { useDropZone } from "@/hooks/useDropZone"
import { cn } from "@/lib/utils"

const PLACEHOLDER_DELAY_MS = 300

interface MemoryBoardProps {
  memories: MemoryRecord[]
  topicOrder: MemoryTopic[]
  disabled?: boolean
  mode?: "review" | "display"
  onChange?: (memoryId: string, text: string) => void
  onChangeTopic: (memoryId: string, topic: string) => void
  onRemove?: (memoryId: string) => void
  onAccept?: (memoryId: string) => void
  onCancel?: (memoryId: string) => void
}

export function MemoryBoard({
  memories,
  topicOrder,
  disabled = false,
  mode = "display",
  onChange,
  onChangeTopic,
  onRemove,
  onAccept,
  onCancel,
}: MemoryBoardProps) {
  const memoryById = useMemo(
    () => new Map(memories.map((m) => [m.id, m])),
    [memories],
  )

  // Dropping a card on its current topic is a no-op, so don't highlight that zone
  const isValidTarget = useCallback((topicName: string, itemId: string) => {
    const mem = memoryById.get(itemId)
    return mem ? normalizeMemoryTopic(mem) !== topicName : false
  }, [memoryById])

  const handleDrop = useCallback((topicName: string, itemId: string) => {
    const mem = memoryById.get(itemId)
    if (mem && normalizeMemoryTopic(mem) !== topicName) {
      onChangeTopic(itemId, topicName)
    }
  }, [memoryById, onChangeTopic])

  const drop = useDropZone({
    onDrop: handleDrop,
    isValidTarget,
    placeholderDelayMs: PLACEHOLDER_DELAY_MS,
  })

  const sections = useMemo(
    () => groupByTopicOrder(topicOrder, memories),
    [topicOrder, memories],
  )

  if (sections.length === 0) {
    return <div className="text-xs italic text-ink-faint">No memories in this profile.</div>
  }

  return (
    <ElevationProvider>
      <ElevatedContainer className="rounded-2xl border border-divider/30 p-3">
        <div className="space-y-3">
          {sections.map(({ topic, items }) => {
            const zone = drop.bindZone(topic.name)

            return (
              <ElevatedContainer
                key={topic.name}
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
                    {formatTopicName(topic.name)}
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
                          disabled={disabled}
                          mode={mode}
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
