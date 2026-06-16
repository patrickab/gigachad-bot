"use client"

import { useCallback, useRef, useState } from "react"

export interface UseDropZoneOptions<T extends string> {
  onDrop: (target: T, itemId: string) => void
  /** Return false to ignore hover/placeholder on this target (e.g. same bucket as source). */
  isValidTarget?: (target: T, itemId: string) => boolean
  placeholderDelayMs?: number
}

export function useDropZone<T extends string>({
  onDrop,
  isValidTarget,
  placeholderDelayMs = 300,
}: UseDropZoneOptions<T>) {
  const [draggedId, setDraggedId] = useState<string | null>(null)
  const [hoverTarget, setHoverTarget] = useState<T | null>(null)
  const [placeholderTarget, setPlaceholderTarget] = useState<T | null>(null)
  const [cardHeight, setCardHeight] = useState(80)

  const enterCounters = useRef<Record<string, number>>({})
  const placeholderTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const draggedIdRef = useRef<string | null>(null)

  const clearPlaceholderTimer = useCallback(() => {
    if (placeholderTimer.current !== null) {
      clearTimeout(placeholderTimer.current)
      placeholderTimer.current = null
    }
  }, [])

  const reset = useCallback(() => {
    clearPlaceholderTimer()
    enterCounters.current = {}
    draggedIdRef.current = null
    setDraggedId(null)
    setHoverTarget(null)
    setPlaceholderTarget(null)
  }, [clearPlaceholderTimer])

  const bindItem = useCallback((itemId: string) => ({
    draggable: true as const,
    isDragging: draggedId === itemId,
    onDragStart: (e: React.DragEvent) => {
      const el = e.currentTarget as HTMLElement
      draggedIdRef.current = itemId
      setCardHeight(el.offsetHeight)
      e.dataTransfer.setDragImage(el, el.offsetWidth / 2, 20)
      e.dataTransfer.setData("text/plain", itemId)
      e.dataTransfer.effectAllowed = "move"
      setTimeout(() => setDraggedId(itemId), 0)
    },
    onDragEnd: reset,
  }), [draggedId, reset])

  const bindZone = useCallback((target: T) => {
    const itemId = draggedIdRef.current
    const valid = itemId ? (isValidTarget?.(target, itemId) ?? true) : false
    const isHovered = hoverTarget === target && valid && draggedId !== null
    const showPlaceholder = placeholderTarget === target && valid && draggedId !== null

    return {
      isHovered,
      showPlaceholder,
      onDragOver: (e: React.DragEvent) => {
        e.preventDefault()
        e.dataTransfer.dropEffect = "move"
      },
      onDragEnter: (e: React.DragEvent) => {
        e.preventDefault()
        enterCounters.current[target] = (enterCounters.current[target] ?? 0) + 1
        if (enterCounters.current[target] !== 1) return

        setHoverTarget(target)
        clearPlaceholderTimer()
        setPlaceholderTarget(null)

        const id = draggedIdRef.current
        if (id && (isValidTarget?.(target, id) ?? true)) {
          placeholderTimer.current = setTimeout(() => {
            setPlaceholderTarget(target)
          }, placeholderDelayMs)
        }
      },
      onDragLeave: () => {
        enterCounters.current[target] = Math.max(0, (enterCounters.current[target] ?? 1) - 1)
        if (enterCounters.current[target] > 0) return

        clearPlaceholderTimer()
        setHoverTarget((prev) => (prev === target ? null : prev))
        setPlaceholderTarget((prev) => (prev === target ? null : prev))
      },
      onDrop: (e: React.DragEvent) => {
        e.preventDefault()
        const droppedId = e.dataTransfer.getData("text/plain")
        reset()
        if (droppedId) onDrop(target, droppedId)
      },
    }
  }, [
    clearPlaceholderTimer,
    draggedId,
    hoverTarget,
    isValidTarget,
    onDrop,
    placeholderDelayMs,
    placeholderTarget,
    reset,
  ])

  return {
    draggedId,
    cardHeight,
    isDragging: draggedId !== null,
    bindItem,
    bindZone,
  }
}
