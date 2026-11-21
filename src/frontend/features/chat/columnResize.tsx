"use client"

import { useCallback, type PointerEvent as ReactPointerEvent } from "react"
import { useStoredString } from "@/hooks/useStoredString"

/** Default chat column width, also the narrowest a dragged column gets. */
export const CHAT_COLUMN_MAX_PX = 960
const STORAGE_KEY = "chat-column-width"

export type ResizeStart = (e: ReactPointerEvent, side: "left" | "right") => void

/**
 * The chat column width the user dragged, shared by every chat and Pi session and kept in `localStorage`.
 *
 * - `width` is the default until a card edge is dragged
 * - `startResize` goes on a card edge's `onPointerDown`: both sides move equally, within the default and the scroller's width
 * - The column is the closest `[data-chat-column]`, its parent the scroller
 */
export function useColumnWidth(): { width: number; startResize: ResizeStart } {
  const [stored, store] = useStoredString(STORAGE_KEY)
  const parsed = Number(stored)
  const width = stored !== null && Number.isFinite(parsed) && parsed >= CHAT_COLUMN_MAX_PX ? parsed : CHAT_COLUMN_MAX_PX
  const startResize = useCallback<ResizeStart>((e, side) => {
    const column = e.currentTarget.closest<HTMLElement>("[data-chat-column]")
    const scroller = column?.parentElement
    if (!column || !scroller) return
    e.preventDefault()
    const startX = e.clientX
    const startWidth = column.offsetWidth
    const sign = side === "right" ? 2 : -2
    const move = (ev: PointerEvent) => {
      const next = startWidth + sign * (ev.clientX - startX)
      store(String(Math.round(Math.min(Math.max(scroller.clientWidth, CHAT_COLUMN_MAX_PX), Math.max(CHAT_COLUMN_MAX_PX, next)))))
    }
    const up = () => {
      window.removeEventListener("pointermove", move)
      window.removeEventListener("pointerup", up)
    }
    window.addEventListener("pointermove", move)
    window.addEventListener("pointerup", up)
  }, [store])
  return { width, startResize }
}

/** Invisible drag strips on a card's side edges; the card's wrapper must be `relative`. */
export function ResizeEdges({ onResizeStart }: { onResizeStart?: ResizeStart }) {
  if (!onResizeStart) return null
  return (["left", "right"] as const).map((side) => (
    <div
      key={side}
      aria-hidden="true"
      onPointerDown={(e) => onResizeStart(e, side)}
      className={`absolute top-0 bottom-0 z-10 w-2 cursor-col-resize touch-none ${side === "left" ? "left-0" : "right-0"}`}
    />
  ))
}
