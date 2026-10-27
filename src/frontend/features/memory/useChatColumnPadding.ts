"use client"

import { useEffect, useState, type CSSProperties } from "react"

/** Gap between a memory window and the chat column's edges, matching the overlay's default padding. */
const COLUMN_GAP_PX = 24

/** Overlay padding that centers a memory window over the visible chat column instead of the whole screen. */
export function useChatColumnPadding(): CSSProperties | undefined {
  const [padding, setPadding] = useState<CSSProperties>()
  useEffect(() => {
    const column = Array.from(document.querySelectorAll("main")).find((el) => el.getClientRects().length > 0)
    if (!column) return
    const measure = () => {
      const rect = column.getBoundingClientRect()
      setPadding({ paddingLeft: rect.left + COLUMN_GAP_PX, paddingRight: window.innerWidth - rect.right + COLUMN_GAP_PX })
    }
    // The observer reports once on observe, then on every column resize (sidebar toggle, window resize).
    const observer = new ResizeObserver(measure)
    observer.observe(column)
    return () => observer.disconnect()
  }, [])
  return padding
}
