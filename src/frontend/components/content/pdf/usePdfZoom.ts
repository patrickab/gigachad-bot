"use client"

import { useCallback, useLayoutEffect, useRef, useState, type RefObject } from "react"
import { useDebouncedValue } from "@/hooks/useDebouncedValue"
import { useEventListener } from "@/hooks/useEventListener"
import { isModKey } from "@/lib/utils"

export const MIN_ZOOM = 1
export const MAX_ZOOM = 2.5
const ZOOM_STEP = 0.25
// zoom steps scale the current bitmaps, pages re-rasterize once the zoom rests this long
const ZOOM_SETTLE_MS = 150

interface ScrollAnchor {
  mouseX: number
  mouseY: number
  scrollLeft: number
  scrollTop: number
  ratio: number
}

export interface PdfZoom {
  zoom: number
  /** `zoom` once it rests, the zoom pages rasterize at. */
  rasterZoom: number
  /** Zoom to `nextZoom`, keeping the client point under it (default: viewport center) fixed. */
  zoomAt: (nextZoom: number, point?: { x: number; y: number }) => void
  /** Zoom one step in (1) or out (-1). */
  step: (direction: 1 | -1) => void
}

/** Zoom of the pages in `scrollRef`, with Ctrl/Cmd + wheel zooming at the cursor. */
export function usePdfZoom(scrollRef: RefObject<HTMLDivElement | null>): PdfZoom {
  const [zoom, setZoom] = useState(1)
  // Re-rasterize once the zoom rests, until then the current bitmaps scale with it.
  const rasterZoom = useDebouncedValue(zoom, ZOOM_SETTLE_MS)
  const zoomRef = useRef(1)
  const anchorRef = useRef<ScrollAnchor | null>(null)

  const zoomAt = useCallback((nextZoom: number, point?: { x: number; y: number }) => {
    const el = scrollRef.current
    if (!el) return

    const prevZoom = zoomRef.current
    const clamped = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, nextZoom))
    if (clamped === prevZoom) return

    const rect = el.getBoundingClientRect()
    anchorRef.current = {
      mouseX: point ? point.x - rect.left : rect.width / 2,
      mouseY: point ? point.y - rect.top : rect.height / 2,
      scrollLeft: el.scrollLeft,
      scrollTop: el.scrollTop,
      ratio: clamped / prevZoom,
    }

    zoomRef.current = clamped
    setZoom(clamped)
  }, [scrollRef])

  const step = useCallback((direction: 1 | -1) => zoomAt(zoomRef.current + direction * ZOOM_STEP), [zoomAt])

  // Ctrl/Cmd + wheel zooms, anchored at the cursor
  useEventListener(scrollRef, "wheel", (e) => {
    if (!isModKey(e)) return
    e.preventDefault()
    zoomAt(zoomRef.current * (e.deltaY < 0 ? 1.08 : 1 / 1.08), { x: e.clientX, y: e.clientY })
  }, { passive: false })

  // Re-scroll after a zoom so the point under the cursor (or viewport center) stays put.
  useLayoutEffect(() => {
    const anchor = anchorRef.current
    if (!anchor) return
    anchorRef.current = null

    const el = scrollRef.current
    if (!el) return

    el.scrollLeft = anchor.scrollLeft * anchor.ratio + anchor.mouseX * (anchor.ratio - 1)
    el.scrollTop = anchor.scrollTop * anchor.ratio + anchor.mouseY * (anchor.ratio - 1)
  }, [zoom, scrollRef])

  return { zoom, rasterZoom, zoomAt, step }
}
