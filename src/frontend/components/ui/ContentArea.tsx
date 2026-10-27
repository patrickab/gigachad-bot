"use client"

import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react"
import { cn } from "@/lib/utils"

/** Horizontal distance in px from a containing block's edges to the content area's edges. */
export interface ContentInsets {
  left: number
  right: number
}

interface ContentAreaRegistry {
  areas: HTMLElement[]
  register: (area: HTMLElement) => () => void
}

const ContentAreaContext = createContext<ContentAreaRegistry | null>(null)

/**
 * Collect content areas, the regions overlays and floating controls center on.
 *
 * - Each workspace's `main` registers, a nested pane (e.g. a PDF beside its chat) narrows it
 * - Hidden and inert (outgoing tab) areas are skipped, without any area overlays center on the viewport
 */
export function ContentAreaProvider({ children }: { children: ReactNode }) {
  const [areas, setAreas] = useState<HTMLElement[]>([])
  const register = useCallback((area: HTMLElement) => {
    setAreas((current) => [...current, area])
    return () => setAreas((current) => current.filter((registered) => registered !== area))
  }, [])
  const registry = useMemo(() => ({ areas, register }), [areas, register])
  return <ContentAreaContext.Provider value={registry}>{children}</ContentAreaContext.Provider>
}

/** Ref callback that registers its element as a content area. */
export function useContentAreaRef(): ((area: HTMLElement | null) => (() => void) | undefined) | undefined {
  const register = useContext(ContentAreaContext)?.register
  return useMemo(() => register && ((area: HTMLElement | null) => area ? register(area) : undefined), [register])
}

function isShownArea(area: HTMLElement): boolean {
  return area.getClientRects().length > 0 && !area.closest("[inert]")
}

function innermostArea(areas: HTMLElement[]): HTMLElement | undefined {
  return areas.find((area) => !areas.some((other) => other !== area && area.contains(other)))
}

function areSameInsets(current: ContentInsets | undefined, next: ContentInsets | undefined): boolean {
  return current?.left === next?.left && current?.right === next?.right
}

/** Insets of the shown innermost content area within `anchor`'s containing block, the viewport for fixed elements. */
export function useContentInsets(anchor: HTMLElement | null): ContentInsets | undefined {
  const areas = useContext(ContentAreaContext)?.areas
  const [insets, setInsets] = useState<ContentInsets>()
  useEffect(() => {
    if (!areas?.length || !anchor) return
    const frame = anchor.offsetParent
    const measure = () => {
      const area = innermostArea(areas.filter(isShownArea))
      const content = area?.getBoundingClientRect()
      const bounds = frame?.getBoundingClientRect() ?? { left: 0, right: window.innerWidth }
      const next = content && { left: Math.max(0, content.left - bounds.left), right: Math.max(0, bounds.right - content.right) }
      setInsets((current) => areSameInsets(current, next) ? current : next)
    }
    // Reports once on observe, then on every resize: sidebar toggle or drag, chat pane, tab shown or hidden
    const observer = new ResizeObserver(measure)
    for (const area of areas) observer.observe(area)
    if (frame) observer.observe(frame)
    window.addEventListener("resize", measure)
    return () => {
      observer.disconnect()
      window.removeEventListener("resize", measure)
    }
  }, [areas, anchor])
  return areas?.length && anchor ? insets : undefined
}

/** Absolute row whose center sits over the content area's center. Set its vertical position via `className`. */
export function ContentCentered({ className, children }: { className?: string; children: ReactNode }) {
  const [row, setRow] = useState<HTMLDivElement | null>(null)
  const insets = useContentInsets(row)
  return <div ref={setRow} className={cn("absolute flex justify-center", className)} style={{ left: insets?.left ?? 0, right: insets?.right ?? 0 }}>
    {children}
  </div>
}
