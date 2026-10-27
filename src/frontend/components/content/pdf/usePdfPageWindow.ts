"use client"

import { useCallback, useEffect, useRef, useState, type RefObject } from "react"

// pages within two viewport heights above or below the visible area keep a rendered canvas
const MOUNT_MARGIN = "200% 0px"
// pages this close keep a low-res placeholder, so scrolling never shows a blank page
const PLACEHOLDER_MARGIN = "1500% 0px"
// scrolling must rest this long before a placeholder renders, so it never competes with visible pages
const PLACEHOLDER_IDLE_MS = 120

type PageSetUpdate = (update: (previous: ReadonlySet<number>) => ReadonlySet<number>) => void

export interface PdfPageWindow {
  /** Pages that hold a live canvas. */
  nearPages: ReadonlySet<number>
  /** Pages that hold a low-res placeholder. */
  keepPages: ReadonlySet<number>
  /** Mounted page slots by page number. */
  slotsRef: RefObject<Map<number, HTMLDivElement>>
  registerSlot: (pageNumber: number, slot: HTMLDivElement | null) => void
  onPlaceholderChange: (pageNumber: number, settled: boolean) => void
  /** The page rendering its placeholder. */
  activePlaceholderTarget: number | null
  /** Record a scroll, so placeholders wait for scrolling to rest. */
  markScrolled: () => void
}

/**
 * Which pages of the scroller hold a canvas or a placeholder, and the one-at-a-time placeholder scheduler.
 *
 * - `ready`: the pages are laid out, placeholders may render
 * - `currentPageRef`: the page placeholders render outward from
 */
export function usePdfPageWindow(scrollRef: RefObject<HTMLDivElement | null>, ready: boolean, currentPageRef: RefObject<number>): PdfPageWindow {
  const [nearPages, setNearPages] = useState<ReadonlySet<number>>(() => new Set())
  const [keepPages, setKeepPages] = useState<ReadonlySet<number>>(() => new Set())
  // the one page currently rendering its placeholder
  const [placeholderTarget, setPlaceholderTarget] = useState<number | null>(null)
  const settledPlaceholdersRef = useRef(new Set<number>())
  const lastScrollRef = useRef(0)
  const slotsRef = useRef(new Map<number, HTMLDivElement>())
  const pageObserversRef = useRef<IntersectionObserver[]>([])

  // Track pages near the viewport: `nearPages` hold a canvas, `keepPages` a placeholder. A hidden scroller (inactive tab) reports nothing visible, its pages stay as they are.
  useEffect(() => {
    const scroller = scrollRef.current
    if (!scroller) return
    const observe = (rootMargin: string, setPages: PageSetUpdate) =>
      new IntersectionObserver((entries) => {
        if (!entries[0]?.rootBounds?.height) return
        setPages((previous) => {
          const next = new Set(previous)
          for (const entry of entries) {
            const pageNumber = Number((entry.target as HTMLElement).dataset.pageNumber)
            if (entry.isIntersecting) next.add(pageNumber)
            else next.delete(pageNumber)
          }
          return next
        })
      }, { root: scroller, rootMargin })
    const observers = [observe(MOUNT_MARGIN, setNearPages), observe(PLACEHOLDER_MARGIN, setKeepPages)]
    pageObserversRef.current = observers
    for (const observer of observers) for (const slot of slotsRef.current.values()) observer.observe(slot)
    return () => {
      for (const observer of observers) observer.disconnect()
      pageObserversRef.current = []
    }
  }, [scrollRef])

  const registerSlot = useCallback((pageNumber: number, slot: HTMLDivElement | null) => {
    const previous = slotsRef.current.get(pageNumber)
    if (previous) for (const observer of pageObserversRef.current) observer.unobserve(previous)
    if (slot) {
      slotsRef.current.set(pageNumber, slot)
      for (const observer of pageObserversRef.current) observer.observe(slot)
    } else {
      slotsRef.current.delete(pageNumber)
    }
  }, [])

  const onPlaceholderChange = useCallback((pageNumber: number, settled: boolean) => {
    setPlaceholderTarget((target) => (target === pageNumber ? null : target))
    if (settled) settledPlaceholdersRef.current.add(pageNumber)
    else settledPlaceholdersRef.current.delete(pageNumber)
  }, [])

  const markScrolled = useCallback(() => {
    lastScrollRef.current = performance.now()
  }, [])

  // A target that got a live canvas or left the kept band is dropped, its scratch render cancels.
  const activePlaceholderTarget = placeholderTarget != null && !nearPages.has(placeholderTarget) && keepPages.has(placeholderTarget) ? placeholderTarget : null

  // While scrolling rests, render placeholders one page at a time for kept pages that have none, nearest to the current page first.
  useEffect(() => {
    if (!ready || activePlaceholderTarget != null) return
    let timer: ReturnType<typeof setTimeout>
    const scheduleNext = (delay: number) => {
      timer = setTimeout(() => {
        const sinceScroll = performance.now() - lastScrollRef.current
        if (sinceScroll < PLACEHOLDER_IDLE_MS) return scheduleNext(PLACEHOLDER_IDLE_MS - sinceScroll)
        let nearest: number | null = null
        for (const page of keepPages) {
          if (nearPages.has(page) || settledPlaceholdersRef.current.has(page)) continue
          if (nearest == null || Math.abs(page - currentPageRef.current) < Math.abs(nearest - currentPageRef.current)) nearest = page
        }
        if (nearest != null) setPlaceholderTarget(nearest)
      }, delay)
    }
    scheduleNext(PLACEHOLDER_IDLE_MS)
    return () => clearTimeout(timer)
  }, [ready, activePlaceholderTarget, nearPages, keepPages, currentPageRef])

  return { nearPages, keepPages, slotsRef, registerSlot, onPlaceholderChange, activePlaceholderTarget, markScrolled }
}
