"use client"

import { useCallback, useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type ReactNode } from "react"
import { animate, motion, useMotionValue, useReducedMotion, useTransform } from "framer-motion"
import { cn } from "@/lib/utils"

const MIN_WIDTH = 200
const MAX_WIDTH = 600
/** Space between the card and the window edge it is anchored to. */
const GUTTER_PX = 20
const EASE_S = 0.2

type Side = "left" | "right"

/** Look of a sidebar card, shared by every floating card that should read as a sidebar (the picker's finder too). */
export const SIDEBAR_CARD_CLASS = "rounded-2xl bg-surface shadow-[var(--shadow-sm)]"

/**
 * The card's width easing to `open ? width : 0`, the content's width, and a drag handler for the inner edge.
 *
 * - The content keeps its open width while the card eases, the card clips it, so it slides with the edge instead of reflowing
 * - A drag sets both at once and reports the final width through `onWidthChange`, external changes never fight it
 */
function usePanelWidth(side: Side, open: boolean, width: number, onWidthChange: ((width: number) => void) | undefined, minWidth: number, maxWidth: number) {
  const cardWidth = useMotionValue(open ? width : 0)
  const contentWidth = useMotionValue(width)
  /** Ends the running drag's listeners and body styles, null while not dragging. */
  const stopDrag = useRef<(() => void) | null>(null)
  const reduceMotion = useReducedMotion()

  useEffect(() => {
    if (stopDrag.current) return
    if (open) contentWidth.set(width)
    const controls = animate(cardWidth, open ? width : 0, { duration: reduceMotion ? 0 : EASE_S, ease: "easeOut" })
    return controls.stop
  }, [open, width, cardWidth, contentWidth, reduceMotion])
  // A drag cut short by unmounting (e.g. closing the tab) must not leave the app unselectable
  useEffect(() => () => stopDrag.current?.(), [])

  // Pointer capture keeps the drag on the handle while the pointer crosses an iframe (learn frame, previews),
  // which would otherwise swallow the moves and the release.
  const handlePointerDown = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 || !onWidthChange) return
    event.preventDefault()
    const handle = event.currentTarget
    handle.setPointerCapture(event.pointerId)
    const startX = event.clientX
    const startWidth = cardWidth.get()
    // The inner edge moves: rightward widens a left card, leftward a right one
    const direction = side === "left" ? 1 : -1
    const clamp = (px: number) => Math.min(maxWidth, Math.max(minWidth, px))
    document.body.style.cursor = "col-resize"
    document.body.style.userSelect = "none"

    const onPointerMove = (moved: PointerEvent) => {
      const next = clamp(startWidth + direction * (moved.clientX - startX))
      cardWidth.set(next)
      contentWidth.set(next)
    }
    const stop = () => {
      stopDrag.current = null
      handle.removeEventListener("pointermove", onPointerMove)
      handle.removeEventListener("lostpointercapture", onCaptureEnd)
      document.body.style.cursor = ""
      document.body.style.userSelect = ""
    }
    // Capture ends on release and on cancel alike
    const onCaptureEnd = () => {
      stop()
      onWidthChange(clamp(cardWidth.get()))
    }
    stopDrag.current = stop
    handle.addEventListener("pointermove", onPointerMove)
    handle.addEventListener("lostpointercapture", onCaptureEnd)
  }, [side, cardWidth, contentWidth, onWidthChange, minWidth, maxWidth])

  return { cardWidth, contentWidth, handlePointerDown }
}

/**
 * A sidebar card floating at the `side` edge of the window, the app's one sidebar shape.
 *
 * - Opens and closes by easing its width, the gutter to the edge and its opacity follow, the content slides with the edge
 * - Stays mounted while closed (state, scroll position and Neovim survive), then inert and hidden from assistive tech
 * - `onWidthChange` makes the inner edge a drag handle between `MIN_WIDTH` and `maxWidth`, the owner keeps the width
 * - Scrolling content goes in a `SidePanelScroll`
 */
export function SidePanel({ side, open, width, onWidthChange, maxWidth = MAX_WIDTH, label, className, children }: {
  side: Side
  open: boolean
  /** Open width in px. */
  width: number
  onWidthChange?: (width: number) => void
  maxWidth?: number
  /** Accessible name of the panel landmark. */
  label?: string
  className?: string
  children: ReactNode
}) {
  const { cardWidth, contentWidth, handlePointerDown } = usePanelWidth(side, open, width, onWidthChange, MIN_WIDTH, maxWidth)
  const gutter = useTransform(cardWidth, (px) => Math.min(GUTTER_PX, px))
  return (
    <motion.aside
      aria-label={label}
      aria-hidden={!open}
      inert={!open}
      style={{ width: cardWidth, [side === "left" ? "marginLeft" : "marginRight"]: gutter }}
      // Fades with the width's easing, closed it casts no shadow
      className={cn("relative my-5 flex shrink-0 overflow-hidden transition-opacity duration-200 ease-out",
        SIDEBAR_CARD_CLASS, !open && "opacity-0", side === "right" && "justify-end", className)}
    >
      <motion.div style={{ width: contentWidth }} className="flex min-h-0 shrink-0 flex-col">
        {children}
      </motion.div>
      {onWidthChange && (
        <div
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize sidebar"
          className={cn("absolute inset-y-0 z-10 w-1.5 cursor-col-resize transition-colors hover:bg-hover active:bg-hover", side === "left" ? "right-0" : "left-0")}
          onPointerDown={handlePointerDown}
        />
      )}
    </motion.aside>
  )
}

/** Height of the fade at a clipped edge, reached once that much content is clipped. */
const EDGE_FADE_PX = 24
const THUMB_MIN_PX = 24
/** The thumb stays this long after the last scroll. */
const THUMB_LINGER_MS = 800

const SCROLLER_STYLE: CSSProperties = {
  scrollbarWidth: "none",
  maskImage: "linear-gradient(to bottom, transparent, #000 var(--fade-top, 0px), #000 calc(100% - var(--fade-bottom, 0px)), transparent)",
}

/**
 * The vertical scroll area of a sidebar, without the native scrollbar.
 *
 * - Content clipped at an edge fades out there, deeper the more is clipped, up to `EDGE_FADE_PX`
 * - A thin overlay thumb shows while hovered or scrolling and can be dragged, it takes no width from the content
 * - `className` styles the scrolling element (padding, gaps)
 */
export function SidePanelScroll({ className, children }: { className?: string; children: ReactNode }) {
  const scrollerRef = useRef<HTMLDivElement>(null)
  const contentRef = useRef<HTMLDivElement>(null)
  const thumbRef = useRef<HTMLDivElement>(null)
  const [scrolling, setScrolling] = useState(false)
  const lingerTimer = useRef<number | undefined>(undefined)

  // Written to the DOM directly: a scroll event per frame must not re-render the content
  const layout = useCallback(() => {
    const scroller = scrollerRef.current
    const thumb = thumbRef.current
    if (!scroller || !thumb) return
    const { scrollTop, scrollHeight, clientHeight } = scroller
    const overflow = scrollHeight - clientHeight
    scroller.style.setProperty("--fade-top", `${Math.min(EDGE_FADE_PX, scrollTop)}px`)
    scroller.style.setProperty("--fade-bottom", `${Math.min(EDGE_FADE_PX, Math.max(0, overflow - scrollTop))}px`)
    thumb.hidden = overflow <= 0
    if (overflow <= 0) return
    const height = Math.max(THUMB_MIN_PX, (clientHeight * clientHeight) / scrollHeight)
    thumb.style.height = `${height}px`
    thumb.style.transform = `translateY(${(scrollTop / overflow) * (clientHeight - height)}px)`
  }, [])

  useLayoutEffect(() => {
    const scroller = scrollerRef.current
    const content = contentRef.current
    if (!scroller || !content) return
    layout()
    if (typeof ResizeObserver === "undefined") return
    const observer = new ResizeObserver(layout)
    observer.observe(scroller)
    observer.observe(content)
    return () => observer.disconnect()
  }, [layout])
  useEffect(() => () => window.clearTimeout(lingerTimer.current), [])

  const onScroll = () => {
    layout()
    setScrolling(true)
    window.clearTimeout(lingerTimer.current)
    lingerTimer.current = window.setTimeout(() => setScrolling(false), THUMB_LINGER_MS)
  }

  const onThumbPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    const scroller = scrollerRef.current
    if (event.button !== 0 || !scroller) return
    event.preventDefault()
    const thumb = event.currentTarget
    thumb.setPointerCapture(event.pointerId)
    const startY = event.clientY
    const startTop = scroller.scrollTop
    // One px of thumb travel scrolls this many px of content
    const ratio = (scroller.scrollHeight - scroller.clientHeight) / (scroller.clientHeight - thumb.offsetHeight)
    const onPointerMove = (moved: PointerEvent) => { scroller.scrollTop = startTop + (moved.clientY - startY) * ratio }
    const onCaptureEnd = () => {
      thumb.removeEventListener("pointermove", onPointerMove)
      thumb.removeEventListener("lostpointercapture", onCaptureEnd)
    }
    thumb.addEventListener("pointermove", onPointerMove)
    thumb.addEventListener("lostpointercapture", onCaptureEnd)
  }

  return (
    <div className="group/scroll relative flex min-h-0 flex-1 flex-col">
      <div ref={scrollerRef} onScroll={onScroll} style={SCROLLER_STYLE} className={cn("min-h-0 flex-1 overflow-y-auto", className)}>
        <div ref={contentRef}>{children}</div>
      </div>
      <div
        ref={thumbRef}
        aria-hidden
        onPointerDown={onThumbPointerDown}
        className={cn(
          "absolute right-0.5 top-0 w-1 rounded-full bg-ink-faint/50 transition-[opacity,width] duration-200 hover:w-1.5 active:w-1.5",
          scrolling ? "opacity-100" : "opacity-0 group-hover/scroll:opacity-100",
        )}
      />
    </div>
  )
}
