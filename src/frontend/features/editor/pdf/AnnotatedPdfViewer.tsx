"use client"

import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type ComponentProps, type ReactNode, type RefObject } from "react"
import { PdfViewer } from "@/components/content/PdfViewer"
import { useTabActive } from "@/contexts/TabContext"
import { InkTools, SEPARATOR, UndoRedoButtons, useInkSettings } from "@/features/editor/canvas/CanvasToolbar"
import { InTopBar, useHasTopBarSlot } from "@/features/navigation/TopBarSlots"
import { DEFAULT_PEN_COLOR, ERASER_RADIUS_PX, appendPointerSamples, finishStroke, newStroke, pointerInkRole, pointerSample, splitHighlights, strokePath, strokesAfterErase } from "@/features/editor/canvas/ink/canvasInk"
import { useCanvasHistory } from "@/features/editor/canvas/useCanvasHistory"
import { useCollaborativeCanvas } from "@/features/editor/canvas/useCollaborativeCanvas"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import { useUndoRedoKeys } from "@/hooks/useUndoRedoKeys"
import { openPdfAnnotations } from "@/lib/backend/api"
import { fileViewerRawUrl } from "@/lib/backend/urls"
import { emptyCanvasDoc } from "@/lib/domain/canvasDocument"
import { inkOutlinePath, type StrokeData } from "@/lib/drawing"
import { cn } from "@/lib/utils"
import { PAGE_UNITS, pageAt, strokesByPage, toPagePoint, type PageBox } from "./pdfInk"

const EMPTY_DOC = emptyCanvasDoc()

/** Ink a stylus writes with when no tool of the viewer's own is selected. */
type HostPen = { color: string; width: number; highlight: boolean }

type ViewerProps = Omit<ComponentProps<typeof PdfViewer>, "url" | "renderPageOverlay" | "onScrollerChange">

type PageOverlay = (page: number, pageWidth: number) => ReactNode

interface AnnotatedPdfViewerProps extends ViewerProps {
  path: string
  /** Stylus ink when the tab's top bar has no pdf slot for the annotation tools, e.g. the host canvas's pen or highlighter. Then nothing else annotates. */
  pen?: HostPen
  /** Show and accept ink on the pages only, margin notes stay hidden. */
  clipToPages?: boolean
}

/** PDF viewer with ink and highlighter annotations, shared by every copy of the same PDF. Falls back to the plain viewer when annotations are unavailable. */
export function AnnotatedPdfViewer({ path, pen, clipToPages = false, ...viewerProps }: AnnotatedPdfViewerProps) {
  const [annotations, setAnnotations] = useState<{ path: string; key: string | null } | null>(null)
  const [scroller, setScroller] = useState<HTMLDivElement | null>(null)
  const [renderPageOverlay, setRenderPageOverlay] = useState<PageOverlay | undefined>(undefined)
  const publishPageOverlay = useCallback((render: PageOverlay | undefined) => setRenderPageOverlay(() => render), [])

  useEffect(() => {
    let cancelled = false
    openPdfAnnotations(path)
      .then((key) => { if (!cancelled) setAnnotations({ path, key }) })
      .catch(() => { if (!cancelled) setAnnotations({ path, key: null }) })
    return () => { cancelled = true }
  }, [path])

  // The PDF loads alongside the key instead of after it, ink joins the same viewer once the key arrives.
  const canvasKey = annotations?.path === path ? annotations.key : null
  return (
    <>
      <PdfViewer key={path} url={fileViewerRawUrl(path)} {...viewerProps} renderPageOverlay={renderPageOverlay} onScrollerChange={setScroller} />
      {canvasKey && <PdfInk canvasKey={canvasKey} scroller={scroller} pen={pen} clipToPages={clipToPages} onPageOverlay={publishPageOverlay} />}
    </>
  )
}

type Gesture =
  | { kind: "draw"; pointerId: number; page: HTMLElement; pageNumber: number; points: number[][]; highlight: boolean; color: string; width: number }
  | { kind: "erase"; pointerId: number }

/** Stroke being drawn, rendered by its page's overlay. */
type LiveStroke = { page: number; highlight: boolean; color: string }

/** Ink and highlighter annotations over the pages in `scroller`: hands `onPageOverlay` the per-page ink and renders the annotation tools.
 *
 * - Memoized: publishing the overlay re-renders the host, which must not re-render this with it
 */
const PdfInk = memo(function PdfInk({ canvasKey, scroller, pen, clipToPages, onPageOverlay }: {
  canvasKey: string
  scroller: HTMLDivElement | null
  pen?: HostPen
  clipToPages: boolean
  onPageOverlay: (render: PageOverlay | undefined) => void
}) {
  const active = useTabActive()
  const { document, replace, status } = useCollaborativeCanvas(canvasKey)
  const doc = document ?? EMPTY_DOC
  const { liveRef, applyChange, commit, snapshotOnce, endGesture, undo, redo, canUndo, canRedo } = useCanvasHistory(doc, replace)
  const loaded = document !== null

  const ink = useInkSettings()
  const [live, setLive] = useState<LiveStroke | null>(null)
  const livePathRef = useRef<SVGPathElement>(null)

  const pages = useMemo(() => strokesByPage(doc.strokes), [doc.strokes])

  // Pointer handlers read the latest settings without re-binding mid-stroke.
  const settingsRef = useRef({ ink, pen, loaded })
  useLayoutEffect(() => { settingsRef.current = { ink, pen, loaded } })

  useEffect(() => {
    if (!scroller) return
    let gesture: Gesture | null = null

    const pageBoxes = (): PageBox[] => Array.from(
      scroller.querySelectorAll<HTMLElement>("[data-page-number]"),
      (element) => ({ page: Number(element.dataset.pageNumber), rect: element.getBoundingClientRect() }),
    )

    const redrawLive = (current: Extract<Gesture, { kind: "draw" }>) => {
      livePathRef.current?.setAttribute("d", inkOutlinePath(current.points, current.width, current.highlight))
    }

    const eraseAt = (clientX: number, clientY: number) => {
      const current = liveRef.current.doc
      const rects = new Map(pageBoxes().map(({ page, rect }) => [page, rect]))
      let changed = false
      const kept: StrokeData[] = []
      for (const [page, pageStrokes] of strokesByPage(current.strokes)) {
        const rect = page === undefined ? undefined : rects.get(page)
        const outsidePage = !rect || (clipToPages && (clientX < rect.left || clientX > rect.right || clientY < rect.top || clientY > rect.bottom))
        if (outsidePage) { kept.push(...pageStrokes); continue }
        const [x, y] = toPagePoint(rect, clientX, clientY)
        const survivors = strokesAfterErase(pageStrokes, x, y, ERASER_RADIUS_PX * PAGE_UNITS / rect.width)
        if (survivors.length !== pageStrokes.length) changed = true
        kept.push(...survivors)
      }
      if (!changed) return
      snapshotOnce()
      applyChange({ ...current, strokes: kept })
    }

    /** Which gesture a press starts, once the annotations have loaded. */
    const gestureKind = (event: PointerEvent): "draw" | "erase" | null => {
      const { ink: { tool: selected }, loaded: ready } = settingsRef.current
      return ready ? pointerInkRole(event, selected) : null
    }

    const onPointerDown = (event: PointerEvent) => {
      const kind = gestureKind(event)
      if (!kind || gesture) return
      if (kind === "erase") {
        gesture = { kind, pointerId: event.pointerId }
        eraseAt(event.clientX, event.clientY)
      } else {
        const target = pageAt(pageBoxes(), event.clientX, event.clientY, clipToPages)
        const page = target && scroller.querySelector<HTMLElement>(`[data-page-number="${target.page}"]`)
        if (!target || !page) return
        const { ink: settings, pen: hostPen } = settingsRef.current
        const host = !settings.tool ? hostPen : undefined
        const highlight = host ? host.highlight : settings.highlighting
        const color = host ? host.color : settings.color
        const width = host ? host.width : settings.width
        gesture = { kind, pointerId: event.pointerId, page, pageNumber: target.page, points: [pointerSample(event, (x, y) => toPagePoint(target.rect, x, y))], highlight, color, width }
        setLive({ page: target.page, highlight, color })
      }
      // No text selection, no focus change, no native scroll for this pointer.
      event.preventDefault()
      event.stopPropagation()
      scroller.setPointerCapture(event.pointerId)
    }

    const onPointerMove = (event: PointerEvent) => {
      if (!gesture || event.pointerId !== gesture.pointerId) return
      if (gesture.kind === "erase") { eraseAt(event.clientX, event.clientY); return }
      // The page can move under a held pointer (wheel scroll), so map every sample against its current rect.
      const rect = gesture.page.getBoundingClientRect()
      appendPointerSamples(gesture.points, event, (x, y) => toPagePoint(rect, x, y))
      redrawLive(gesture)
    }

    const onPointerEnd = (event: PointerEvent) => {
      if (!gesture || event.pointerId !== gesture.pointerId) return
      const finished = gesture
      gesture = null
      if (scroller.hasPointerCapture(event.pointerId)) scroller.releasePointerCapture(event.pointerId)
      if (finished.kind === "erase") { endGesture(); return }
      // pointercancel carries no usable final position, the last move sample is the end.
      const points = finishStroke(finished.points, event, (x, y) => toPagePoint(finished.page.getBoundingClientRect(), x, y))
      setLive(null)
      if (!points) return
      const stroke: StrokeData = { ...newStroke(points, finished), page: finished.pageNumber }
      const current = liveRef.current.doc
      commit({ ...current, strokes: [...current.strokes, stroke] })
    }

    // A hovering stylus disables native panning, so its strokes never scroll the page. Fingers scroll again once it leaves.
    const onPointerOver = (event: PointerEvent) => { if (event.pointerType === "pen") scroller.style.touchAction = "none" }
    const onPointerLeave = (event: PointerEvent) => { if (event.pointerType === "pen" && !gesture) scroller.style.touchAction = "" }

    scroller.addEventListener("pointerdown", onPointerDown)
    scroller.addEventListener("pointermove", onPointerMove)
    scroller.addEventListener("pointerup", onPointerEnd)
    scroller.addEventListener("pointercancel", onPointerEnd)
    scroller.addEventListener("lostpointercapture", onPointerEnd)
    scroller.addEventListener("pointerover", onPointerOver)
    scroller.addEventListener("pointerleave", onPointerLeave)
    return () => {
      scroller.removeEventListener("pointerdown", onPointerDown)
      scroller.removeEventListener("pointermove", onPointerMove)
      scroller.removeEventListener("pointerup", onPointerEnd)
      scroller.removeEventListener("pointercancel", onPointerEnd)
      scroller.removeEventListener("lostpointercapture", onPointerEnd)
      scroller.removeEventListener("pointerover", onPointerOver)
      scroller.removeEventListener("pointerleave", onPointerLeave)
      scroller.style.touchAction = ""
    }
  }, [scroller, clipToPages, liveRef, applyChange, commit, snapshotOnce, endGesture])

  // A selected tool turns the mouse into that tool over the pages.
  const tool = ink.tool
  useEffect(() => {
    if (!scroller || !tool) return
    scroller.classList.add("cursor-crosshair")
    return () => scroller.classList.remove("cursor-crosshair")
  }, [scroller, tool])

  const hasToolbar = useHasTopBarSlot("pdf")
  useEscapeLayer(ink.clearTool, hasToolbar && tool !== null)

  // Undo and redo belong to the top-bar owner, a canvas host keeps its own.
  useUndoRedoKeys(undo, redo, hasToolbar && active, true)

  const renderPageOverlay = useCallback((page: number, pageWidth: number) => (
    <PageInk
      strokes={pages.get(page)}
      scale={pageWidth / PAGE_UNITS}
      clip={clipToPages}
      live={live?.page === page ? live : null}
      livePathRef={livePathRef}
    />
  ), [pages, clipToPages, live])

  // Layout effect: the viewer re-renders with the new ink before the browser paints.
  useLayoutEffect(() => {
    onPageOverlay(renderPageOverlay)
    return () => onPageOverlay(undefined)
  }, [onPageOverlay, renderPageOverlay])

  const saveFailed = status === "error" || status === "offline"

  return (
    <InTopBar name="pdf">
      <div className="flex items-center gap-1">
        <InkTools ink={ink} />
        {SEPARATOR}
        <UndoRedoButtons onUndo={undo} onRedo={redo} canUndo={canUndo} canRedo={canRedo} />
        {saveFailed && <span role="status" className="ml-1 text-[10px] text-danger">Annotations not saved</span>}
      </div>
    </InTopBar>
  )
})

/** One page's ink: highlights blend into the page under the pen strokes. Unclipped strokes spill into the margins. */
const PageInk = memo(function PageInk({ strokes, scale, clip, live, livePathRef }: {
  strokes: StrokeData[] | undefined
  scale: number
  clip: boolean
  live: LiveStroke | null
  livePathRef: RefObject<SVGPathElement | null>
}) {
  if (!strokes && !live) return null
  const { highlights, pen } = splitHighlights(strokes ?? [])
  const layer = cn("pdf-ink absolute inset-0 h-full w-full pointer-events-none", clip ? "overflow-hidden" : "overflow-visible")
  const transform = `scale(${scale})`
  const livePath = (highlight: boolean) => live && live.highlight === highlight && (
    <path ref={livePathRef} fill={live.color} className={inkClass(live.color)} />
  )
  return (
    <>
      <svg className={cn(layer, "ink-highlight")} aria-hidden>
        <g transform={transform}>
          {highlights.map((stroke) => <path key={stroke.id} d={strokePath(stroke)} fill={stroke.color} />)}
          {livePath(true)}
        </g>
      </svg>
      <svg className={layer} aria-hidden>
        <g transform={transform}>
          {pen.map((stroke) => <path key={stroke.id} d={strokePath(stroke)} fill={stroke.color} className={inkClass(stroke.color)} />)}
          {livePath(false)}
        </g>
      </svg>
    </>
  )
})

function inkClass(color: string): string | undefined {
  return color === DEFAULT_PEN_COLOR ? "pdf-ink-default" : undefined
}
