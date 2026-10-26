"use client"

import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type ComponentProps, type RefObject } from "react"
import { createPortal } from "react-dom"
import { Redo2, Undo2 } from "lucide-react"
import { PdfViewer } from "@/components/content/PdfViewer"
import { useTabActive } from "@/contexts/TabContext"
import { InkTools, SEPARATOR, useInkSettings } from "@/features/editor/canvas/CanvasToolbar"
import { DEFAULT_PEN_COLOR, appendStrokePoint, strokePath, strokesAfterErase } from "@/features/editor/canvas/ink/canvasInk"
import { useCanvasHistory } from "@/features/editor/canvas/useCanvasHistory"
import { useCollaborativeCanvas } from "@/features/editor/canvas/useCollaborativeCanvas"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import { openPdfAnnotations } from "@/lib/backend/api"
import { fileViewerRawUrl } from "@/lib/backend/urls"
import { canvasEntityId, emptyCanvasDoc } from "@/lib/domain/canvasDocument"
import { inkOutlinePath, penPressure } from "@/lib/drawing"
import { cn } from "@/lib/utils"
import { PAGE_UNITS, pageAt, strokesByPage, toPagePoint, type PageBox, type PdfStroke } from "./pdfInk"

const ERASER_RADIUS_PX = 15
const EMPTY_DOC = emptyCanvasDoc()

/** Ink a stylus writes with when no tool of the viewer's own is selected. */
type HostPen = { color: string; width: number; highlight: boolean }

type ViewerProps = Omit<ComponentProps<typeof PdfViewer>, "url" | "renderPageOverlay" | "onScrollerChange">

interface AnnotatedPdfViewerProps extends ViewerProps {
  path: string
  /** Top-bar element the annotation tools render into. Without it the stylus writes with `pen` and nothing else annotates. */
  toolbarSlot?: HTMLElement | null
  /** Stylus ink when there is no toolbar, e.g. the host canvas's pen or highlighter. */
  pen?: HostPen
  /** Show and accept ink on the pages only, margin notes stay hidden. */
  clipToPages?: boolean
}

/** PDF viewer with ink and highlighter annotations, shared by every copy of the same PDF. Falls back to the plain viewer when annotations are unavailable. */
export function AnnotatedPdfViewer({ path, toolbarSlot, pen, clipToPages = false, ...viewerProps }: AnnotatedPdfViewerProps) {
  const [annotations, setAnnotations] = useState<{ path: string; key: string | null } | null>(null)

  useEffect(() => {
    let cancelled = false
    openPdfAnnotations(path)
      .then((key) => { if (!cancelled) setAnnotations({ path, key }) })
      .catch(() => { if (!cancelled) setAnnotations({ path, key: null }) })
    return () => { cancelled = true }
  }, [path])

  // Wait for the key, so the PDF loads once instead of again when ink arrives.
  if (annotations?.path !== path) return null
  if (!annotations.key) return <PdfViewer url={fileViewerRawUrl(path)} {...viewerProps} />
  return <InkedPdfViewer canvasKey={annotations.key} path={path} toolbarSlot={toolbarSlot} pen={pen} clipToPages={clipToPages} viewerProps={viewerProps} />
}

type Gesture =
  | { kind: "draw"; pointerId: number; page: HTMLElement; pageNumber: number; points: number[][]; highlight: boolean; color: string; width: number }
  | { kind: "erase"; pointerId: number }

/** Stroke being drawn, rendered by its page's overlay. */
type LiveStroke = { page: number; highlight: boolean; color: string }

function InkedPdfViewer({ canvasKey, path, toolbarSlot, pen, clipToPages, viewerProps }: {
  canvasKey: string
  path: string
  toolbarSlot?: HTMLElement | null
  pen?: HostPen
  clipToPages: boolean
  viewerProps: ViewerProps
}) {
  const active = useTabActive()
  const { document, replace, status } = useCollaborativeCanvas(canvasKey)
  const doc = document ?? EMPTY_DOC
  const { liveRef, applyChange, commit, snapshotOnce, endGesture, undo, redo, canUndo, canRedo } = useCanvasHistory(doc, replace)
  const loaded = document !== null

  const [scroller, setScroller] = useState<HTMLDivElement | null>(null)
  const ink = useInkSettings()
  const [live, setLive] = useState<LiveStroke | null>(null)
  const livePathRef = useRef<SVGPathElement>(null)

  const pages = useMemo(() => strokesByPage(doc.strokes as PdfStroke[]), [doc.strokes])

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
      const strokes = current.strokes as PdfStroke[]
      const rects = new Map(pageBoxes().map(({ page, rect }) => [page, rect]))
      let changed = false
      const kept: PdfStroke[] = []
      for (const [page, pageStrokes] of strokesByPage(strokes)) {
        const rect = rects.get(page)
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

    /** Which gesture a press starts: the pen's eraser end always erases, the stylus writes even without a tool, the mouse needs one. */
    const gestureKind = (event: PointerEvent): "draw" | "erase" | null => {
      const { ink: { tool: selected }, loaded: ready } = settingsRef.current
      if (!ready || event.pointerType === "touch") return null
      if (event.pointerType === "pen" && (event.button === 5 || (event.buttons & 32) !== 0)) return "erase"
      if (event.button !== 0) return null
      if (event.pointerType === "mouse" && !selected) return null
      return selected === "eraser" ? "erase" : "draw"
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
        gesture = { kind, pointerId: event.pointerId, page, pageNumber: target.page, points: [[...toPagePoint(target.rect, event.clientX, event.clientY), penPressure(event)]], highlight, color, width }
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
      // Coalesced events carry the pen's full sample rate instead of one point per frame.
      for (const sample of event.getCoalescedEvents?.() ?? [event]) {
        appendStrokePoint(gesture.points, [...toPagePoint(rect, sample.clientX, sample.clientY), penPressure(sample)])
      }
      redrawLive(gesture)
    }

    const onPointerEnd = (event: PointerEvent) => {
      if (!gesture || event.pointerId !== gesture.pointerId) return
      const finished = gesture
      gesture = null
      if (scroller.hasPointerCapture(event.pointerId)) scroller.releasePointerCapture(event.pointerId)
      if (finished.kind === "erase") { endGesture(); return }
      // pointercancel carries no usable final position, the last move sample is the end.
      if (event.type === "pointerup") {
        appendStrokePoint(finished.points, [...toPagePoint(finished.page.getBoundingClientRect(), event.clientX, event.clientY), penPressure(event)])
      }
      setLive(null)
      if (finished.points.length < 2) return
      const stroke: PdfStroke = { id: canvasEntityId(), page: finished.pageNumber, points: finished.points, color: finished.color, width: finished.width, ...(finished.highlight && { highlight: true }) }
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

  const hasToolbar = !!toolbarSlot
  useEscapeLayer(ink.clearTool, hasToolbar && active && tool !== null)

  // Undo and redo belong to the top-bar owner, a canvas host keeps its own.
  useEffect(() => {
    if (!hasToolbar || !active) return
    const onKeyDown = (event: KeyboardEvent) => {
      if (!(event.ctrlKey || event.metaKey) || isEditable(event.target)) return
      const key = event.key.toLowerCase() // Shift uppercases it, so Ctrl+Shift+Z arrives as "Z"
      if (key === "z" && !event.shiftKey) {
        event.preventDefault()
        undo()
      } else if (key === "y" || event.code === "KeyY" || (key === "z" && event.shiftKey)) {
        event.preventDefault()
        redo()
      }
    }
    window.addEventListener("keydown", onKeyDown)
    return () => window.removeEventListener("keydown", onKeyDown)
  }, [hasToolbar, active, undo, redo])

  const renderPageOverlay = useCallback((page: number, pageWidth: number) => (
    <PageInk
      strokes={pages.get(page)}
      scale={pageWidth / PAGE_UNITS}
      clip={clipToPages}
      live={live?.page === page ? live : null}
      livePathRef={livePathRef}
    />
  ), [pages, clipToPages, live])

  const saveFailed = status === "error" || status === "offline"

  return (
    <>
      <PdfViewer url={fileViewerRawUrl(path)} {...viewerProps} renderPageOverlay={renderPageOverlay} onScrollerChange={setScroller} />
      {toolbarSlot && createPortal(
        <div className="flex items-center gap-1">
          <InkTools ink={ink} />
          {SEPARATOR}
          <button aria-label="Undo" onClick={undo} disabled={!canUndo} className="rounded p-1 text-ink-subtle hover:text-ink hover:bg-hover disabled:opacity-30 transition-colors">
            <Undo2 className="h-3.5 w-3.5" />
          </button>
          <button aria-label="Redo" onClick={redo} disabled={!canRedo} className="rounded p-1 text-ink-subtle hover:text-ink hover:bg-hover disabled:opacity-30 transition-colors">
            <Redo2 className="h-3.5 w-3.5" />
          </button>
          {saveFailed && <span role="status" className="ml-1 text-[10px] text-danger">Annotations not saved</span>}
        </div>,
        toolbarSlot,
      )}
    </>
  )
}

function isEditable(target: EventTarget | null): boolean {
  return target instanceof HTMLElement && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)
}

/** One page's ink: highlights blend into the page under the pen strokes. Unclipped strokes spill into the margins. */
const PageInk = memo(function PageInk({ strokes, scale, clip, live, livePathRef }: {
  strokes: PdfStroke[] | undefined
  scale: number
  clip: boolean
  live: LiveStroke | null
  livePathRef: RefObject<SVGPathElement | null>
}) {
  if (!strokes && !live) return null
  const layer = cn("pdf-ink absolute inset-0 h-full w-full pointer-events-none", clip ? "overflow-hidden" : "overflow-visible")
  const transform = `scale(${scale})`
  const livePath = (highlight: boolean) => live && live.highlight === highlight && (
    <path ref={livePathRef} fill={live.color} className={inkClass(live.color)} />
  )
  return (
    <>
      <svg className={cn(layer, "ink-highlight")} aria-hidden>
        <g transform={transform}>
          {strokes?.map((stroke) => stroke.highlight && <path key={stroke.id} d={strokePath(stroke)} fill={stroke.color} />)}
          {livePath(true)}
        </g>
      </svg>
      <svg className={layer} aria-hidden>
        <g transform={transform}>
          {strokes?.map((stroke) => !stroke.highlight && <path key={stroke.id} d={strokePath(stroke)} fill={stroke.color} className={inkClass(stroke.color)} />)}
          {livePath(false)}
        </g>
      </svg>
    </>
  )
})

function inkClass(color: string): string | undefined {
  return color === DEFAULT_PEN_COLOR ? "pdf-ink-default" : undefined
}
