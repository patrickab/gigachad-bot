"use client"

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react"
import { Document, Page, pdfjs } from "react-pdf"
import { Minus, Plus } from "lucide-react"
import "react-pdf/dist/esm/Page/AnnotationLayer.css"
import "react-pdf/dist/esm/Page/TextLayer.css"

pdfjs.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs"

const RENDER_WIDTH = 900
const PAGE_GAP = 1
const MIN_ZOOM = 1
const MAX_ZOOM = 2.5
const ZOOM_STEP = 0.25

interface ScrollAnchor {
  mouseX: number
  mouseY: number
  scrollLeft: number
  scrollTop: number
  ratio: number
}

export function PdfViewer({ url }: { url: string }) {
  const [numPages, setNumPages] = useState<number | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const [containerWidth, setContainerWidth] = useState<number | null>(null)
  const [zoom, setZoom] = useState(1)
  const zoomRef = useRef(1)
  const anchorRef = useRef<ScrollAnchor | null>(null)
  const hideTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [controlsVisible, setControlsVisible] = useState(false)

  useEffect(() => {
    const el = scrollRef.current
    if (!el) return

    const update = () => {
      const w = el.clientWidth
      if (w > 0) setContainerWidth(w)
    }

    update()

    const ro = new ResizeObserver(update)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const showControls = useCallback(() => {
    if (hideTimerRef.current) clearTimeout(hideTimerRef.current)
    setControlsVisible(true)
    hideTimerRef.current = setTimeout(() => setControlsVisible(false), 1500)
  }, [])

  useEffect(() => {
    const el = scrollRef.current
    if (!el) return

    const onWheel = (e: WheelEvent) => {
      if (!e.ctrlKey && !e.metaKey) return
      e.preventDefault()

      const prevZoom = zoomRef.current
      const factor = e.deltaY < 0 ? 1.08 : 1 / 1.08
      const nextZoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, prevZoom * factor))
      if (nextZoom === prevZoom) return

      const rect = el.getBoundingClientRect()
      anchorRef.current = {
        mouseX: e.clientX - rect.left,
        mouseY: e.clientY - rect.top,
        scrollLeft: el.scrollLeft,
        scrollTop: el.scrollTop,
        ratio: nextZoom / prevZoom,
      }

      zoomRef.current = nextZoom
      setZoom(nextZoom)
      showControls()
    }

    el.addEventListener("wheel", onWheel, { passive: false })
    return () => el.removeEventListener("wheel", onWheel)
  }, [showControls])

  useLayoutEffect(() => {
    const anchor = anchorRef.current
    if (!anchor) return
    anchorRef.current = null

    const el = scrollRef.current
    if (!el) return

    el.scrollLeft = anchor.scrollLeft * anchor.ratio + anchor.mouseX * (anchor.ratio - 1)
    el.scrollTop = anchor.scrollTop * anchor.ratio + anchor.mouseY * (anchor.ratio - 1)
  }, [zoom])

  const adjustZoom = useCallback((nextZoom: number) => {
    const el = scrollRef.current
    if (!el) return

    const prevZoom = zoomRef.current
    const clamped = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, nextZoom))
    if (clamped === prevZoom) return

    const rect = el.getBoundingClientRect()
    anchorRef.current = {
      mouseX: rect.width / 2,
      mouseY: rect.height / 2,
      scrollLeft: el.scrollLeft,
      scrollTop: el.scrollTop,
      ratio: clamped / prevZoom,
    }

    zoomRef.current = clamped
    setZoom(clamped)
    showControls()
  }, [showControls])

  const onDocumentLoadSuccess = useCallback(({ numPages }: { numPages: number }) => {
    setNumPages(numPages)
  }, [])

  const baseScale = containerWidth != null ? Math.min(1, (containerWidth - 16) / RENDER_WIDTH) : null
  const renderWidth = baseScale != null ? Math.round(RENDER_WIDTH * baseScale) : RENDER_WIDTH

  const zoomLabel = `${Math.round(zoom * 100)}%`

  return (
    <div
      className="relative w-full h-full min-h-[60vh] flex flex-col"
      onMouseEnter={showControls}
      onMouseMove={showControls}
    >
      <div
        className={`absolute bottom-3 left-1/2 -translate-x-1/2 z-10 flex items-center gap-1 rounded-full bg-white shadow-[0_1px_6px_rgba(0,0,0,0.12)] px-2 py-1 transition-opacity duration-300 pointer-events-auto ${controlsVisible ? "opacity-100" : "opacity-0 pointer-events-none"}`}
      >
        <button
          onClick={() => adjustZoom(zoomRef.current - ZOOM_STEP)}
          disabled={zoom <= MIN_ZOOM}
          className="rounded-full p-1 text-slate-400 hover:text-slate-700 hover:bg-slate-100 disabled:text-slate-300 disabled:pointer-events-none transition-colors"
        >
          <Minus className="h-3.5 w-3.5" strokeWidth={2} />
        </button>
        <span className="text-[10px] font-medium tabular-nums text-slate-500 w-9 text-center select-none">{zoomLabel}</span>
        <button
          onClick={() => adjustZoom(zoomRef.current + ZOOM_STEP)}
          disabled={zoom >= MAX_ZOOM}
          className="rounded-full p-1 text-slate-400 hover:text-slate-700 hover:bg-slate-100 disabled:text-slate-300 disabled:pointer-events-none transition-colors"
        >
          <Plus className="h-3.5 w-3.5" strokeWidth={2} />
        </button>
      </div>
      <div ref={scrollRef} className="flex-1 min-h-0 overflow-auto">
        <Document file={url} onLoadSuccess={onDocumentLoadSuccess}>
          {baseScale != null && numPages != null && (
            <div style={{ width: renderWidth * zoom }}>
              <div style={{ transform: `scale(${zoom})`, transformOrigin: "0 0" }}>
                {Array.from({ length: numPages }, (_, i) => (
                  <div key={i} style={{ marginBottom: PAGE_GAP }}>
                    <Page pageNumber={i + 1} width={renderWidth} />
                  </div>
                ))}
              </div>
            </div>
          )}
        </Document>
      </div>
    </div>
  )
}