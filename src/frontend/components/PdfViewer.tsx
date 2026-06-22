"use client"

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react"
import { createPortal } from "react-dom"
import { Document, Page, pdfjs } from "react-pdf"
import { ChevronsLeftRight, ChevronsRightLeft, Maximize2, Minimize2, Minus, Plus } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"
import "react-pdf/dist/esm/Page/AnnotationLayer.css"

pdfjs.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs"

const RENDER_WIDTH = 900
const PAGE_GAP = 12
const MIN_ZOOM = 1
const MAX_ZOOM = 2.5
const ZOOM_STEP = 0.25
const ZOOM_PRESETS = [1, 1.25, 1.5, 1.75, 2, 2.5]

interface ScrollAnchor {
  mouseX: number
  mouseY: number
  scrollLeft: number
  scrollTop: number
  ratio: number
}

// image bbox as fractions of the page (0..1), so it survives zoom/width changes
interface ImageRect { left: number; top: number; width: number; height: number }

// Walk a page's operator list and return the bounding box of every raster image
// (XObject / inline image). Used to punch images out of the dark-mode invert so
// figures keep their true colors. ponytail: skips image *masks* (often 1-bit text
// stencils); add OPS.paintImageMaskXObject if photos render as masks somewhere.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function collectImageRects(page: any): Promise<ImageRect[]> {
  const ops = await page.getOperatorList()
  const vp = page.getViewport({ scale: 1 })
  const OPS = pdfjs.OPS
  type M = [number, number, number, number, number, number]
  const mul = (a: M, b: M): M => [
    a[0] * b[0] + a[2] * b[1], a[1] * b[0] + a[3] * b[1],
    a[0] * b[2] + a[2] * b[3], a[1] * b[2] + a[3] * b[3],
    a[0] * b[4] + a[2] * b[5] + a[4], a[1] * b[4] + a[3] * b[5] + a[5],
  ]
  const apply = (m: M, x: number, y: number): [number, number] => [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]]

  const rects: ImageRect[] = []
  let ctm: M = [1, 0, 0, 1, 0, 0]
  const stack: M[] = []
  for (let i = 0; i < ops.fnArray.length; i++) {
    const fn = ops.fnArray[i]
    if (fn === OPS.save) stack.push(ctm)
    else if (fn === OPS.restore) ctm = stack.pop() ?? ctm
    else if (fn === OPS.transform) ctm = mul(ctm, ops.argsArray[i] as M)
    else if (fn === OPS.paintImageXObject || fn === OPS.paintInlineImageXObject) {
      const dev = mul(vp.transform as M, ctm) // image fills the unit square under ctm
      const pts = [apply(dev, 0, 0), apply(dev, 1, 0), apply(dev, 0, 1), apply(dev, 1, 1)]
      const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1])
      const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys)
      rects.push({
        left: minX / vp.width,
        top: minY / vp.height,
        width: (maxX - minX) / vp.width,
        height: (maxY - minY) / vp.height,
      })
    }
  }
  return rects
}

// vertical scrollbar gutter reserved so a fitted page never triggers horizontal scroll
const SCROLLBAR = 16

function PdfViewerInner({
  url,
  isFullscreen,
  onToggleFullscreen,
  isWide,
  onToggleWide,
  fitWidth,
  onPageAspect,
}: {
  url: string
  isFullscreen: boolean
  onToggleFullscreen: () => void
  isWide?: boolean
  onToggleWide?: () => void
  // when set, the parent controls width and the viewer sizes its own height to
  // fit exactly one page. Otherwise it fills its parent in both dimensions.
  fitWidth?: boolean
  onPageAspect?: (ratio: number) => void
}) {
  const [numPages, setNumPages] = useState<number | null>(null)
  const [imageRects, setImageRects] = useState<Record<number, ImageRect[]>>({})
  const rootRef = useRef<HTMLDivElement>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const [availWidth, setAvailWidth] = useState<number | null>(null)
  const [pageAspect, setPageAspect] = useState<number | null>(null) // height / width
  const [zoom, setZoom] = useState(1)
  const zoomRef = useRef(1)
  const anchorRef = useRef<ScrollAnchor | null>(null)
  const hideTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [controlsVisible, setControlsVisible] = useState(false)
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)

  // measure the parent-controlled dimension — debounced so resizing doesn't re-render every frame
  useEffect(() => {
    const el = rootRef.current
    if (!el) return

    let timer: ReturnType<typeof setTimeout> | null = null
    const update = () => {
      if (el.clientWidth > 0) setAvailWidth(el.clientWidth)
    }

    update()

    const ro = new ResizeObserver(() => {
      if (timer) clearTimeout(timer)
      timer = setTimeout(update, 150)
    })
    ro.observe(el)
    return () => { ro.disconnect(); if (timer) clearTimeout(timer) }
  }, [])

  useEffect(() => {
    if (!dropdownOpen) return
    function handleClick(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setDropdownOpen(false)
      }
    }
    document.addEventListener("mousedown", handleClick)
    return () => document.removeEventListener("mousedown", handleClick)
  }, [dropdownOpen])

  const showControls = useCallback(() => {
    if (hideTimerRef.current) clearTimeout(hideTimerRef.current)
    setControlsVisible(true)
    hideTimerRef.current = setTimeout(() => {
      setControlsVisible(false)
      setDropdownOpen(false)
    }, 2500)
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

  useEffect(() => {
    const el = scrollRef.current
    if (!el || !numPages) return

    // arrow keys step one page from whichever page is currently nearest the top
    const onKeyDown = (e: KeyboardEvent) => {
      const dir = (e.key === "ArrowUp" || e.key === "ArrowLeft") ? -1
        : (e.key === "ArrowDown" || e.key === "ArrowRight") ? 1 : 0
      if (dir === 0) return
      e.preventDefault()
      const pageH = el.scrollHeight / numPages
      const current = pageH > 0 ? Math.round(el.scrollTop / pageH) : 0 // 0-based
      const target = Math.min(numPages - 1, Math.max(0, current + dir))
      if (target === current) return
      const node = el.querySelector(`[data-page-number="${target + 1}"]`) as HTMLElement | null
      node?.scrollIntoView({ behavior: "smooth", block: "start" })
    }

    el.addEventListener("keydown", onKeyDown)
    return () => el.removeEventListener("keydown", onKeyDown)
  }, [numPages])

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

  const onDocumentLoadSuccess = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    async (pdf: any) => {
      setNumPages(pdf.numPages)
      setImageRects({})
      // read the real first-page dimensions (works for any page size, not just A4)
      try {
        const page = await pdf.getPage(1)
        const vp = page.getViewport({ scale: 1 })
        const ratio = vp.height / vp.width
        setPageAspect(ratio)
        onPageAspect?.(ratio)
      } catch {
        /* leave pageAspect null → falls back to fill-parent rendering */
      }
      // collect raster-image regions per page so dark mode can leave them un-inverted.
      // ponytail: eager loop over all pages — fine for typical docs; lazy per-page on
      // scroll only if huge PDFs feel slow on open.
      for (let n = 1; n <= pdf.numPages; n++) {
        try {
          const rects = await collectImageRects(await pdf.getPage(n))
          if (rects.length) setImageRects((prev) => ({ ...prev, [n]: rects }))
        } catch { /* skip pages that fail to parse */ }
      }
    },
    [onPageAspect],
  )

  // Page is rendered at the available width (minus scrollbar gutter), capped at
  // RENDER_WIDTH. In fitWidth mode the box height is derived to show one page.
  let renderWidth = RENDER_WIDTH
  let boxHeight: number | undefined
  if (availWidth != null) {
    renderWidth = Math.round(Math.min(RENDER_WIDTH, availWidth - SCROLLBAR))
    if (fitWidth && pageAspect != null) {
      boxHeight = Math.round(renderWidth * pageAspect)
    }
  }

  const ready = numPages != null && availWidth != null && (!fitWidth || pageAspect != null)

  const zoomLabel = `${Math.round(zoom * 100)}%`

  // fitWidth: fill parent width, self-size height to one page. Otherwise fill parent.
  const rootStyle: React.CSSProperties = fitWidth
    ? { width: "100%", height: boxHeight ?? "auto" }
    : { width: "100%", height: "100%", minHeight: 0 }

  return (
    <div
      ref={rootRef}
      className="relative flex flex-col"
      onMouseEnter={showControls}
      onMouseMove={showControls}
      style={rootStyle}
    >
      <div
        ref={dropdownRef}
        className={`absolute top-3 right-3 z-10 flex items-center gap-1 rounded-lg bg-surface-elevated/90 backdrop-blur-sm border border-divider/60 shadow-[var(--shadow-md)] transition-opacity duration-300 pointer-events-auto ${controlsVisible ? "opacity-100" : "opacity-0 pointer-events-none"}`}
      >
        <button
          onClick={() => adjustZoom(zoomRef.current - ZOOM_STEP)}
          disabled={zoom <= MIN_ZOOM}
          className="rounded-l-lg p-1.5 text-ink-subtle hover:text-ink hover:bg-hover disabled:text-ink-faint disabled:pointer-events-none transition-colors"
        >
          <Minus className="h-3.5 w-3.5" strokeWidth={2} />
        </button>
        <div className="relative">
          <button
            onClick={() => setDropdownOpen((o) => !o)}
            className="px-1.5 py-1.5 text-[10px] font-medium tabular-nums text-ink-muted hover:text-ink hover:bg-hover transition-colors select-none"
          >
            {zoomLabel}
          </button>
          <AnimatePresence>
            {dropdownOpen && (
              <motion.div
                initial={{ opacity: 0, y: -4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                transition={{ duration: 0.12 }}
                className="absolute top-full left-1/2 -translate-x-1/2 mt-1 py-1 rounded-lg bg-surface-elevated border border-divider shadow-[var(--shadow-lg)] z-20"
              >
                {ZOOM_PRESETS.map((preset) => (
                  <button
                    key={preset}
                    onClick={() => {
                      adjustZoom(preset)
                      setDropdownOpen(false)
                    }}
                    className={`block w-full px-4 py-1 text-[10px] font-medium tabular-nums text-left hover:bg-hover transition-colors ${
                      Math.abs(zoom - preset) < 0.01 ? "text-ink" : "text-ink-muted"
                    }`}
                  >
                    {Math.round(preset * 100)}%
                  </button>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
        <button
          onClick={() => adjustZoom(zoomRef.current + ZOOM_STEP)}
          disabled={zoom >= MAX_ZOOM}
          className="p-1.5 text-ink-subtle hover:text-ink hover:bg-hover disabled:text-ink-faint disabled:pointer-events-none transition-colors"
        >
          <Plus className="h-3.5 w-3.5" strokeWidth={2} />
        </button>
        {onToggleWide && (
          <>
            <div className="w-px h-4 bg-divider" />
            <button
              onClick={onToggleWide}
              className="p-1.5 text-ink-subtle hover:text-ink hover:bg-hover transition-colors"
            >
              {isWide ? <ChevronsRightLeft className="h-3.5 w-3.5" /> : <ChevronsLeftRight className="h-3.5 w-3.5" />}
            </button>
          </>
        )}
        <div className="w-px h-4 bg-divider" />
        <button
          onClick={onToggleFullscreen}
          className="rounded-r-lg p-1.5 text-ink-subtle hover:text-ink hover:bg-hover transition-colors"
        >
          {isFullscreen ? <Minimize2 className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
        </button>
      </div>
      <div
        ref={scrollRef}
        className="flex-1 min-h-0 overflow-auto focus:outline-none"
        tabIndex={0}
        style={fitWidth ? { scrollSnapType: "y proximity" } : undefined}
      >
        <Document file={url} onLoadSuccess={onDocumentLoadSuccess}>
          {ready && (
            <div style={{ width: renderWidth * zoom }} className="mx-auto">
              <div style={{ transform: `scale(${zoom})`, transformOrigin: "0 0" }}>
                {Array.from({ length: numPages! }, (_, i) => (
                  <div
                    key={i}
                    className="pdf-page relative"
                    style={{ marginBottom: PAGE_GAP, scrollSnapAlign: fitWidth ? "start" : undefined }}
                    data-page-number={i + 1}
                  >
                    <Page pageNumber={i + 1} width={renderWidth} renderTextLayer={false} />
                    <div className="pdf-page-tint absolute inset-0 pointer-events-none" />
                    {(imageRects[i + 1] ?? []).map((r, k) => (
                      <div
                        key={k}
                        className="pdf-image-mask absolute pointer-events-none"
                        style={{ left: `${r.left * 100}%`, top: `${r.top * 100}%`, width: `${r.width * 100}%`, height: `${r.height * 100}%` }}
                      />
                    ))}
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

export function PdfViewer({
  url,
  isWide,
  onToggleWide,
  fitWidth,
  onPageAspect,
}: {
  url: string
  isWide?: boolean
  onToggleWide?: () => void
  fitWidth?: boolean
  onPageAspect?: (ratio: number) => void
}) {
  const [isFullscreen, setIsFullscreen] = useState(false)

  useEffect(() => {
    if (!isFullscreen) return
    function handler(e: KeyboardEvent) {
      if (e.key === "Escape") {
        e.preventDefault()
        setIsFullscreen(false)
      }
    }
    document.addEventListener("keydown", handler)
    return () => document.removeEventListener("keydown", handler)
  }, [isFullscreen])

  const toggleFullscreen = useCallback(() => setIsFullscreen((f) => !f), [])

  if (!isFullscreen) {
    return (
      <PdfViewerInner
        url={url}
        isFullscreen={false}
        onToggleFullscreen={toggleFullscreen}
        isWide={isWide}
        onToggleWide={onToggleWide}
        fitWidth={fitWidth}
        onPageAspect={onPageAspect}
      />
    )
  }

  return createPortal(
    <AnimatePresence>
      <motion.div
        key="pdf-fullscreen"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.15 }}
        className="fixed inset-0 z-[100] bg-backdrop backdrop-blur-xl"
      >
        <PdfViewerInner url={url} isFullscreen={true} onToggleFullscreen={toggleFullscreen} />
      </motion.div>
    </AnimatePresence>,
    document.body,
  )
}