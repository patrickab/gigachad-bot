"use client"

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, useSyncExternalStore } from "react"
import { createPortal, flushSync } from "react-dom"
import { Document, Page, pdfjs, type DocumentProps } from "react-pdf"
import { Maximize2, Minimize2, Minus, Moon, Plus, Sun } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"
import "react-pdf/dist/esm/Page/AnnotationLayer.css"
import "react-pdf/dist/esm/Page/TextLayer.css"
import { switchColors } from "@/lib/themeTransition"

pdfjs.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs"

// React-PDF logs expected text-layer cancellations as errors. Suppress only that message.
if (typeof window !== "undefined") {
  const patchFlag = console.error as typeof console.error & { __pdfTextLayerPatched?: boolean }
  if (!patchFlag.__pdfTextLayerPatched) {
    const originalError = console.error.bind(console)
    const patched = (...args: unknown[]) => {
      if (typeof args[0] === "string" && args[0].includes("TextLayer task cancelled")) return
      originalError(...args)
    }
    ;(patched as typeof console.error & { __pdfTextLayerPatched?: boolean }).__pdfTextLayerPatched = true
    console.error = patched
  }
}

// Max render width in px, and vertical gap between pages in px.
const RENDER_WIDTH = 900
const PAGE_GAP = 12
const CONTROLS_PADDING = 12
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

type PdfDocument = Parameters<NonNullable<DocumentProps["onLoadSuccess"]>>[0]
type PdfPage = Awaited<ReturnType<PdfDocument["getPage"]>>

// Walk a page's operator list and return the bounding box of every raster image
// (XObject / inline image). Used to punch images out of the theme color layers (and
// the dark-mode invert) so figures keep their true colors. ponytail: skips image *masks* (often 1-bit text
// stencils); add OPS.paintImageMaskXObject if photos render as masks somewhere.
async function collectImageRects(page: PdfPage): Promise<ImageRect[]> {
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

/** Re-attach a one-shot media query per DPR value, since each query only matches the DPR it was built for. */
function subscribeDevicePixelRatio(onChange: () => void) {
  let query: MediaQueryList
  const listen = () => {
    query = window.matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`)
    query.addEventListener("change", handleChange, { once: true })
  }
  const handleChange = () => {
    listen()
    onChange()
  }
  listen()
  return () => query.removeEventListener("change", handleChange)
}

/** Live `window.devicePixelRatio`, it changes with browser zoom and when moving between monitors. */
function useDevicePixelRatio(): number {
  return useSyncExternalStore(subscribeDevicePixelRatio, () => window.devicePixelRatio || 1, () => 1)
}

/** Size the canvas to exactly bitmap / DPR so every bitmap pixel lands on one device pixel.
 *
 * - react-pdf floors the CSS size independently of the truncated bitmap size
 * - At fractional DPRs (1.25, 1.5) that mismatch makes the browser resample, i.e. blur
 */
function mapCanvasToDevicePixels(canvas: HTMLCanvasElement | null | undefined, devicePixelRatio: number) {
  if (!canvas) return
  canvas.style.width = `${canvas.width / devicePixelRatio}px`
  canvas.style.height = `${canvas.height / devicePixelRatio}px`
}

/** CSS mask that keeps a page-wide color layer off `rects`, so raster images keep their full range. */
function imageHoleMask(rects: ImageRect[]): React.CSSProperties | undefined {
  if (!rects.length) return undefined
  const holes = rects.map((r) => `M${r.left} ${r.top}h${r.width}v${r.height}h${-r.width}Z`).join("")
  const svg = `<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1 1' preserveAspectRatio='none'><path fill-rule='evenodd' d='M0 0H1V1H0Z${holes}'/></svg>`
  const image = `url("data:image/svg+xml,${encodeURIComponent(svg)}")`
  return { maskImage: image, WebkitMaskImage: image, maskSize: "100% 100%", WebkitMaskSize: "100% 100%" }
}

function PdfViewerInner({
  url,
  isFullscreen,
  onToggleFullscreen,
  fitWidth,
  onPageAspect,
  initialPage = 1,
  onPageChange,
  topInset,
}: {
  url: string
  isFullscreen: boolean
  onToggleFullscreen: () => void
  // when set, the parent controls width and the viewer sizes its own height to
  // fit exactly one page. Otherwise it fills its parent in both dimensions.
  fitWidth?: boolean
  onPageAspect?: (ratio: number) => void
  initialPage?: number
  onPageChange?: (page: number) => void
  topInset: number
}) {
  const [numPages, setNumPages] = useState<number | null>(null)
  const [imageRects, setImageRects] = useState<Record<number, ImageRect[]>>({})
  // pages whose canvas has drawn once; masks over an empty page would show as light boxes
  const [renderedPages, setRenderedPages] = useState<ReadonlySet<number>>(() => new Set())
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
  const [lightPaper, setLightPaper] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)
  const restoredPageUrlRef = useRef<string | null>(null)
  const pageRef = useRef(1)
  const onPageChangeRef = useRef(onPageChange)
  useLayoutEffect(() => {
    onPageChangeRef.current = onPageChange
  })
  const pageRefs = useRef(new Map<number, HTMLDivElement>())
  const devicePixelRatio = useDevicePixelRatio()

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

  /** Zooms to `nextZoom`, keeping the client point under it (default: viewport center) fixed. */
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
    showControls()
  }, [showControls])

  useEffect(() => {
    const el = scrollRef.current
    if (!el) return

    // Ctrl/Cmd + wheel zooms, anchored at the cursor
    const onWheel = (e: WheelEvent) => {
      if (!e.ctrlKey && !e.metaKey) return
      e.preventDefault()
      zoomAt(zoomRef.current * (e.deltaY < 0 ? 1.08 : 1 / 1.08), { x: e.clientX, y: e.clientY })
    }

    el.addEventListener("wheel", onWheel, { passive: false })
    return () => el.removeEventListener("wheel", onWheel)
  }, [zoomAt])

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

  // Re-scroll after a zoom so the point under the cursor (or viewport center) stays put.
  useLayoutEffect(() => {
    const anchor = anchorRef.current
    if (!anchor) return
    anchorRef.current = null

    const el = scrollRef.current
    if (!el) return

    el.scrollLeft = anchor.scrollLeft * anchor.ratio + anchor.mouseX * (anchor.ratio - 1)
    el.scrollTop = anchor.scrollTop * anchor.ratio + anchor.mouseY * (anchor.ratio - 1)
  }, [zoom])

  const onDocumentLoadSuccess = useCallback(
    async (pdf: PdfDocument) => {
      setNumPages(pdf.numPages)
      setImageRects({})
      setRenderedPages(new Set())
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

  useLayoutEffect(() => {
    restoredPageUrlRef.current = null
    pageRef.current = 1
  }, [url])

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

  // Keep the document spot at the viewport's top edge fixed across width
  // changes (container resize). Pages scale with width but the PAGE_GAPs
  // don't, so map scrollTop exactly: page index + fraction within that page,
  // re-projected onto the new page height. Runs before paint; the minHeight
  // floor below guarantees the new scrollHeight already exists, so the
  // assignment can't be clamped away. Requires overflow-anchor: none on the
  // scroller — otherwise native scroll anchoring makes its own correction and
  // this one lands past the right spot.
  const prevRenderWidthRef = useRef(renderWidth)
  useLayoutEffect(() => {
    const prev = prevRenderWidthRef.current
    prevRenderWidthRef.current = renderWidth
    const el = scrollRef.current
    if (!el || prev === renderWidth || prev <= 0) return
    if (pageAspect != null) {
      const pageOld = prev * zoom * pageAspect
      const pageNew = renderWidth * zoom * pageAspect
      const page = Math.floor(el.scrollTop / (pageOld + PAGE_GAP))
      const frac = (el.scrollTop - page * (pageOld + PAGE_GAP)) / pageOld
      el.scrollTop = page * (pageNew + PAGE_GAP) + frac * pageNew
    } else {
      el.scrollTop *= renderWidth / prev
    }
    el.scrollLeft *= renderWidth / prev
    // eslint-disable-next-line react-hooks/exhaustive-deps -- reposition only when the width actually changes
  }, [renderWidth])

  // Scroll to `initialPage` once per url, after layout is ready.
  useLayoutEffect(() => {
    const el = scrollRef.current
    const page = Math.min(numPages ?? 1, Math.max(1, initialPage))
    const firstPageEl = pageRefs.current.get(1)
    const pageEl = pageRefs.current.get(page)
    if (!el || !ready || !firstPageEl || !pageEl || restoredPageUrlRef.current === url) return
    // offsetTop is in the scroller's layout coordinate system, unlike client rects
    // after the canvas scales this attachment with CSS.
    el.scrollTop = pageEl.offsetTop - firstPageEl.offsetTop
    pageRef.current = page
    restoredPageUrlRef.current = url
  }, [ready, url, initialPage, numPages, renderWidth, zoom, pageAspect])

  const trackVisiblePage = useCallback(() => {
    const el = scrollRef.current
    if (!el || numPages == null) return
    const top = el.getBoundingClientRect().top + topInset + 1
    let page = 1
    for (const [number, pageEl] of pageRefs.current) {
      if (pageEl.getBoundingClientRect().top > top) break
      page = number
    }
    if (page === pageRef.current) return
    pageRef.current = page
    onPageChangeRef.current?.(page)
  }, [numPages, topInset])

  const zoomLabel = `${Math.round(zoom * 100)}%`

  // fitWidth: fill parent width, self-size height to one page. Otherwise fill parent.
  const rootStyle: React.CSSProperties = fitWidth
    ? { width: "100%", height: boxHeight ?? "auto" }
    : { width: "100%", height: "100%", minHeight: 0 }

  // Controls sit CONTROLS_PADDING px inside the page's top-right corner (pages are centered in the scroller).
  const pageRightGap = availWidth == null ? 0 : (availWidth - Math.round(renderWidth * zoom)) / 2
  const controlsRight = Math.max(0, pageRightGap) + CONTROLS_PADDING

  // Holes only once a page has drawn: before that the hole would show the bare page background.
  const overlayMasks = useMemo(() => {
    const masks: Record<number, React.CSSProperties | undefined> = {}
    for (const page of renderedPages) masks[page] = imageHoleMask(imageRects[page] ?? [])
    return masks
  }, [imageRects, renderedPages])

  return (
    <div
      ref={rootRef}
      className={`relative flex flex-col${lightPaper ? " pdf-light-paper" : ""}`}
      onMouseEnter={showControls}
      onMouseMove={showControls}
      style={rootStyle}
    >
      <div
        ref={dropdownRef}
        style={{ top: topInset + CONTROLS_PADDING, right: controlsRight }}
        className={`absolute z-10 flex items-center gap-1 rounded-lg bg-surface-elevated/90 backdrop-blur-sm border border-divider/60 shadow-[var(--shadow-md)] transition-opacity duration-300 pointer-events-auto ${controlsVisible ? "opacity-100" : "opacity-0 pointer-events-none"}`}
      >
        <button aria-label="Zoom out"
          onClick={() => zoomAt(zoomRef.current - ZOOM_STEP)}
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
                      zoomAt(preset)
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
        <button aria-label="Zoom in"
          onClick={() => zoomAt(zoomRef.current + ZOOM_STEP)}
          disabled={zoom >= MAX_ZOOM}
          className="p-1.5 text-ink-subtle hover:text-ink hover:bg-hover disabled:text-ink-faint disabled:pointer-events-none transition-colors"
        >
          <Plus className="h-3.5 w-3.5" strokeWidth={2} />
        </button>
        <div className="pdf-paper-toggle flex items-center">
          <div className="w-px h-4 bg-divider" />
          <button
            aria-label={lightPaper ? "Dark PDF pages" : "Light PDF pages"}
            onClick={() => switchColors(() => flushSync(() => setLightPaper((light) => !light)))}
            className="p-1.5 text-ink-subtle hover:text-ink hover:bg-hover transition-colors"
          >
            {lightPaper ? <Moon className="h-3.5 w-3.5" /> : <Sun className="h-3.5 w-3.5" />}
          </button>
        </div>
        <div className="w-px h-4 bg-divider" />
        <button
          aria-label={isFullscreen ? "Exit PDF fullscreen" : "Enter PDF fullscreen"}
          onClick={onToggleFullscreen}
          className="rounded-r-lg p-1.5 text-ink-subtle hover:text-ink hover:bg-hover transition-colors"
        >
          {isFullscreen ? <Minimize2 className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
        </button>
      </div>
      <div
        ref={scrollRef}
        className="pdf-scroll flex-1 min-h-0 overflow-auto focus:outline-none"
        tabIndex={0}
        onScroll={trackVisiblePage}
        // overflow-anchor off: the width-change effect above owns the scroll
        // position; native scroll anchoring would apply a second correction
        style={{ overflowAnchor: "none", paddingTop: topInset, ...(fitWidth ? { scrollSnapType: "y proximity" as const } : undefined) }}
      >
        <Document file={url} onLoadSuccess={onDocumentLoadSuccess}>
          {ready && (
            <div style={{ width: Math.round(renderWidth * zoom) }} className="mx-auto">
              {Array.from({ length: numPages! }, (_, i) => (
                <div
                  key={i}
                  ref={(el) => {
                    if (el) pageRefs.current.set(i + 1, el)
                    else pageRefs.current.delete(i + 1)
                  }}
                  className="pdf-page relative"
                  // minHeight floor: react-pdf pages collapse to a tiny placeholder
                  // while re-rasterizing, which caves in scrollHeight and clamps
                  // scrollTop to 0 — "the PDF jumped back to page 1". ponytail:
                  // floor uses the first page's aspect; mixed page sizes drift a
                  // little but never collapse.
                  style={{
                    marginBottom: PAGE_GAP,
                    minHeight: pageAspect != null ? Math.round(renderWidth * zoom * pageAspect) : undefined,
                    scrollSnapAlign: fitWidth ? "start" : undefined,
                  }}
                  data-page-number={i + 1}
                >
                  <Page
                    pageNumber={i + 1}
                    width={Math.round(renderWidth * zoom)}
                    devicePixelRatio={devicePixelRatio}
                    onRenderSuccess={() => {
                      mapCanvasToDevicePixels(pageRefs.current.get(i + 1)?.querySelector("canvas"), devicePixelRatio)
                      setRenderedPages((pages) => pages.has(i + 1) ? pages : new Set(pages).add(i + 1))
                    }}
                    renderTextLayer
                  />
                  <div className="pdf-page-tint absolute inset-0 pointer-events-none" style={overlayMasks[i + 1]} />
                  <div className="pdf-page-ink absolute inset-0 pointer-events-none" style={overlayMasks[i + 1]} />
                  {renderedPages.has(i + 1) && (imageRects[i + 1] ?? []).map((r, k) => (
                    <div
                      key={k}
                      className="pdf-image-mask absolute pointer-events-none"
                      style={{ left: `${r.left * 100}%`, top: `${r.top * 100}%`, width: `${r.width * 100}%`, height: `${r.height * 100}%` }}
                    />
                  ))}
                </div>
              ))}
            </div>
          )}
        </Document>
      </div>
    </div>
  )
}

function subscribeNever() {
  return () => {}
}

function getDocumentBody() {
  return document.body
}

function getNoBody() {
  return null
}

/** PDF viewer with zoom, page tracking and fullscreen. Fullscreen reparents one mounted viewer via a portal.
 *
 * - `topInset` px of the viewer sit under a floating bar: pages start below it and scroll on underneath
 */
export function PdfViewer({
  url,
  fitWidth,
  onPageAspect,
  initialPage,
  onPageChange,
  topInset = 0,
}: {
  url: string
  fitWidth?: boolean
  onPageAspect?: (ratio: number) => void
  initialPage?: number
  onPageChange?: (page: number) => void
  topInset?: number
}) {
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [inlineEl, setInlineEl] = useState<HTMLDivElement | null>(null)
  const [overlayEl, setOverlayEl] = useState<HTMLDivElement | null>(null)
  // document.body exists only on the client, so the overlay is skipped during SSR and hydration.
  const body = useSyncExternalStore(subscribeNever, getDocumentBody, getNoBody)

  const toggleFullscreen = useCallback(() => setIsFullscreen((f) => !f), [])

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

  const inner = (
    <PdfViewerInner
      url={url}
      isFullscreen={isFullscreen}
      onToggleFullscreen={toggleFullscreen}
      fitWidth={isFullscreen ? undefined : fitWidth}
      onPageAspect={isFullscreen ? undefined : onPageAspect}
      initialPage={initialPage}
      onPageChange={onPageChange}
      topInset={isFullscreen ? 0 : topInset}
    />
  )

  const portalTarget = isFullscreen ? overlayEl : inlineEl

  // Keep one viewer mounted so fullscreen preserves its scroll and zoom state.
  return (
    <>
      <div ref={setInlineEl} className="contents" />
      {body && createPortal(
        <div
          ref={setOverlayEl}
          className="fixed inset-0 z-[100] bg-backdrop backdrop-blur-xl transition-opacity duration-150"
          style={{ opacity: isFullscreen ? 1 : 0, pointerEvents: isFullscreen ? "auto" : "none" }}
        />,
        body,
      )}
      {portalTarget && createPortal(inner, portalTarget)}
    </>
  )
}

/** Every page of a PDF stacked at `width` px, each at its own aspect ratio, without text or annotation layers. `onFirstPageAspect` gets page 1's height / width. */
export function PdfPages({ url, width, onFirstPageAspect }: { url: string; width: number; onFirstPageAspect?: (ratio: number) => void }) {
  const [pageCount, setPageCount] = useState(0)
  return (
    <Document file={url} loading={null} error={null} onLoadSuccess={({ numPages }) => setPageCount(numPages)}>
      <div className="flex flex-col gap-2">
        {Array.from({ length: pageCount }, (_, index) => (
          <Page
            key={index}
            pageNumber={index + 1}
            width={width}
            renderTextLayer={false}
            renderAnnotationLayer={false}
            loading={null}
            onLoadSuccess={index === 0 && onFirstPageAspect ? (page) => onFirstPageAspect(page.originalHeight / page.originalWidth) : undefined}
            className="overflow-hidden rounded"
          />
        ))}
      </div>
    </Document>
  )
}
