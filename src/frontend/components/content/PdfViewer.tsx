"use client"

import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, useSyncExternalStore, type ReactNode } from "react"
import { flushSync } from "react-dom"
import { Document, Page, pdfjs, type DocumentProps } from "react-pdf"
import { Minus, Moon, Plus, Sun } from "lucide-react"
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
// controls show while the pointer is within this many px of them
const CONTROLS_VICINITY = 80
const MIN_ZOOM = 1
const MAX_ZOOM = 2.5
const ZOOM_STEP = 0.25
const ZOOM_PRESETS = [1, 1.25, 1.5, 1.75, 2, 2.5]
// zoom steps scale the current bitmaps, pages re-rasterize once the zoom rests this long
const ZOOM_SETTLE_MS = 150
// pages within one viewport height above or below the visible area keep a rendered canvas
const MOUNT_MARGIN = "100% 0px"

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

/** Bounding box of every link annotation, so link text keeps the PDF's own link color. */
async function collectLinkRects(page: PdfPage): Promise<ImageRect[]> {
  const vp = page.getViewport({ scale: 1 })
  const annotations = await page.getAnnotations()
  return annotations
    .filter((annotation) => annotation.subtype === "Link")
    .map((annotation) => {
      const [x1, y1, x2, y2] = vp.convertToViewportRectangle(annotation.rect)
      return {
        left: Math.min(x1, x2) / vp.width,
        top: Math.min(y1, y2) / vp.height,
        width: Math.abs(x2 - x1) / vp.width,
        height: Math.abs(y2 - y1) / vp.height,
      }
    })
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

/** CSS mask for a page-wide color layer: `holes` -> layer everywhere except `rects`, else -> layer only on `rects`. */
function rectMask(rects: ImageRect[], holes: boolean): React.CSSProperties | undefined {
  if (!rects.length) return holes ? undefined : { display: "none" }
  const shapes = rects.map((r) => `M${r.left} ${r.top}h${r.width}v${r.height}h${-r.width}Z`).join("")
  const path = holes ? `M0 0H1V1H0Z${shapes}` : shapes
  const svg = `<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1 1' preserveAspectRatio='none'><path fill-rule='evenodd' d='${path}'/></svg>`
  const image = `url("data:image/svg+xml,${encodeURIComponent(svg)}")`
  return { maskImage: image, WebkitMaskImage: image, maskSize: "100% 100%", WebkitMaskSize: "100% 100%" }
}

/** Free a canvas' bitmap and hide it. */
function releaseCanvas(canvas: HTMLCanvasElement | null) {
  if (!canvas) return
  canvas.width = 0
  canvas.height = 0
  canvas.style.display = "none"
}

/** Every page's height / width, `fallback` for pages that fail to load. */
async function readPageAspects(pdf: PdfDocument, fallback: number): Promise<number[]> {
  return Promise.all(Array.from({ length: pdf.numPages }, async (_, index) => {
    try {
      const viewport = (await pdf.getPage(index + 1)).getViewport({ scale: 1 })
      return viewport.height / viewport.width
    } catch {
      return fallback
    }
  }))
}

/** Top offset of every page at `pageWidth`, relative to the first page. */
function pageTops(aspects: number[], pageWidth: number): number[] {
  const tops: number[] = []
  let top = 0
  for (const aspect of aspects) {
    tops.push(top)
    top += Math.round(pageWidth * aspect) + PAGE_GAP
  }
  return tops
}

interface PageRegions { images: ImageRect[]; links: ImageRect[] }

/** One page's box at its final size, its canvas exists only while `mounted`.
 *
 * - `rasterWidth` is the canvas' render width, CSS-scaled to `width` while a zoom settles
 * - Re-rasterizing blanks the canvas, a backup copy of the last bitmap covers the page until the new one lands
 */
const PdfPageSlot = memo(function PdfPageSlot({
  pageNumber,
  width,
  height,
  rasterWidth,
  devicePixelRatio,
  mounted,
  snap,
  registerSlot,
  renderPageOverlay,
}: {
  pageNumber: number
  width: number
  height: number
  rasterWidth: number
  devicePixelRatio: number
  mounted: boolean
  snap: boolean
  registerSlot: (pageNumber: number, slot: HTMLDivElement | null) => void
  renderPageOverlay?: (page: number, pageWidth: number) => ReactNode
}) {
  const [regions, setRegions] = useState<PageRegions | null>(null)
  // masks over a blank canvas would show as light boxes
  const [drawn, setDrawn] = useState(false)
  if (!mounted && drawn) setDrawn(false)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const drawnCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const backupRef = useRef<HTMLCanvasElement>(null)
  const regionsRequestedRef = useRef(false)

  const slotRef = useCallback((slot: HTMLDivElement | null) => registerSlot(pageNumber, slot), [registerSlot, pageNumber])

  // Layout effects run before react-pdf's passive effects clear the canvas, so the last bitmap is still intact here.
  useLayoutEffect(() => {
    const drawnCanvas = drawnCanvasRef.current
    const backup = backupRef.current
    drawnCanvasRef.current = null
    // a backup still showing is from an earlier, unfinished re-render, keep it
    if (!drawnCanvas?.width || !backup || backup.width) return
    backup.width = drawnCanvas.width
    backup.height = drawnCanvas.height
    backup.getContext("2d")?.drawImage(drawnCanvas, 0, 0)
    backup.style.display = ""
  }, [rasterWidth, devicePixelRatio])

  useLayoutEffect(() => {
    if (mounted) return
    drawnCanvasRef.current = null
    releaseCanvas(backupRef.current)
  }, [mounted])

  const onRenderSuccess = useCallback(() => {
    mapCanvasToDevicePixels(canvasRef.current, devicePixelRatio)
    drawnCanvasRef.current = canvasRef.current
    releaseCanvas(backupRef.current)
    setDrawn(true)
  }, [devicePixelRatio])

  // Images skip both color layers, link text skips the ink layer. Shares the operator list the canvas render fetches.
  const collectRegions = useCallback((page: PdfPage) => {
    if (regionsRequestedRef.current) return
    regionsRequestedRef.current = true
    Promise.all([collectImageRects(page), collectLinkRects(page)])
      .then(([images, links]) => setRegions({ images, links }))
      .catch(() => { /* page without regions keeps plain layers */ })
  }, [])

  // The tint skips images, the ink layer also skips links, and the link layer caps link text at one link blue.
  const masks = useMemo(() => {
    if (!drawn) return null
    const images = regions?.images ?? []
    const links = regions?.links ?? []
    return { tint: rectMask(images, true), ink: rectMask([...images, ...links], true), link: rectMask(links, false) }
  }, [drawn, regions])

  const scale = width / rasterWidth
  return (
    <div
      ref={slotRef}
      className="pdf-page relative"
      style={{ width, height, marginBottom: PAGE_GAP, scrollSnapAlign: snap ? "start" : undefined }}
      data-page-number={pageNumber}
    >
      <div className="pdf-page-surface">
        {mounted && (
          <div style={{ width: rasterWidth, transformOrigin: "0 0", transform: scale === 1 ? undefined : `scale(${scale})` }}>
            <Page
              pageNumber={pageNumber}
              width={rasterWidth}
              devicePixelRatio={devicePixelRatio}
              canvasRef={canvasRef}
              loading={null}
              onLoadSuccess={collectRegions}
              onRenderSuccess={onRenderSuccess}
              renderTextLayer
            />
          </div>
        )}
        <canvas ref={backupRef} width={0} height={0} aria-hidden className="pdf-page-backup absolute inset-0 h-full w-full pointer-events-none" style={{ display: "none" }} />
        <div className="pdf-page-tint absolute inset-0 pointer-events-none" style={masks?.tint} />
        <div className="pdf-page-ink absolute inset-0 pointer-events-none" style={masks?.ink} />
        <div className="pdf-page-link absolute inset-0 pointer-events-none" style={masks?.link ?? { display: "none" }} />
        {drawn && regions?.images.map((r, k) => (
          <div
            key={k}
            className="pdf-image-mask absolute pointer-events-none"
            style={{ left: `${r.left * 100}%`, top: `${r.top * 100}%`, width: `${r.width * 100}%`, height: `${r.height * 100}%` }}
          />
        ))}
      </div>
      {renderPageOverlay?.(pageNumber, width)}
    </div>
  )
})

/** PDF viewer with zoom and page tracking.
 *
 * - `topInset` px of the viewer sit under a floating bar: pages start below it and scroll on underneath
 * - `renderPageOverlay` draws over each page, unclipped, given the page's rendered width in px
 * - `onScrollerChange` hands out the scrolling element, e.g. for pointer input over the pages
 * - Controls show only while the pointer is within CONTROLS_VICINITY px of them
 * - Only pages near the viewport hold a canvas, the rest are sized placeholders
 */
export function PdfViewer({
  url,
  fitWidth,
  onPageAspect,
  initialPage = 1,
  onPageChange,
  topInset = 0,
  renderPageOverlay,
  onScrollerChange,
}: {
  url: string
  // when set, the parent controls width and the viewer sizes its own height to
  // fit exactly one page. Otherwise it fills its parent in both dimensions.
  fitWidth?: boolean
  onPageAspect?: (ratio: number) => void
  initialPage?: number
  onPageChange?: (page: number) => void
  topInset?: number
  renderPageOverlay?: (page: number, pageWidth: number) => ReactNode
  onScrollerChange?: (scroller: HTMLDivElement | null) => void
}) {
  const [numPages, setNumPages] = useState<number | null>(null)
  const [pageAspects, setPageAspects] = useState<number[] | null>(null) // height / width per page
  const [nearPages, setNearPages] = useState<ReadonlySet<number>>(() => new Set())
  const documentRef = useRef<PdfDocument | null>(null)
  const rootRef = useRef<HTMLDivElement>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const [availWidth, setAvailWidth] = useState<number | null>(null)
  const [zoom, setZoom] = useState(1)
  const [rasterZoom, setRasterZoom] = useState(1)
  const zoomRef = useRef(1)
  const anchorRef = useRef<ScrollAnchor | null>(null)
  const [nearControls, setNearControls] = useState(false)
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const [lightPaper, setLightPaper] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)
  const restoredPageUrlRef = useRef<string | null>(null)
  const pageRef = useRef(1)
  const onPageChangeRef = useRef(onPageChange)
  const onScrollerChangeRef = useRef(onScrollerChange)
  useLayoutEffect(() => {
    onPageChangeRef.current = onPageChange
    onScrollerChangeRef.current = onScrollerChange
  })
  const setScroller = useCallback((scroller: HTMLDivElement | null) => {
    scrollRef.current = scroller
    onScrollerChangeRef.current?.(scroller)
  }, [])
  const pageRefs = useRef(new Map<number, HTMLDivElement>())
  const pageObserverRef = useRef<IntersectionObserver | null>(null)
  const devicePixelRatio = useDevicePixelRatio()
  const pageAspect = pageAspects?.[0] ?? null

  // Mount pages near the viewport. A hidden scroller (inactive tab) reports nothing visible, its pages stay as they are.
  useEffect(() => {
    const scroller = scrollRef.current
    if (!scroller) return
    const observer = new IntersectionObserver((entries) => {
      if (!entries[0]?.rootBounds?.height) return
      setNearPages((previous) => {
        const next = new Set(previous)
        for (const entry of entries) {
          const pageNumber = Number((entry.target as HTMLElement).dataset.pageNumber)
          if (entry.isIntersecting) next.add(pageNumber)
          else next.delete(pageNumber)
        }
        return next
      })
    }, { root: scroller, rootMargin: MOUNT_MARGIN })
    pageObserverRef.current = observer
    for (const slot of pageRefs.current.values()) observer.observe(slot)
    return () => {
      observer.disconnect()
      pageObserverRef.current = null
    }
  }, [])

  const registerSlot = useCallback((pageNumber: number, slot: HTMLDivElement | null) => {
    const previous = pageRefs.current.get(pageNumber)
    if (previous) pageObserverRef.current?.unobserve(previous)
    if (slot) {
      pageRefs.current.set(pageNumber, slot)
      pageObserverRef.current?.observe(slot)
    } else {
      pageRefs.current.delete(pageNumber)
    }
  }, [])

  // Re-rasterize once the zoom rests, until then the current bitmaps scale with it.
  useEffect(() => {
    const timer = setTimeout(() => setRasterZoom(zoom), ZOOM_SETTLE_MS)
    return () => clearTimeout(timer)
  }, [zoom])

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
  }, [])

  // Controls show while the pointer is near them, or while the zoom menu is open.
  const trackControlsVicinity = useCallback((event: React.MouseEvent) => {
    const rect = dropdownRef.current?.getBoundingClientRect()
    if (!rect) return
    const dx = Math.max(rect.left - event.clientX, 0, event.clientX - rect.right)
    const dy = Math.max(rect.top - event.clientY, 0, event.clientY - rect.bottom)
    setNearControls(Math.hypot(dx, dy) <= CONTROLS_VICINITY)
  }, [])
  const controlsVisible = nearControls || dropdownOpen

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
      documentRef.current = pdf
      // first page's real dimensions (any page size, not just A4) lay out every page until all are read
      let firstAspect = Math.SQRT2
      try {
        const viewport = (await pdf.getPage(1)).getViewport({ scale: 1 })
        firstAspect = viewport.height / viewport.width
        onPageAspect?.(firstAspect)
      } catch {
        /* lay out at A4 */
      }
      if (documentRef.current !== pdf) return
      setNumPages(pdf.numPages)
      setPageAspects(Array.from({ length: pdf.numPages }, () => firstAspect))
      const aspects = await readPageAspects(pdf, firstAspect)
      if (documentRef.current === pdf) setPageAspects(aspects)
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

  const ready = numPages != null && availWidth != null && pageAspects != null
  const pageWidth = Math.round(renderWidth * zoom)
  const rasterPageWidth = Math.round(renderWidth * rasterZoom)

  // Keep the document spot at the viewport's top edge fixed across width
  // changes (container resize). Pages scale with width but the PAGE_GAPs
  // don't, so map scrollTop exactly: page index + fraction within that page,
  // re-projected onto the new page height. Runs before paint, page slots have
  // their final heights, so the assignment can't be clamped away. Requires
  // overflow-anchor: none on the scroller, otherwise native scroll anchoring
  // makes its own correction and this one lands past the right spot.
  const prevRenderWidthRef = useRef(renderWidth)
  useLayoutEffect(() => {
    const prev = prevRenderWidthRef.current
    prevRenderWidthRef.current = renderWidth
    const el = scrollRef.current
    if (!el || prev === renderWidth || prev <= 0) return
    if (pageAspects?.length) {
      const oldTops = pageTops(pageAspects, Math.round(prev * zoom))
      const newTops = pageTops(pageAspects, pageWidth)
      let page = 0
      while (page + 1 < oldTops.length && oldTops[page + 1] <= el.scrollTop) page++
      const fraction = (el.scrollTop - oldTops[page]) / (Math.round(prev * zoom) * pageAspects[page])
      el.scrollTop = newTops[page] + fraction * pageWidth * pageAspects[page]
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
  }, [ready, url, initialPage, numPages, renderWidth, zoom, pageAspects])

  // Internal links (references within this PDF) jump to their page, which lands just below the floating bar.
  const goToPage = useCallback(({ pageNumber }: { pageNumber: number }) => {
    const el = scrollRef.current
    const firstPageEl = pageRefs.current.get(1)
    const pageEl = pageRefs.current.get(pageNumber)
    if (!el || !firstPageEl || !pageEl) return
    el.scrollTo({ top: pageEl.offsetTop - firstPageEl.offsetTop, behavior: "smooth" })
  }, [])

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
  const pageRightGap = availWidth == null ? 0 : (availWidth - pageWidth) / 2
  const controlsRight = Math.max(0, pageRightGap) + CONTROLS_PADDING

  return (
    <div
      ref={rootRef}
      className={`relative flex flex-col${lightPaper ? " pdf-light-paper" : ""}`}
      onMouseMove={trackControlsVicinity}
      onMouseLeave={() => setNearControls(false)}
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
          className="p-1.5 text-ink-subtle hover:text-ink hover:bg-hover disabled:text-ink-faint disabled:pointer-events-none transition-colors in-[.light]:rounded-r-lg"
        >
          <Plus className="h-3.5 w-3.5" strokeWidth={2} />
        </button>
        <div className="pdf-paper-toggle flex items-center">
          <div className="w-px h-4 bg-divider" />
          <button
            aria-label={lightPaper ? "Dark PDF pages" : "Light PDF pages"}
            onClick={() => switchColors(() => flushSync(() => setLightPaper((light) => !light)))}
            className="rounded-r-lg p-1.5 text-ink-subtle hover:text-ink hover:bg-hover transition-colors"
          >
            {lightPaper ? <Moon className="h-3.5 w-3.5" /> : <Sun className="h-3.5 w-3.5" />}
          </button>
        </div>
      </div>
      <div
        ref={setScroller}
        className="pdf-scroll flex-1 min-h-0 overflow-auto focus:outline-none"
        tabIndex={0}
        onScroll={trackVisiblePage}
        // overflow-anchor off: the width-change effect above owns the scroll
        // position; native scroll anchoring would apply a second correction
        style={{ overflowAnchor: "none", paddingTop: topInset, ...(fitWidth ? { scrollSnapType: "y proximity" as const } : undefined) }}
      >
        <Document file={url} onLoadSuccess={onDocumentLoadSuccess} onItemClick={goToPage} externalLinkTarget="_blank" externalLinkRel="noopener noreferrer">
          {ready && pageAspects && (
            <div style={{ width: pageWidth }} className="mx-auto">
              {pageAspects.map((aspect, index) => (
                <PdfPageSlot
                  key={`${url}:${index}`}
                  pageNumber={index + 1}
                  width={pageWidth}
                  height={Math.round(pageWidth * aspect)}
                  rasterWidth={rasterPageWidth}
                  devicePixelRatio={devicePixelRatio}
                  mounted={nearPages.has(index + 1)}
                  snap={!!fitWidth}
                  registerSlot={registerSlot}
                  renderPageOverlay={renderPageOverlay}
                />
              ))}
            </div>
          )}
        </Document>
      </div>
    </div>
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
