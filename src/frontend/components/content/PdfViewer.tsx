"use client"

import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react"
import { Document, Page, pdfjs } from "react-pdf"
import "react-pdf/dist/esm/Page/AnnotationLayer.css"
import "react-pdf/dist/esm/Page/TextLayer.css"
import { useEventListener } from "@/hooks/useEventListener"
import { PdfPageSlot } from "./pdf/PdfPageSlot"
import { PdfZoomControls } from "./pdf/PdfZoomControls"
import { useDevicePixelRatio } from "./pdf/pdfCanvas"
import { PAGE_GAP, PAGE_SIDE_GAP, readPageAspects, RENDER_WIDTH, scrollTopAfterWidthChange, type PdfDocument } from "./pdf/pdfLayout"
import { usePdfPageWindow } from "./pdf/usePdfPageWindow"
import { usePdfZoom } from "./pdf/usePdfZoom"

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

/** Page layout set by the parent: page `width`, and the scroller's px above the first page and below the last. */
export type PdfPageBox = { width: number; top: number; bottom: number }

/** PDF viewer with zoom and page tracking.
 *
 * - `topInset` px of the viewer sit under a floating bar: pages start below it and scroll on underneath
 * - `renderPageOverlay` draws over each page, unclipped, given the page's rendered width in px
 * - `onScrollerChange` hands out the scrolling element, e.g. for pointer input over the pages
 * - Controls show only while the pointer is near them
 * - `documentKey` names the document apart from its `url`: a new `url` under the same key reloads in place, keeping zoom and scroll position, while a new key starts over
 * - `onPageRendered` fires each time a page finished drawing
 * - `pageBox` sets the page width and the space around the pages, e.g. to line them up with a neighboring pane
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
  documentKey,
  onPageRendered,
  pageBox,
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
  documentKey?: string
  onPageRendered?: () => void
  pageBox?: PdfPageBox
}) {
  const identity = documentKey ?? url
  const [numPages, setNumPages] = useState<number | null>(null)
  const [pageAspects, setPageAspects] = useState<number[] | null>(null) // height / width per page
  const documentRef = useRef<PdfDocument | null>(null)
  const rootRef = useRef<HTMLDivElement>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const [availWidth, setAvailWidth] = useState<number | null>(null)
  // Both scrollbar gutters together, 0 for overlay scrollbars
  const [scrollbarGutters, setScrollbarGutters] = useState(0)
  const [lightPaper, setLightPaper] = useState(false)
  const restoredPageUrlRef = useRef<string | null>(null)
  const pageRef = useRef(1)
  const onPageChangeRef = useRef(onPageChange)
  const onScrollerChangeRef = useRef(onScrollerChange)
  const onPageRenderedRef = useRef(onPageRendered)
  useLayoutEffect(() => {
    onPageChangeRef.current = onPageChange
    onScrollerChangeRef.current = onScrollerChange
    onPageRenderedRef.current = onPageRendered
  })
  const notifyPageRendered = useCallback(() => onPageRenderedRef.current?.(), [])
  // The pages' height, held as a blank spacer while a reload under the same `documentKey` loads, so the scroll position survives
  const pagesRef = useRef<HTMLDivElement>(null)
  const pagesHeightRef = useRef(0)
  useLayoutEffect(() => {
    if (documentKey !== undefined && pagesRef.current) pagesHeightRef.current = pagesRef.current.offsetHeight
  })
  const holdPagesHeight = useCallback((spacer: HTMLDivElement | null) => {
    if (spacer) spacer.style.height = `${pagesHeightRef.current}px`
  }, [])
  const setScroller = useCallback((scroller: HTMLDivElement | null) => {
    scrollRef.current = scroller
    onScrollerChangeRef.current?.(scroller)
  }, [])
  const devicePixelRatio = useDevicePixelRatio()
  const pageAspect = pageAspects?.[0] ?? null
  const ready = numPages != null && availWidth != null && pageAspects != null
  const { nearPages, keepPages, slotsRef, registerSlot, onPlaceholderChange, activePlaceholderTarget, markScrolled } = usePdfPageWindow(scrollRef, ready, pageRef)

  // measure the parent-controlled dimension — debounced so resizing doesn't re-render every frame
  useEffect(() => {
    const el = rootRef.current
    if (!el) return

    let timer: ReturnType<typeof setTimeout> | null = null
    const update = () => {
      if (el.clientWidth > 0) setAvailWidth(el.clientWidth)
      const scroller = scrollRef.current
      if (scroller) setScrollbarGutters(scroller.offsetWidth - scroller.clientWidth)
    }

    update()

    const ro = new ResizeObserver(() => {
      if (timer) clearTimeout(timer)
      timer = setTimeout(update, 150)
    })
    ro.observe(el)
    return () => { ro.disconnect(); if (timer) clearTimeout(timer) }
  }, [])

  const zoomState = usePdfZoom(scrollRef)
  const { zoom, rasterZoom } = zoomState

  // arrow keys step one page from whichever page is currently nearest the top
  useEventListener(scrollRef, "keydown", (e) => {
    if (!numPages) return
    const el = e.currentTarget as HTMLElement
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
  })

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
  }, [identity])

  // Page is rendered at the available width (minus scrollbar gutter), capped at
  // RENDER_WIDTH. In fitWidth mode the box height is derived to show one page.
  let renderWidth = RENDER_WIDTH
  let boxHeight: number | undefined
  if (availWidth != null) {
    renderWidth = Math.round(Math.min(pageBox?.width ?? RENDER_WIDTH, availWidth - scrollbarGutters - 2 * PAGE_SIDE_GAP))
    if (fitWidth && pageAspect != null) {
      boxHeight = Math.round(renderWidth * pageAspect)
    }
  }

  const pageWidth = Math.round(renderWidth * zoom)
  const rasterPageWidth = Math.round(renderWidth * rasterZoom)

  // Keep the document spot at the viewport's top edge fixed across width
  // changes (container resize). Runs before paint, page slots have their final
  // heights, so the assignment can't be clamped away. Requires
  // overflow-anchor: none on the scroller, otherwise native scroll anchoring
  // makes its own correction and this one lands past the right spot.
  const prevRenderWidthRef = useRef(renderWidth)
  useLayoutEffect(() => {
    const prev = prevRenderWidthRef.current
    prevRenderWidthRef.current = renderWidth
    const el = scrollRef.current
    if (!el || prev === renderWidth || prev <= 0) return
    if (pageAspects?.length) el.scrollTop = scrollTopAfterWidthChange(el.scrollTop, pageAspects, Math.round(prev * zoom), pageWidth)
    else el.scrollTop *= renderWidth / prev
    el.scrollLeft *= renderWidth / prev
    // eslint-disable-next-line react-hooks/exhaustive-deps -- reposition only when the width actually changes
  }, [renderWidth])

  // Scroll to `initialPage` once per url, after layout is ready.
  useLayoutEffect(() => {
    const el = scrollRef.current
    const page = Math.min(numPages ?? 1, Math.max(1, initialPage))
    const firstPageEl = slotsRef.current.get(1)
    const pageEl = slotsRef.current.get(page)
    if (!el || !ready || !firstPageEl || !pageEl || restoredPageUrlRef.current === identity) return
    // offsetTop is in the scroller's layout coordinate system, unlike client rects
    // after the canvas scales this attachment with CSS.
    el.scrollTop = pageEl.offsetTop - firstPageEl.offsetTop
    pageRef.current = page
    restoredPageUrlRef.current = identity
  }, [ready, identity, initialPage, numPages, renderWidth, zoom, pageAspects, slotsRef])

  // Internal links (references within this PDF) jump to their page, which lands just below the floating bar.
  const goToPage = useCallback(({ pageNumber }: { pageNumber: number }) => {
    const el = scrollRef.current
    const firstPageEl = slotsRef.current.get(1)
    const pageEl = slotsRef.current.get(pageNumber)
    if (!el || !firstPageEl || !pageEl) return
    el.scrollTo({ top: pageEl.offsetTop - firstPageEl.offsetTop, behavior: "smooth" })
  }, [slotsRef])

  const trackVisiblePage = useCallback(() => {
    const el = scrollRef.current
    if (!el || numPages == null) return
    const top = el.getBoundingClientRect().top + topInset + 1
    let page = 1
    for (const [number, pageEl] of slotsRef.current) {
      if (pageEl.getBoundingClientRect().top > top) break
      page = number
    }
    if (page === pageRef.current) return
    pageRef.current = page
    onPageChangeRef.current?.(page)
  }, [numPages, topInset, slotsRef])

  // fitWidth: fill parent width, self-size height to one page. Otherwise fill parent.
  const rootStyle: React.CSSProperties = fitWidth
    ? { width: "100%", height: boxHeight ?? "auto" }
    : { width: "100%", height: "100%", minHeight: 0 }

  return (
    <div
      ref={rootRef}
      className={`relative flex flex-col${lightPaper ? " pdf-light-paper" : ""}`}
      style={rootStyle}
    >
      <PdfZoomControls
        hostRef={rootRef}
        zoom={zoomState}
        topInset={topInset}
        // pages are centered in the scroller
        pageRightGap={availWidth == null ? 0 : (availWidth - pageWidth) / 2}
        lightPaper={lightPaper}
        setLightPaper={setLightPaper}
      />
      <div
        ref={setScroller}
        // A page box sets the page width exactly, a scrollbar gutter would narrow the pages
        className={`pdf-scroll flex-1 min-h-0 overflow-auto focus:outline-none ${pageBox ? "[scrollbar-width:none]" : "[scrollbar-gutter:stable_both-edges]"}`}
        tabIndex={0}
        onScroll={() => {
          markScrolled()
          trackVisiblePage()
        }}
        // overflow-anchor off: the width-change effect above owns the scroll
        // position; native scroll anchoring would apply a second correction
        style={{
          overflowAnchor: "none",
          paddingTop: pageBox?.top ?? topInset,
          paddingBottom: pageBox && Math.max(0, pageBox.bottom - PAGE_GAP),
          ...(fitWidth ? { scrollSnapType: "y proximity" as const } : undefined),
        }}
      >
        {/* A new url gets a fresh Document: pages kept across the swap would render against the destroyed old one */}
        <Document key={url} file={url} onLoadSuccess={onDocumentLoadSuccess} onItemClick={goToPage} externalLinkTarget="_blank" externalLinkRel="noopener noreferrer" loading={documentKey === undefined ? undefined : <div ref={holdPagesHeight} />}>
          {ready && pageAspects && (
            <div ref={pagesRef} style={{ width: pageWidth }} className="mx-auto">
              {pageAspects.map((aspect, index) => (
                <PdfPageSlot
                  key={`${identity}:${index}`}
                  pageNumber={index + 1}
                  width={pageWidth}
                  height={Math.round(pageWidth * aspect)}
                  rasterWidth={rasterPageWidth}
                  devicePixelRatio={devicePixelRatio}
                  mounted={nearPages.has(index + 1)}
                  keep={keepPages.has(index + 1)}
                  rendersPlaceholder={activePlaceholderTarget === index + 1}
                  documentRef={documentRef}
                  onPlaceholderChange={onPlaceholderChange}
                  snap={!!fitWidth}
                  registerSlot={registerSlot}
                  renderPageOverlay={renderPageOverlay}
                  onRendered={notifyPageRendered}
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
