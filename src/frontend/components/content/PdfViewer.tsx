"use client"

import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react"
import { Document } from "react-pdf"
import "react-pdf/dist/esm/Page/AnnotationLayer.css"
import "react-pdf/dist/esm/Page/TextLayer.css"
import { useEventListener } from "@/hooks/useEventListener"
import { PdfPageSlot } from "./pdf/PdfPageSlot"
import { PdfZoomControls } from "./pdf/PdfZoomControls"
import { useDevicePixelRatio } from "./pdf/pdfCanvas"
import { PAGE_GAP, PAGE_SIDE_GAP, pageTop, readPageAspects, RENDER_WIDTH, scrollTopAfterWidthChange, type PdfDocument } from "./pdf/pdfLayout"
import { clickedWord, doubleClickedWord, revealedY, type PdfWord } from "./pdf/pdfText"
import "./pdf/pdfjsWorker"
import { usePdfPageWindow } from "./pdf/usePdfPageWindow"
import { usePdfZoom } from "./pdf/usePdfZoom"

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

/** A spot to bring into view: `y` PDF points down from the top of 1-based `page`, or the middle of the first text line of the page holding one of `words`. A new `key` reveals it again. */
export type PdfReveal = { page: number; y: number; words?: readonly string[]; key: number }

/** Share of the scroller's height above a revealed spot. */
const REVEAL_SHARE_ABOVE = 1 / 3
/** How long a revealed spot stays marked, its mark fades out over that time. */
const REVEAL_MARK_MS = 1500

/** A loaded document, its url and its pages' height / width ratios. */
type PdfLayout = { url: string; pdf: PdfDocument; aspects: number[] }

/** Callback for the incoming layer, whose pages stay out of the page window until they swap in. */
function ignore() {}

/** PDF viewer with zoom and page tracking.
 *
 * - `topInset` px of the viewer sit under a floating bar: pages start below it and scroll on underneath
 * - `renderPageOverlay` draws over each page, unclipped, given the page's rendered width in px and the document's page count
 * - `onScrollerChange` hands out the scrolling element, e.g. for pointer input over the pages
 * - Controls show only while the pointer is near them
 * - `documentKey` names the document apart from its `url`: a new `url` under the same key loads behind the shown pages and swaps in once its pages near the viewport have drawn, keeping zoom and scroll position, while a new key starts over
 * - `onPageRendered` gets the shown url each time one of its pages finished drawing, or failed to, and when a new url swaps in
 * - `pageBox` sets the page width and the space around the pages, e.g. to line them up with a neighboring pane
 * - `onPageDoubleClick` gets the double-clicked page, the spot on it in PDF points from its top-left corner at any zoom, and the word the double-click selected there, else the page's word nearest the spot (empty when none is near), with the clicked character's offset in it and its occurrence on its rendered line
 * - `reveal` scrolls its spot to a third of the scroller's height once the document has laid out and marks it briefly, a new `key` repeats it
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
  onPageDoubleClick,
  reveal,
}: {
  url: string
  // when set, the parent controls width and the viewer sizes its own height to
  // fit exactly one page. Otherwise it fills its parent in both dimensions.
  fitWidth?: boolean
  onPageAspect?: (ratio: number) => void
  initialPage?: number
  onPageChange?: (page: number) => void
  topInset?: number
  renderPageOverlay?: (page: number, pageWidth: number, pageCount: number) => ReactNode
  onScrollerChange?: (scroller: HTMLDivElement | null) => void
  documentKey?: string
  onPageRendered?: (url: string) => void
  pageBox?: PdfPageBox
  onPageDoubleClick?: (page: number, x: number, y: number, word: PdfWord) => void
  reveal?: PdfReveal
}) {
  const identity = documentKey ?? url
  // The url on screen. Under the same `documentKey` a new url is the incoming one, loading hidden until it swaps in
  const [shownUrl, setShownUrl] = useState(url)
  const [shownIdentity, setShownIdentity] = useState(identity)
  const [layout, setLayout] = useState<PdfLayout | null>(null)
  const [incoming, setIncoming] = useState<PdfLayout | null>(null)
  // Incoming pages that finished drawing
  const [incomingDrawn, setIncomingDrawn] = useState<ReadonlySet<number>>(() => new Set())
  // The last layout that swapped in, its callbacks fire after the commit
  const [swapped, setSwapped] = useState<PdfLayout | null>(null)
  // The document each layer loaded last, so a superseded load never lays out
  const shownLoadRef = useRef<PdfDocument | null>(null)
  const incomingLoadRef = useRef<PdfDocument | null>(null)
  if (url !== shownUrl && !(documentKey !== undefined && identity === shownIdentity && layout?.url === shownUrl)) setShownUrl(url)
  if (identity !== shownIdentity) setShownIdentity(identity)
  const incomingUrl = url === shownUrl ? null : url
  // A url change while a swap is pending replaces the pending one
  const [incomingFor, setIncomingFor] = useState<string | null>(null)
  if (incomingFor !== incomingUrl) {
    setIncomingFor(incomingUrl)
    setIncoming(null)
    setIncomingDrawn(new Set())
  }
  useLayoutEffect(() => {
    incomingLoadRef.current = null
  }, [incomingUrl])
  const numPages = layout?.aspects.length ?? null
  const pageAspects = layout?.aspects ?? null // height / width per page
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
  const onPageAspectRef = useRef(onPageAspect)
  const onPageDoubleClickRef = useRef(onPageDoubleClick)
  useLayoutEffect(() => {
    onPageChangeRef.current = onPageChange
    onScrollerChangeRef.current = onScrollerChange
    onPageRenderedRef.current = onPageRendered
    onPageAspectRef.current = onPageAspect
    onPageDoubleClickRef.current = onPageDoubleClick
  })
  const notifyPageRendered = useCallback(() => onPageRenderedRef.current?.(shownUrl), [shownUrl])
  const onIncomingPageRendered = useCallback((pageNumber: number) => {
    setIncomingDrawn((drawn) => (drawn.has(pageNumber) ? drawn : new Set(drawn).add(pageNumber)))
  }, [])
  const setScroller = useCallback((scroller: HTMLDivElement | null) => {
    scrollRef.current = scroller
    onScrollerChangeRef.current?.(scroller)
  }, [])
  const devicePixelRatio = useDevicePixelRatio()
  const pageAspect = pageAspects?.[0] ?? null
  const ready = numPages != null && availWidth != null && pageAspects != null
  const { nearPages, keepPages, slotsRef, registerSlot, onPlaceholderChange, activePlaceholderTarget, markScrolled } = usePdfPageWindow(scrollRef, ready, pageRef)

  // Swap once every incoming page near the viewport has drawn: the incoming layer becomes the shown one in this render
  if (incoming !== null && incoming.url === incomingUrl && [...nearPages].every((page) => page > incoming.aspects.length || incomingDrawn.has(page))) {
    setShownUrl(incoming.url)
    setLayout(incoming)
    setIncoming(null)
    setSwapped(incoming)
  }
  useEffect(() => {
    if (!swapped) return
    onPageAspectRef.current?.(swapped.aspects[0])
    onPageRenderedRef.current?.(swapped.url)
  }, [swapped])

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

  const loadDocument = useCallback(async (layerUrl: string, incomingLayer: boolean, pdf: PdfDocument) => {
    const loadRef = incomingLayer ? incomingLoadRef : shownLoadRef
    loadRef.current = pdf
    // first page's real dimensions (any page size, not just A4) lay out every page until all are read
    let firstAspect = Math.SQRT2
    try {
      const viewport = (await pdf.getPage(1)).getViewport({ scale: 1 })
      firstAspect = viewport.height / viewport.width
      if (!incomingLayer && loadRef.current === pdf) onPageAspectRef.current?.(firstAspect)
    } catch {
      /* lay out at A4 */
    }
    if (loadRef.current !== pdf) return
    const loaded = { url: layerUrl, pdf, aspects: Array.from({ length: pdf.numPages }, () => firstAspect) }
    if (incomingLayer) setIncoming(loaded)
    else setLayout(loaded)
    const aspects = await readPageAspects(pdf, firstAspect)
    // the layer may have swapped in meanwhile
    const withAspects = (current: PdfLayout | null) => (current?.pdf === pdf ? { ...current, aspects } : current)
    setLayout(withAspects)
    setIncoming(withAspects)
  }, [])

  // An incoming url that failed to load swaps in right away, showing its error
  const showFailedLoad = useCallback((failedUrl: string) => {
    setShownUrl(failedUrl)
    setLayout(null)
    setIncoming(null)
  }, [])

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

  // The spot's share of the page box in CSS px, which a pinch transform or the canvas' device pixels do not change, scaled to the page's size in points
  const shownPdf = layout?.pdf
  const reportPageDoubleClick = useCallback(async (event: React.MouseEvent) => {
    const slot = (event.target as Element).closest<HTMLElement>(".pdf-page[data-page-number]")
    if (!onPageDoubleClickRef.current || !slot || !shownPdf) return
    const page = Number(slot.dataset.pageNumber)
    const box = slot.getBoundingClientRect()
    if (box.width <= 0 || box.height <= 0) return
    const across = (event.clientX - box.left) / box.width
    const down = (event.clientY - box.top) / box.height
    // The browser selects the word on the second press, before the dblclick event. A press between lines or words selects only a blank
    const selected = doubleClickedWord(slot, event.clientX)
    try {
      const pdfPage = await shownPdf.getPage(page)
      const { width, height } = pdfPage.getViewport({ scale: 1 })
      const [x, y] = [across * width, down * height]
      onPageDoubleClickRef.current?.(page, x, y, await clickedWord(pdfPage, x, y, selected))
    } catch {
      /* the document went away meanwhile */
    }
  }, [shownPdf])

  // The revealed spot's mark: its page and its share of the page's height
  const [revealMark, setRevealMark] = useState<{ key: number; page: number; down: number } | null>(null)
  const revealedKeyRef = useRef<number | null>(null)
  // Placeholders have their final heights, so the spot's offset follows from the page sizes whether its page drew yet or not
  useEffect(() => {
    const el = scrollRef.current
    if (!reveal || !ready || !shownPdf || !pageAspects || !el || revealedKeyRef.current === reveal.key) return
    const page = Math.min(pageAspects.length, Math.max(1, Math.round(reveal.page)))
    let cancelled = false
    shownPdf.getPage(page).then(async (pdfPage) => {
      const y = await revealedY(pdfPage, reveal.y, reveal.words)
      if (cancelled) return
      const { height } = pdfPage.getViewport({ scale: 1 })
      const down = height > 0 ? Math.min(1, Math.max(0, y / height)) : 0
      const spot = (pageBox?.top ?? topInset) + pageTop(pageAspects, pageWidth, page) + down * Math.round(pageWidth * pageAspects[page - 1])
      el.scrollTop = Math.max(0, spot - el.clientHeight * REVEAL_SHARE_ABOVE)
      revealedKeyRef.current = reveal.key
      setRevealMark({ key: reveal.key, page, down })
    }, () => {
      /* the document went away meanwhile */
    })
    return () => { cancelled = true }
  }, [reveal, ready, shownPdf, pageAspects, pageWidth, pageBox?.top, topInset])
  useEffect(() => {
    if (!revealMark) return
    const timer = setTimeout(() => setRevealMark(null), REVEAL_MARK_MS)
    return () => clearTimeout(timer)
  }, [revealMark])
  const pageOverlay = useCallback((page: number, width: number) => (
    <>
      {renderPageOverlay?.(page, width, numPages ?? 0)}
      {revealMark?.page === page && (
        <span
          key={revealMark.key}
          aria-hidden
          className="pdf-reveal-mark pointer-events-none absolute inset-x-0 h-6 -translate-y-1/2 bg-ink-subtle/25 opacity-0 transition-opacity duration-[1500ms] ease-out starting:opacity-100"
          style={{ top: `${revealMark.down * 100}%` }}
        />
      )}
    </>
  ), [renderPageOverlay, revealMark, numPages])

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
        onDoubleClick={(event) => void reportPageDoubleClick(event)}
        // overflow-anchor off: the width-change effect above owns the scroll
        // position; native scroll anchoring would apply a second correction
        style={{
          overflowAnchor: "none",
          paddingTop: pageBox?.top ?? topInset,
          paddingBottom: pageBox && Math.max(0, pageBox.bottom - PAGE_GAP),
          ...(fitWidth ? { scrollSnapType: "y proximity" as const } : undefined),
        }}
      >
        {/* Each url gets its own Document: pages kept across a url change would render against the destroyed old one */}
        {(incomingUrl === null ? [shownUrl] : [shownUrl, incomingUrl]).map((layerUrl) => {
          const incomingLayer = layerUrl === incomingUrl
          const loaded = incomingLayer ? incoming : ready ? layout : null
          const aspects = availWidth != null && loaded?.url === layerUrl ? loaded.aspects : null
          return (
            // The incoming layer draws its pages out of sight, swapping in only flips this wrapper and drops the shown one
            <div key={layerUrl} inert={incomingLayer} className={incomingLayer ? "invisible h-0 overflow-hidden" : undefined}>
              <Document
                file={layerUrl}
                onLoadSuccess={(pdf) => loadDocument(layerUrl, incomingLayer, pdf)}
                onLoadError={incomingLayer ? () => showFailedLoad(layerUrl) : undefined}
                onItemClick={goToPage}
                externalLinkTarget="_blank"
                externalLinkRel="noopener noreferrer"
                loading={documentKey === undefined ? undefined : null}
              >
                {loaded && aspects && (
                  <div style={{ width: pageWidth }} className="mx-auto">
                    {aspects.map((aspect, index) => (
                      <PdfPageSlot
                        key={`${identity}:${index}`}
                        pageNumber={index + 1}
                        width={pageWidth}
                        height={Math.round(pageWidth * aspect)}
                        rasterWidth={rasterPageWidth}
                        devicePixelRatio={devicePixelRatio}
                        mounted={nearPages.has(index + 1)}
                        keep={!incomingLayer && keepPages.has(index + 1)}
                        rendersPlaceholder={!incomingLayer && activePlaceholderTarget === index + 1}
                        pdf={loaded.pdf}
                        onPlaceholderChange={incomingLayer ? ignore : onPlaceholderChange}
                        snap={!!fitWidth}
                        registerSlot={incomingLayer ? ignore : registerSlot}
                        renderPageOverlay={pageOverlay}
                        onRendered={incomingLayer ? onIncomingPageRendered : notifyPageRendered}
                      />
                    ))}
                  </div>
                )}
              </Document>
            </div>
          )
        })}
      </div>
    </div>
  )
}
