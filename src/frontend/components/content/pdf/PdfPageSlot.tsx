"use client"

import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react"
import { Page } from "react-pdf"
import { mapCanvasToDevicePixels, releaseCanvas, renderPlaceholder, showPlaceholder, type RenderTask } from "./pdfCanvas"
import { PAGE_GAP, type PdfDocument, type PdfPage } from "./pdfLayout"
import { collectImageRects, collectLinkRects, rectMask, type PageRegions } from "./pdfPageRegions"

/** One page's box at its final size, its live canvas exists only while `mounted`.
 *
 * - `rasterWidth` is the canvas' render width, CSS-scaled to `width` while a zoom settles
 * - The backup canvas covers the page whenever the live canvas is blank: the last bitmap while re-rasterizing, a placeholder while unmounted
 * - `keep` pages hold a low-res placeholder, copied from the canvas on unmount or rendered while `rendersPlaceholder`
 * - `onPlaceholderChange` reports `settled`: a placeholder exists or its render was attempted
 */
export const PdfPageSlot = memo(function PdfPageSlot({
  pageNumber,
  width,
  height,
  rasterWidth,
  devicePixelRatio,
  mounted,
  keep,
  rendersPlaceholder,
  documentRef,
  onPlaceholderChange,
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
  keep: boolean
  rendersPlaceholder: boolean
  documentRef: { readonly current: PdfDocument | null }
  onPlaceholderChange: (pageNumber: number, settled: boolean) => void
  snap: boolean
  registerSlot: (pageNumber: number, slot: HTMLDivElement | null) => void
  renderPageOverlay?: (page: number, pageWidth: number) => ReactNode
}) {
  const [regions, setRegions] = useState<PageRegions | null>(null)
  // masks over a blank canvas would show as light boxes
  const [drawn, setDrawn] = useState(false)
  const [placeholder, setPlaceholder] = useState(false)
  if (!mounted && drawn) setDrawn(false)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const drawnCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const backupRef = useRef<HTMLCanvasElement>(null)
  const regionsRequestedRef = useRef(false)

  const slotRef = useCallback((slot: HTMLDivElement | null) => registerSlot(pageNumber, slot), [registerSlot, pageNumber])

  const publishPlaceholder = useCallback((hasBitmap: boolean, settled = hasBitmap) => {
    setPlaceholder(hasBitmap)
    onPlaceholderChange(pageNumber, settled)
  }, [onPlaceholderChange, pageNumber])

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

  // Same timing on unmount: the detached canvas still holds its bitmap, shrink it into the backup.
  useLayoutEffect(() => {
    if (mounted) return
    const drawnCanvas = drawnCanvasRef.current
    const backup = backupRef.current
    drawnCanvasRef.current = null
    if (!keep) releaseCanvas(backup)
    else if (drawnCanvas?.width && backup) showPlaceholder(backup, drawnCanvas)
    publishPlaceholder(!!backup?.width)
  }, [mounted, keep, publishPlaceholder])

  // Pages that never rendered get a placeholder from a scratch render, the live canvas takes over once mounted.
  useEffect(() => {
    const pdf = documentRef.current
    const backup = backupRef.current
    if (!rendersPlaceholder || mounted || !pdf || !backup) return
    let cancelled = false
    let task: RenderTask | null = null
    renderPlaceholder(pdf, pageNumber, (renderTask) => { task = renderTask })
      .then((scratch) => {
        if (cancelled) return
        showPlaceholder(backup, scratch)
        publishPlaceholder(true)
      })
      .catch(() => {
        // settled without a bitmap, so the scheduler moves on instead of retrying
        if (!cancelled) publishPlaceholder(false, true)
      })
    return () => {
      cancelled = true
      task?.cancel()
    }
  }, [rendersPlaceholder, mounted, documentRef, pageNumber, publishPlaceholder])

  const onRenderSuccess = useCallback(() => {
    mapCanvasToDevicePixels(canvasRef.current, devicePixelRatio)
    drawnCanvasRef.current = canvasRef.current
    releaseCanvas(backupRef.current)
    publishPlaceholder(false)
    setDrawn(true)
  }, [devicePixelRatio, publishPlaceholder])

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
    if (!drawn && !placeholder) return null
    const images = regions?.images ?? []
    const links = regions?.links ?? []
    return { tint: rectMask(images, true), ink: rectMask([...images, ...links], true), link: rectMask(links, false) }
  }, [drawn, placeholder, regions])

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
