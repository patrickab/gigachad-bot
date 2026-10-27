"use client"

import { useCallback, useRef, type RefObject } from "react"
import { renderPageToPng, type CanvasEmbedRect } from "@/lib/canvasExport"
import type { CanvasDocument } from "@/lib/domain/canvasDocument"
import { imageFrameEmbeds } from "@/lib/domain/canvasRender"

const SCREENSHOT_MIN = 4 // canvas units — smaller drags are accidents

interface CanvasScreenshotOptions {
  liveRef: { readonly current: { doc: CanvasDocument } }
  containerRef: RefObject<HTMLDivElement | null>
  screenToCanvas: (clientX: number, clientY: number) => [number, number]
}

/** Rasterizes a canvas-space rectangle and copies it to the clipboard as PNG; only the latest capture writes. */
export function useCanvasScreenshot({ liveRef, containerRef, screenToCanvas }: CanvasScreenshotOptions) {
  const screenshotSequence = useRef(0)

  return useCallback(async ([x0, y0]: [number, number], [x1, y1]: [number, number]) => {
    const x = Math.min(x0, x1)
    const y = Math.min(y0, y1)
    const w = Math.abs(x1 - x0)
    const h = Math.abs(y1 - y0)
    if (w < SCREENSHOT_MIN || h < SCREENSHOT_MIN) return
    const sequence = ++screenshotSequence.current
    const { frames, strokes, texts } = liveRef.current.doc
    const images = imageFrameEmbeds(frames)
    // Rendered PDF pages are reused as already-decoded canvases.
    const pdfPages: CanvasEmbedRect[] = Array.from(
      containerRef.current?.querySelectorAll<HTMLCanvasElement>("[data-canvas-attachment] .react-pdf__Page__canvas") ?? [],
    ).flatMap((image) => {
      const rect = image.getBoundingClientRect()
      if (rect.width === 0 || rect.height === 0) return []
      const [left, top] = screenToCanvas(rect.left, rect.top)
      const [right, bottom] = screenToCanvas(rect.right, rect.bottom)
      return [{ image, x: left, y: top, width: right - left, aspect: (bottom - top) / (right - left) }]
    })
    try {
      const pngBytes = await renderPageToPng(strokes, x, y, w, h, images, texts, pdfPages)
      if (sequence !== screenshotSequence.current) return
      const blob = new Blob([pngBytes.buffer as ArrayBuffer], { type: "image/png" })
      await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })])
    } catch { /* clipboard write can fail without permission — silently drop */ }
  }, [liveRef, containerRef, screenToCanvas])
}
