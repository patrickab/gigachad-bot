"use client"

import { useSyncExternalStore } from "react"
import type { PdfDocument, PdfPage } from "./pdfLayout"

const PLACEHOLDER_WIDTH = 480

export type RenderTask = ReturnType<PdfPage["render"]>

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
export function useDevicePixelRatio(): number {
  return useSyncExternalStore(subscribeDevicePixelRatio, () => window.devicePixelRatio || 1, () => 1)
}

/** Size the canvas to exactly bitmap / DPR so every bitmap pixel lands on one device pixel.
 *
 * - react-pdf floors the CSS size independently of the truncated bitmap size
 * - At fractional DPRs (1.25, 1.5) that mismatch makes the browser resample, i.e. blur
 */
export function mapCanvasToDevicePixels(canvas: HTMLCanvasElement | null | undefined, devicePixelRatio: number) {
  if (!canvas) return
  canvas.style.width = `${canvas.width / devicePixelRatio}px`
  canvas.style.height = `${canvas.height / devicePixelRatio}px`
}

/** Free a canvas' bitmap and hide it. */
export function releaseCanvas(canvas: HTMLCanvasElement | null) {
  if (!canvas) return
  canvas.width = 0
  canvas.height = 0
  canvas.style.display = "none"
}

/** Draw `source` into `target` scaled down to at most PLACEHOLDER_WIDTH px wide, and show it. */
export function showPlaceholder(target: HTMLCanvasElement, source: HTMLCanvasElement) {
  const scale = Math.min(1, PLACEHOLDER_WIDTH / source.width)
  target.width = Math.max(1, Math.round(source.width * scale))
  target.height = Math.max(1, Math.round(source.height * scale))
  const context = target.getContext("2d")
  if (!context) return
  context.imageSmoothingQuality = "high"
  context.drawImage(source, 0, 0, target.width, target.height)
  target.style.display = ""
}

/** Rasterize a page at PLACEHOLDER_WIDTH into a scratch canvas, `onRenderTask` hands out the task for cancelling. */
export async function renderPlaceholder(pdf: PdfDocument, pageNumber: number, onRenderTask: (task: RenderTask | null) => void): Promise<HTMLCanvasElement> {
  const page = await pdf.getPage(pageNumber)
  const viewport = page.getViewport({ scale: PLACEHOLDER_WIDTH / page.getViewport({ scale: 1 }).width })
  const scratch = document.createElement("canvas")
  scratch.width = Math.ceil(viewport.width)
  scratch.height = Math.ceil(viewport.height)
  const canvasContext = scratch.getContext("2d")
  if (!canvasContext) throw new Error("2d canvas context unavailable")
  const task = page.render({ canvasContext, viewport })
  onRenderTask(task)
  await task.promise
  onRenderTask(null)
  return scratch
}
